# cff_punctions.py

from pathlib import Path
import logging
import textwrap
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import re
from typing import List


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_lca_data(FilePath: Path) -> pd.DataFrame:
    """
    Load one LCA.csv with a two-row header and process-name index,
    Return a DataFrame with MultiIndex columns (method, year).

    """
    if not FilePath.exists():
        logging.error(f"Missing LCA file: {FilePath}")
        raise FileNotFoundError(FilePath)

    df = pd.read_csv(FilePath, header=[0, 1], index_col=0)
    df.columns.names = ['method', 'scenario_raw']

    years = (
        df.columns
        .get_level_values('scenario_raw')
        .str.rsplit(' - ', n=1)
        .str[-1]
        .astype(int)
    )   

    methods = df.columns.get_level_values('method')
    df.columns = pd.MultiIndex.from_arrays(
        [methods, years],
        names=['method', 'year']
    )

    logging.info(f"Loaded {FilePath.name}: methods={methods.unique()} years={sorted(years.unique())}")
    return df



def interpolate_to_annual(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a MultiIndexed LCA DataFrame, interpolate linearly to every year,
    Returns a long-form DataFrame with columns: process, method, year, value.

    """
    methods = df.columns.get_level_values('method').unique()
    y_min = df.columns.get_level_values('year').min()
    y_max = df.columns.get_level_values('year').max()
    full_years = list(range(y_min, y_max + 1))

    records = []

    for method in methods:
        df_m = df.xs(method, axis=1, level='method').reindex(columns=full_years)
        df_int = df_m.interpolate(axis=1)
        df_long = (
            df_int
              .stack()
              .rename('value')
              .reset_index()
              .rename(columns={'level_0':'process','level_1':'year'})
        )
        df_long['method'] = method
        records.append(df_long)

    result = pd.concat(records, ignore_index=True)[['process','method','year','value']]
    
    logging.info(f"Interpolated to annual frequency: years {y_min}-{y_max}")
    return result


def compute_footprints_all(  
    df_lca, df_mfa, methods, rcps, scenarios=('BAU','Moderate','Aggressive'),
    A=0.2, km_per_vehicle=50000, occupancy=10
):
    """
    Compute Annual (AEF) and Cumulative (CEF) footprints for each combination of Method × Scenario × RCP,
    Returns a DataFrame with columns: method, RCP, scenario, year, AEF, CEF.
    
    """
    # Map processes to numeric IDs index
    all_procs = df_lca['process'].unique()
    normalized = [re.sub(r'\s+', ' ', proc) for proc in all_procs]
    
    # Define patterns in the exact order you want IDs assigned:
    patterns = {
        'prod_tot':     r'passenger\s+bus\s*\|\s*P1_e_bus_production\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'opr_tot':      r'transport,\s*passenger\s+bus\s*\|\s*P2_e_bus_operation\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'eol_tot':      r'used\s+bus\s*\|\s*P3_e_bus_eol_without_battery\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'eol_credit':   r'used\s+Li-ion\s+battery\s*\|\s*P4_LFP_eol\s*\|\s*GLO\s*\|\s*LFP_e_bus',
        'v_li':         r'lithium\s+carbonate\s*\|\s*P5_Li_virgin_production\s*\|\s*RoW\s*\|\s*LFP_e_bus',
        'mat_recovery': r'used\s+LFP\s+battery\s*\|\s*P6_LFP_recycling\s*\|\s*GLO\s*\|\s*LFP_e_bus',
        'rec_li':       r'lithium\s+carbonate\s*\|\s*P7_Li_recovery\s*\|\s*GLO\s*\|\s*LFP_e_bus',
    }

    # 1) match names in order
    name_map = {}
    ordered_procs = []
    for key, pat in patterns.items():
        matches = [
            raw for raw, norm in zip(all_procs, normalized)
            if re.search(pat, norm, flags=re.IGNORECASE)
        ]
        if not matches:
            raise KeyError(f"No process for {key!r}")
        if len(matches) > 1:
            raise KeyError(f"Ambiguous matches for {key!r}: {matches}")
        proc_name = matches[0]
        name_map[key] = proc_name
        ordered_procs.append(proc_name)

    # 2) enumerate in that forced order
    proc_idx = { name: idx for idx, name in enumerate(ordered_procs, start=1) }
    df_lca['proc_id'] = df_lca['process'].map(proc_idx)

    # DEBUG
    logging.info("Process → ID (forced ordering):")
    for name, idx in proc_idx.items():
        logging.info(f"  {idx}: {name!r}")

    # 3) build id_map for easy look-up
    id_map = { key: proc_idx[name_map[key]] for key in patterns }

    # Identify the needed processes by name
    idx_prod_tot     = id_map['prod_tot']
    idx_opr_tot      = id_map['opr_tot']
    idx_eol_tot      = id_map['eol_tot']
    idx_eol_vir      = id_map['eol_credit']
    idx_v_li         = id_map['v_li']
    idx_mat_recovery = id_map['mat_recovery']
    idx_rec_li       = id_map['rec_li']

    # Loops
    records = []

    for rcp in rcps:
        sub = df_lca[df_lca['RCP']==rcp]
        for method in methods:
            pivot = (
                sub[sub['method']==method]
                .pivot(index='year', columns='proc_id', values='value')
                .sort_index()
            )
            years = pivot.index.intersection(df_mfa['Year'])
            pivot = pivot.loc[years]
            mfa   = df_mfa.set_index('Year').loc[years]

            # Sanity check
            for idx in (idx_prod_tot, idx_opr_tot, idx_eol_tot,
                        idx_eol_vir, idx_v_li, idx_rec_li):
                if idx not in pivot.columns:
                    raise ValueError(f"Process {idx} missing for {method}/{rcp}")
            
            if idx_mat_recovery not in pivot.columns:
                logging.warning(f"P[6] mat_recovery (idx {idx_mat_recovery}) not found; skipping it")

            P = {i: pivot[i] for i in pivot.columns}

            # Build a lowercase→actual map for this year's MFA columns
            col_map = {c.lower(): c for c in mfa.columns}

            for sce in scenarios:
                key_r1 = f'r1_{sce.lower()}'
                key_r2 = f'r2_{sce.lower()}'
                if key_r1 not in col_map or key_r2 not in col_map:
                    missing = [k for k in (key_r1, key_r2) if k not in col_map]
                    raise KeyError(f"MFA columns missing for scenarios {missing}")
                R1 = mfa[ col_map[key_r1] ] / 100
                R2 = mfa[ col_map[key_r2] ] / 100
                N_in  = mfa['Inflow'];     N_stock = mfa['Stock'];  N_out = mfa['Outflow']

                # Fix CFF_prd per Eq(2)
                Ev      = P[idx_v_li]
                Erec    = P[idx_rec_li]
                ErecEoL = P[idx_mat_recovery]
                CFF_prd = (1-R1)*Ev + R1*(A*Erec + (1-A)*Ev)

                # unchanged
                CFF_eol = (1-A)*R2*(ErecEoL - Ev)

                E_prd = P[idx_prod_tot] - Ev + CFF_prd
                E_opr = P[idx_opr_tot]
                E_eol = P[idx_eol_tot] + (1-R2) * P[idx_eol_vir] + CFF_eol

                num = N_in*E_prd + N_stock*E_opr + N_out*E_eol
                pkm = N_stock * km_per_vehicle * occupancy

                AEF = num / pkm
                CEF = num.cumsum() / pkm.cumsum()

                df_tmp = pd.DataFrame({
                  'method': method, 'RCP': rcp, 'scenario': sce,
                  'year': years, 'AEF': AEF, 'CEF': CEF
                })
                records.append(df_tmp)

    return pd.concat(records, ignore_index=True)


def plot_interpolated(
    df: pd.DataFrame,
    methods: List[str],
    rcps: List[str],
    figsize=(12,6),
    wrap_width=30
):
    for method in methods:
        fig, axes = plt.subplots(1, len(rcps), figsize=figsize, sharey=True)
        for ax, rcp in zip(axes, rcps):
            sub = df[(df['method']==method)&(df['RCP']==rcp)]
            pivot = sub.pivot(index='year',columns='process',values='value')
            for proc in pivot.columns:
                ax.plot(pivot.index, pivot[proc], label=proc)
            ax.set_title(textwrap.fill(f"Impact — {method} ({rcp})", wrap_width))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6,integer=True))
            ax.grid('--', alpha=0.7)
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        axes[0].set_ylabel('Impact')
        axes[-1].legend(title='Process', bbox_to_anchor=(1.05,1), loc='upper left')
        fig.tight_layout()
        plt.show()



def compute_footprints_with_components(
    df_lca, df_mfa, methods, rcps, scenarios=('BAU','Moderate','Aggressive'),
    A=0.2, km_per_vehicle=50000, occupancy=10
):
    """
    Same as compute_footprints_all, but also returns for each year:
      - N_in * E_prd
      - N_stock * E_opr
      - N_out * E_eol

    Returns a DataFrame with columns:
      method, RCP, scenario, year,
      N_in_E_prd, N_stock_E_opr, N_out_E_eol, 
      AEF, CEF
    """
    # Map processes to numeric IDs index
    all_procs = df_lca['process'].unique()
    normalized = [re.sub(r'\s+', ' ', proc) for proc in all_procs]
    
    # Define patterns in the exact order you want IDs assigned:
    patterns = {
        'prod_tot':     r'passenger\s+bus\s*\|\s*P1_e_bus_production\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'opr_tot':      r'transport,\s*passenger\s+bus\s*\|\s*P2_e_bus_operation\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'eol_tot':      r'used\s+bus\s*\|\s*P3_e_bus_eol_without_battery\s*\|\s*CH\s*\|\s*LFP_e_bus',
        'eol_credit':   r'used\s+Li-ion\s+battery\s*\|\s*P4_LFP_eol\s*\|\s*GLO\s*\|\s*LFP_e_bus',
        'v_li':         r'lithium\s+carbonate\s*\|\s*P5_Li_virgin_production\s*\|\s*RoW\s*\|\s*LFP_e_bus',
        'mat_recovery': r'used\s+LFP\s+battery\s*\|\s*P6_LFP_recycling\s*\|\s*GLO\s*\|\s*LFP_e_bus',
        'rec_li':       r'lithium\s+carbonate\s*\|\s*P7_Li_recovery\s*\|\s*GLO\s*\|\s*LFP_e_bus',
    }

    # 1) match names in order
    name_map = {}
    ordered_procs = []
    for key, pat in patterns.items():
        matches = [
            raw for raw, norm in zip(all_procs, normalized)
            if re.search(pat, norm, flags=re.IGNORECASE)
        ]
        if not matches:
            raise KeyError(f"No process for {key!r}")
        if len(matches) > 1:
            raise KeyError(f"Ambiguous matches for {key!r}: {matches}")
        proc_name = matches[0]
        name_map[key] = proc_name
        ordered_procs.append(proc_name)

    # 2) enumerate in that forced order
    proc_idx = { name: idx for idx, name in enumerate(ordered_procs, start=1) }
    df_lca['proc_id'] = df_lca['process'].map(proc_idx)

    # DEBUG
    logging.info("Process → ID (forced ordering):")
    for name, idx in proc_idx.items():
        logging.info(f"  {idx}: {name!r}")

    # 3) build id_map for easy look-up
    id_map = { key: proc_idx[name_map[key]] for key in patterns }

    # Identify the needed processes by name
    idx_prod_tot     = id_map['prod_tot']
    idx_opr_tot      = id_map['opr_tot']
    idx_eol_tot      = id_map['eol_tot']
    idx_eol_vir      = id_map['eol_credit']
    idx_v_li         = id_map['v_li']
    idx_mat_recovery = id_map['mat_recovery']
    idx_rec_li       = id_map['rec_li']

    # Loops
    records = []

    for rcp in rcps:
        sub = df_lca[df_lca['RCP']==rcp]
        for method in methods:
            pivot = (
                sub[sub['method']==method]
                .pivot(index='year', columns='proc_id', values='value')
                .sort_index()
            )
            years = pivot.index.intersection(df_mfa['Year'])
            pivot = pivot.loc[years]
            mfa   = df_mfa.set_index('Year').loc[years]

            # Sanity check
            for idx in (idx_prod_tot, idx_opr_tot, idx_eol_tot,
                        idx_eol_vir, idx_v_li, idx_rec_li):
                if idx not in pivot.columns:
                    raise ValueError(f"Process {idx} missing for {method}/{rcp}")
            
            if idx_mat_recovery not in pivot.columns:
                logging.warning(f"P[6] mat_recovery (idx {idx_mat_recovery}) not found; skipping it")

            P = {i: pivot[i] for i in pivot.columns}

            # Build a lowercase→actual map for this year's MFA columns
            col_map = {c.lower(): c for c in mfa.columns}

            for sce in scenarios:
                key_r1 = f'r1_{sce.lower()}'
                key_r2 = f'r2_{sce.lower()}'
                if key_r1 not in col_map or key_r2 not in col_map:
                    missing = [k for k in (key_r1, key_r2) if k not in col_map]
                    raise KeyError(f"MFA columns missing for scenarios {missing}")
                R1 = mfa[ col_map[key_r1] ] / 100
                R2 = mfa[ col_map[key_r2] ] / 100
                N_in  = mfa['Inflow'];     N_stock = mfa['Stock'];  N_out = mfa['Outflow']

                # Fix CFF_prd per Eq(2)
                Ev      = P[idx_v_li]
                Erec    = P[idx_rec_li]
                ErecEoL = P[idx_mat_recovery]
                CFF_prd = (1-R1)*Ev + R1*(A*Erec + (1-A)*Ev)

                # unchanged
                CFF_eol = (1-A)*R2*(ErecEoL - Ev)

                E_prd = P[idx_prod_tot] - Ev + CFF_prd
                E_opr = P[idx_opr_tot]
                E_eol = P[idx_eol_tot] + (1-R2) * P[idx_eol_vir] + CFF_eol

                # new component terms
                comp_prd = N_in * E_prd
                comp_opr = N_stock * E_opr
                comp_eol = N_out * E_eol

                num = comp_prd + comp_opr + comp_eol
                pkm = N_stock * km_per_vehicle * occupancy

                AEF = num / pkm
                CEF = num.cumsum() / pkm.cumsum()

                df_tmp = pd.DataFrame({
                    'method':   method,
                    'RCP':      rcp,
                    'scenario': sce,
                    'year':     years,
                    # ← new columns here:
                    'N_in_E_prd':     comp_prd,
                    'N_stock_E_opr':  comp_opr,
                    'N_out_E_eol':    comp_eol,
                    'AEF':      AEF,
                    'CEF':      CEF
                })
                records.append(df_tmp)

    return pd.concat(records, ignore_index=True)