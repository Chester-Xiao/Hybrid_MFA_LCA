"""
functions_ODYM.py
================

This module contains custom functions for the BEB's CRM MFA analysis. It provides functions for:
 - Calculating e-bus inflow using different growth curves.
 - Computing recycling rates for each CRM based on scenario targets.
 - Exporting pandas DataFrames to CSV in a results folder.
 - Plotting fleet dynamics and CRM trajectories.
 - (Additional) Calculating and plotting unrecovered (waste) CRM flows.

Author: Chester Xiao
Date: 2025-03
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Union
import numpy as np
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def bebs_inflow(year: int,
                start: int, 
                end: int,
                total: int, 
                mode: str='linear') -> float:
    """
    Calculate the number of new e-buses registered in a given year.
    From `start` to `end`, uses `linear` or other function.
    After `end`, return 0; downstream code will handle constant fleet.
                    
    """
    if year < start or year > end:
        return 0.0
    if mode == 'linear':
        num_years = end - start + 1
        return total / num_years
    elif mode == 'logistic':
        k = 0.5  # growth rate parameter; can be tuned
        mid = (start + end) / 2
        # Logistic function scaled so that the maximum inflow approximates total/num_years
        return (total / (end - start + 1)) / (1 + np.exp(-k * (year - mid)))
    else:
        raise ValueError("Unknown inflow mode.")


def generate_demand_share (years: List[int],
                           config: Dict[str, Dict[str, float]]
                           ) -> Dict[str, np.ndarray]:
    """
    Build recycled-share arrays for total lithium demand:
        share[scenario][i] = factor * (years[i] - years[0])
    
    """
    base_year = years[0]
    result: Dict[str, np.ndarray] = {}
    for scen, params in config.items():
        f = params['factor']
        result[scen] = np.array([f*(y-base_year) for y in years])
    return result


def generate_eol_recovery_share (years: List[int],
                                 config: Dict[str, Dict[str, float]]
                                 ) -> Dict[str, np.ndarray]:
    """
    Build EoL recovery-share arrays:
        share[scenario][i] = base + rate * (year[i] - years[0])

    """
    base_year = years[0]
    result: Dict[str, np.ndarray] = {}
    for scen, params in config.items():
        b, r = params['base'], params['rate']
        result[scen] = np.array([b + r*(y-base_year) for y in years])
    return result






def export_dataframe_to_csv(df, filename, results_folder):
    """
    Export a pandas DataFrame to a CSV file in the results folder.
    
    Parameters:
        df (pandas.DataFrame): DataFrame to export.
        filename (str):        The filename (without path) for the CSV file.
        results_folder (str):  The folder path where results should be saved.
        
    Returns:
        str: Full path of the saved CSV file.
    """
    full_path = os.path.join(results_folder, filename)
    df.to_csv(full_path, index=False)
    return full_path


def plot_fleet_dynamics(fleet_df):
    """
    Plot the e-bus fleet dynamics (inflow, stock, outflow) over time,
    with distinct line styles.
    
    """
    years   = fleet_df['Year'].values
    inflow  = fleet_df['Inflow'].values
    stock   = fleet_df['Stock'].values
    outflow = fleet_df['Outflow'].values

    # Plot settings for an academic look
    plt.rcParams['font.family'] = 'calibri'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


    # 1) Smooth via linear interp on a dense x‐axis
    x_new     = np.linspace(years.min(), years.max(), 500)
    inflow_s  = np.interp(x_new, years, inflow)
    stock_s   = np.interp(x_new, years, stock)
    outflow_s = np.interp(x_new, years, outflow)

    # 2) Normalize your RGBs
    c_inflow  = '#f16c23'
    c_stock   = '#2b6a99'
    c_outflow = '#1b7c3d'

    # 3) Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(x_new, inflow_s,  color=c_inflow,  linestyle='--', lw=1.5, label='Inflow (buses/yr)')
    ax.plot(x_new, stock_s,   color=c_stock,   linestyle='-',  lw=1.5, label='Stock (buses)')
    ax.plot(x_new, outflow_s, color=c_outflow, linestyle=':',  lw=1.5, label='EoL outflow (buses/yr)')

    # 4) Optional: mark the actual data points
    ax.scatter(years, inflow,  s=15, color=c_inflow)
    ax.scatter(years, stock,   s=15, color=c_stock)
    ax.scatter(years, outflow, s=15, color=c_outflow)

    # 5) Styling
    # ax.set_title("Dynamic BEB Fleet (2026–2050)", fontsize=14)
    ax.set_xlabel("Year",               fontsize=12)
    ax.set_ylabel("Number of Buses",    fontsize=12)
    ax.grid(False)
    # ax.grid(True, linestyle='--', alpha=0.4)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.legend(
              frameon=False,
              fontsize=11,
              title_fontsize=12,
              loc='upper left')
    plt.tight_layout()
    plt.show()



def plot_scenario_series(series_dict: Dict[str, np.ndarray],
                         years: List[int],
                         title: str,
                         ylabel: str):
    """
    Plot time series of absolute values for different scenarios.
    e.g. recycled‐input mass, EoL recovered mass, etc.

    Parameters:
      series_dict: dict of {scenario: np.array(values)}
      years:       list of years
      title:       plot title
      ylabel:      label for the y‐axis
    """
    
    # ——— Matplotlib rc for an academic look ———
    plt.rcParams['font.family']    = 'calibri'
    plt.rcParams['font.size']      = 12
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['axes.grid']      = False
    plt.rcParams['xtick.direction']= 'in'
    plt.rcParams['ytick.direction']= 'in'
    
    
    # Dense x-axis
    x_new = np.linspace(min(years), max(years), 500)

    # Scenario colors
    colors = {
        'BAU':        '#f16c23',
        'Moderate':   '#2b6a99',
        'Aggressive': '#1b7c3d'
    }

    linestyles = {
        'BAU':        '--',
        'Moderate':   '-',
        'Aggressive': ':'
    }
    markers = {
        'BAU':        'o',
        'Moderate':   's',
        'Aggressive': '^'
    }

    fig, ax = plt.subplots(figsize=(6,6))
    for scen, vals in series_dict.items():
        # smooth curve
        y_s = np.interp(x_new, years, vals)
        ax.plot(x_new, y_s,
                color=colors[scen],
                linestyle=linestyles[scen],
                lw=1.5,
                label=scen)
        # overlay the actual data points
        ax.scatter(years, vals,
                   color=colors[scen],
                   marker=markers[scen],
                   s=30,
                   edgecolor='white',
                   linewidth=0.8)

    ax.set_title(title,            fontsize=14)
    ax.set_xlabel("Year",          fontsize=12)
    ax.set_ylabel(ylabel,          fontsize=12)
    ax.grid(False)
    # ax.grid(True, linestyle='--', alpha=0.3)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.legend(title="Scenario",
              frameon=False,
              fontsize=11,
              title_fontsize=12,
              loc='upper left')
    plt.tight_layout()
    plt.show()





def plot_3d_li_input_lines(years: List[int],
                           total_li: Dict[str, np.ndarray],
                           li_recycled: Dict[str, np.ndarray]):
    """
    x = Year, y = Scenario (categorical), z = Li mass.
    Solid line & fill = recycled; dashed line & lighter fill = total.
    """
    scenarios = list(total_li.keys())
    y_pos    = np.arange(len(scenarios))

    #— rcParams for a clean look —#
    plt.rcParams.update({
        'font.family':    'calibri',
        'font.size':      12,
        'axes.linewidth': 1.2,
        'axes.grid':      False,
        'xtick.direction':'in',
        'ytick.direction':'in'
    })

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    # hide panes
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)

    colors    = {'BAU':'#f16c23','Moderate':'#2b6a99','Aggressive':'#1b7c3d'}
    linestyles= {'BAU':'--','Moderate':'-','Aggressive':':'}

    for i, scen in enumerate(scenarios):
        x     = np.array(years)
        z_tot = total_li[scen]
        z_rec = li_recycled[scen]
        y     = np.full_like(x, y_pos[i], dtype=float)

        # recycled: solid line
        ax.plot(x, y, z_rec,
                color=colors[scen],
                linestyle=linestyles[scen],
                lw=1.8,
                label=f'{scen} – recycled')
        # total: dashed line
        ax.plot(x, y, z_tot,
                color=colors[scen],
                linestyle=linestyles[scen],
                lw=1.5,
                alpha=0.5,
                label=f'{scen} – total')

        # fill under recycled down to zero
        verts_rec = [(x[j], y[j], z_rec[j]) for j in range(len(x))] + \
                    [(x[j], y[j], 0       ) for j in reversed(range(len(x)))]
        ax.add_collection3d(Poly3DCollection(
            [verts_rec], facecolor=colors[scen], alpha=0.25, edgecolor=None))

        # fill between recycled and total
        verts_vir = [(x[j], y[j], z_tot[j]) for j in range(len(x))] + \
                    [(x[j], y[j], z_rec[j]) for j in reversed(range(len(x)))]
        ax.add_collection3d(Poly3DCollection(
            [verts_vir], facecolor='lightgray', alpha=0.2, edgecolor=None))

    ax.set_xlabel("Year",    labelpad=8)
    ax.set_ylabel("Scenario",labelpad=8)
    ax.set_zlabel("Li input (kg/yr)", labelpad=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios)
    plt.tight_layout()
    plt.show()


def plot_3d_li_output_lines(years: List[int],
                            total_li_EOL: Dict[str, np.ndarray],
                            li_recovered: Dict[str, np.ndarray]):
    """
    x = Year, y = Scenario, z = EoL Li mass.
    Solid line & fill = recovered; dashed = total outflow; grey fill = loss.
    """
    scenarios = list(total_li_EOL.keys())
    y_pos    = np.arange(len(scenarios))

    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111, projection='3d')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_visible(False)
    ax.grid(False)

    colors    = {'BAU':'#f16c23','Moderate':'#2b6a99','Aggressive':'#1b7c3d'}
    linestyles= {'BAU':'--','Moderate':'-','Aggressive':':'}

    for i, scen in enumerate(scenarios):
        x      = np.array(years)
        z_tot  = total_li_EOL[scen]
        z_rec  = li_recovered[scen]
        y      = np.full_like(x, y_pos[i], dtype=float)

        ax.plot(x, y, z_rec,
                color=colors[scen],
                linestyle=linestyles[scen],
                lw=1.8,
                label=f'{scen} – recovered')
        ax.plot(x, y, z_tot,
                color=colors[scen],
                linestyle=linestyles[scen],
                lw=1.5,
                alpha=0.5,
                label=f'{scen} – total EoL')

        # fill under recovered
        verts_rec = [(x[j], y[j], z_rec[j]) for j in range(len(x))] + \
                    [(x[j], y[j], 0       ) for j in reversed(range(len(x)))]
        ax.add_collection3d(Poly3DCollection(
            [verts_rec], facecolor=colors[scen], alpha=0.25, edgecolor=None))

        # fill for loss (total – recovered)
        verts_loss = [(x[j], y[j], z_tot[j]) for j in range(len(x))] + \
                     [(x[j], y[j], z_rec[j]) for j in reversed(range(len(x)))]
        ax.add_collection3d(Poly3DCollection(
            [verts_loss], facecolor='lightgray', alpha=0.2, edgecolor=None))

    ax.set_xlabel("Year",           labelpad=8)
    ax.set_ylabel("Scenario",       labelpad=8)
    ax.set_zlabel("Li at EoL (kg/yr)", labelpad=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(scenarios)
    plt.tight_layout()
    plt.show()