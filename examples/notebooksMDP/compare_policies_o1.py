#!/usr/bin/env python
# coding: utf-8

import sys
import os
from pathlib import Path
import subprocess
import platform

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from dolo import yaml_import, time_iteration, tabulate
from dolo.algos.egm import egm, EGMResult

def set_cwd_to_script_or_notebook():
    """
    Attempts to detect whether we are running in a script or Jupyter environment,
    and sets the current working directory to the script's location if possible.
    """
    try:
        # Check if running in Jupyter
        if "__file__" not in globals():
            notebook_dir = Path.cwd()
            print(f"Detected Jupyter environment. Setting CWD to: {notebook_dir}")
        else:
            notebook_dir = Path(__file__).parent.resolve()
            print(f"Detected script environment. Setting CWD to: {notebook_dir}")
        os.chdir(notebook_dir)
        print("Current working directory set to:", os.getcwd())
    except Exception as e:
        print(f"Failed to set the current working directory: {e}")

def compute_euler_error(model, dr, w_grid):
    """
    Compute Euler eq. error (log10 of absolute error) at each grid point w.
    For a standard consumption-savings model with
        Euler eq: 1 = β * r * (c'(w_next)/c(w))^(-γ)
        w_next = r*(w - c(w))
    """
    beta = model.calibration['β']
    gamma = model.calibration['γ']
    r = model.calibration['r']

    # Get policy function values on a fine grid
    tab = tabulate(model, dr, 'w')
    
    errors = np.zeros(len(w_grid))
    for i, wval in enumerate(w_grid):
        # Get current consumption using interpolation
        c_now = np.interp(wval, tab['w'], tab['c'])
        
        # Next period wealth
        w_next = r*(wval - c_now)
        if w_next < 0:
            errors[i] = np.nan
            continue
        
        # Get next period consumption using interpolation
        c_next = np.interp(w_next, tab['w'], tab['c'])
        
        # Compute Euler equation error
        lhs = beta*r*(c_next/c_now)**(-gamma)
        err = abs(1.0 - lhs)
        errors[i] = np.log10(max(err, 1e-15))  # Avoid -inf

    return errors

def open_file(filepath):
    """Open file with default system viewer"""
    system = platform.system()
    try:
        if system == 'Darwin':  # macOS
            subprocess.run(['open', filepath])
        elif system == 'Linux':
            subprocess.run(['xdg-open', filepath])
        else:
            print(f"File saved but auto-open not supported on {system}")
    except Exception as e:
        print(f"Could not open file: {e}")

def main():
    # 1) Possibly fix working directory
    set_cwd_to_script_or_notebook()

    # 2) Load the time iteration model
    model_ti_path = "../models/consumption_savings_iid.yaml"
    model_ti = yaml_import(model_ti_path)
    print("Time iteration model loaded.")

    # 3) Solve using time iteration - only 1 iteration
    dr_ti = time_iteration(model_ti, maxit=1)
    print("Time iteration solution complete (1 iteration).")

    # Tabulate c(w)
    tab_ti = tabulate(model_ti, dr_ti, 'w')
    w_ti = tab_ti['w']
    c_ti = tab_ti['c']

    # 4) Load the EGM model
    model_egm_path = "../models/consumption_savings_iid_egm.yaml"
    model_egm = yaml_import(model_egm_path)
    print("EGM model loaded.")

    # Solve using EGM - only 1 iteration
    a_grid = np.linspace(0.01, 6.0, 200)
    egm_solution = egm(model_egm, a_grid=a_grid, verbose=False, maxit=1)
    dr_egm = egm_solution.dr
    print("EGM solution complete (1 iteration).")

    # Tabulate c(w) from EGM
    tab_egm = tabulate(model_egm, dr_egm, 'w')
    w_egm = tab_egm['w']
    c_egm = tab_egm['c']

    # 5) Plot policy c(w) from both methods and the difference
    plt.figure(figsize=(10,5))

    # Panel (1) - c(w) from both
    plt.subplot(1,2,1)
    plt.plot(w_ti, c_ti, label="Time Iteration")
    plt.plot(w_egm, c_egm, '--', label="EGM")
    plt.plot(w_ti, w_ti, 'k:', label="45 deg line")
    plt.xlabel("w(t)")
    plt.ylabel("c(t)")
    plt.title("Policy Function Comparison")
    plt.legend()
    plt.grid(True)

    # Panel (2) - difference in c(w)
    # Use numpy's interp instead of interpolation.splines
    c_egm_on_ti = np.interp(w_ti, w_egm, c_egm)
    diff = c_egm_on_ti - c_ti

    plt.subplot(1,2,2)
    plt.plot(w_ti, diff, color='purple')
    plt.axhline(y=0.0, color='black', linestyle='--')
    plt.xlabel("w(t)")
    plt.ylabel("c_EGM - c_TI")
    plt.title("Difference in Policy Functions")
    plt.grid(True)

    plt.tight_layout()
    
    # Save and open figure 1
    fig1_path = '/tmp/policy_comparison.png'
    plt.savefig(fig1_path)
    print(f"Saved figure to: {fig1_path}")
    plt.close()
    open_file(fig1_path)

    # 6) Compute Euler equation errors for each method, on a shared grid
    # Start grid at r to avoid invalid regions
    r = model_ti.calibration['r']
    w_shared = np.linspace(r, 4.0, 100)  # Changed starting point from 0.01 to r
    err_ti = compute_euler_error(model_ti, dr_ti, w_shared)
    err_egm = compute_euler_error(model_egm, dr_egm, w_shared)
    diff_err = err_egm - err_ti

    # 7) Plot Euler eq. errors
    plt.figure(figsize=(10,5))

    # Panel (1): log10 errors
    plt.subplot(1,2,1)
    plt.plot(w_shared, err_ti, label="Time Iteration")
    plt.plot(w_shared, err_egm, '--', label="EGM")
    plt.axhline(0.0, color='k', lw=0.8)
    plt.legend()
    plt.xlabel("w(t)")
    plt.ylabel("log10(euler error)")
    plt.title("Euler equation errors (w ≥ r)")
    plt.grid(True)

    # Panel (2): difference in log10 errors
    plt.subplot(1,2,2)
    plt.plot(w_shared, diff_err, color='purple')
    plt.axhline(0.0, color='k', lw=0.8)
    plt.title("Euler error difference: EGM - TI (w ≥ r)")
    plt.xlabel("w(t)")
    plt.grid(True)

    plt.tight_layout()
    
    # Save and open figure 2
    fig2_path = '/tmp/euler_errors.png'
    plt.savefig(fig2_path)
    print(f"Saved figure to: {fig2_path}")
    plt.close()
    open_file(fig2_path)


if __name__ == "__main__":
    main()