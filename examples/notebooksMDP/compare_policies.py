#!/usr/bin/env python
# coding: utf-8

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from dolo import yaml_import, simulate, tabulate
from dolo.algos.egm import egm
from dolo import time_iteration

def set_cwd_to_script_or_notebook():
    try:
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
    """Compute Euler equation error at each grid point"""
    β = model.calibration['β']
    γ = model.calibration['γ']
    r = model.calibration['r']
    
    # Get consumption values using tabulate
    tab = tabulate(model, dr, 'w')
    
    errors = np.zeros_like(w_grid)
    for i, w in enumerate(w_grid):
        # Interpolate to get consumption at w
        idx = np.searchsorted(tab['w'], w)
        c = tab['c'][idx]
        
        # Next period wealth
        w_next = r*(w - c)
        
        # Interpolate to get next period consumption
        idx_next = np.searchsorted(tab['w'], w_next)
        c_next = tab['c'][idx_next]
        
        # Euler equation: 1 = β*r*E[(c'/c)^(-γ)]
        euler_lhs = β*r*(c_next/c)**(-γ)
        errors[i] = np.log10(abs(1 - euler_lhs))
    
    return errors

def main():
    set_cwd_to_script_or_notebook()

    # Load model
    model = yaml_import("../models/consumption_savings_iid.yaml")  # Standard version
    model_egm = yaml_import("../models/consumption_savings_iid_egm.yaml")  # EGM version
    
    # Solve using time iteration
    dr_ti = time_iteration(model)
    print("Time iteration solution complete.")
    
    # Solve using EGM
    a_grid = np.linspace(0.01, 6.0, 200)
    solution_egm = egm(model_egm, a_grid=a_grid)
    dr_egm = solution_egm.dr
    print("EGM solution complete.")
    
    # Get policy values on regular grid
    tab_ti = tabulate(model, dr_ti, 'w')
    tab_egm = tabulate(model_egm, dr_egm, 'w')
    
    # Create plots
    plt.figure(figsize=(12,4))
    
    # Plot both policy functions
    plt.subplot(121)
    plt.plot(tab_ti['w'], tab_ti['c'], label='Time Iteration')
    plt.plot(tab_egm['w'], tab_egm['c'], '--', label='EGM')
    plt.plot(tab_ti['w'], tab_ti['w'], 'k:', label='45° line')
    plt.xlabel('w(t)')
    plt.ylabel('c(t)')
    plt.grid(True)
    plt.legend()
    plt.title('Policy Functions Comparison')
    
    # Plot difference
    plt.subplot(122)
    plt.plot(tab_ti['w'], tab_egm['c'] - tab_ti['c'])
    plt.xlabel('w(t)')
    plt.ylabel('c_EGM(t) - c_TI(t)')
    plt.grid(True)
    plt.title('Difference in Policy Functions')
    
    # Add Euler errors plot
    w_grid = np.linspace(0.01, 6.0, 200)
    errors_ti = compute_euler_error(model, dr_ti, w_grid)
    errors_egm = compute_euler_error(model_egm, dr_egm, w_grid)
    
    plt.figure(figsize=(6,4))
    plt.plot(w_grid, errors_ti, label='Time Iteration')
    plt.plot(w_grid, errors_egm, '--', label='EGM')
    plt.xlabel('w(t)')
    plt.ylabel('log10|Euler Error|')
    plt.grid(True)
    plt.legend()
    plt.title('Euler Equation Errors')
    plt.show()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 