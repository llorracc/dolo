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

def set_cwd_to_script_or_notebook():
    """
    Attempts to detect whether we are running in a script or Jupyter environment,
    and sets the current working directory to the script's location if possible.
    """
    try:
        # Check if running in Jupyter notebook
        if "__file__" not in globals():
            # If `__file__` is not available, assume Jupyter notebook
            notebook_dir = Path.cwd()  # Use the current notebook's directory
            print(f"Detected Jupyter environment. Setting CWD to: {notebook_dir}")
        else:
            # If `__file__` exists, use the script's directory
            notebook_dir = Path(__file__).parent.resolve()
            print(f"Detected script environment. Setting CWD to: {notebook_dir}")

        os.chdir(notebook_dir)
        print("Current working directory set to:", os.getcwd())
    except Exception as e:
        print(f"Failed to set the current working directory: {e}")


def main():
    # 1) Attempt to set the current working directory
    set_cwd_to_script_or_notebook()

    # 2) Load the EGM version of the consumption-savings model
    model_path = "../models/consumption_savings_iid_egm.yaml"
    print(f"Loading model from: {model_path}")
    model = yaml_import(model_path)
    print("Loaded model:\n", model)

    # 3) Solve using EGM
    # supply an asset grid
    a_grid = np.linspace(0.01, 6.0, 200)
    solution = egm(model, a_grid=a_grid, verbose=True)
    dr_egm = solution.dr  # The actual decision rule is stored here
    print("EGM solution complete.")

    # 4) Quick simulation
    sim = simulate(model, dr_egm, N=5, T=30)
    plt.figure(figsize=(8,4))
    for i in range(5):
        plt.plot(sim.sel(N=i, V='w'), label=f"Sim {i}", alpha=0.7)
    plt.title("Wealth over time (first 5 simulations)")
    plt.ylabel("w_t")
    plt.xlabel("t")
    plt.grid(True)

    # 5) Plot policy function c(w) from the EGM solution
    tab = tabulate(model, dr_egm, 'w')
    stable_wealth = model.eval_formula('1/r + (1 - 1/r)*w(0)', tab)

    plt.figure(figsize=(6,4))
    plt.plot(tab['w'], tab['w'], color='black', linestyle='--', label='45-degree line')
    plt.plot(tab['w'], stable_wealth, color='black', linestyle='--', label='Stable wealth?')
    plt.plot(tab['w'], tab['c'], label='c(w)')
    plt.xlabel("w(t)")
    plt.ylabel("c(t)")
    plt.grid(True)
    plt.legend()
    plt.title("EGM Policy Function c(w)")
    
    # Show all figures at once at the end
    plt.show()

if __name__ == "__main__":
    main()