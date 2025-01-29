#!/usr/bin/env python
# coding: utf-8

import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn

from dolo import yaml_import, simulate
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
    # Attempt to set the current working directory
    set_cwd_to_script_or_notebook()

    # Load the EGM version of the consumption-savings model
    model_path = "../models/consumption_savings_iid_egm.yaml"
    print(f"Loading model from: {model_path}")
    model = yaml_import(model_path)

    print("Loaded model:\n", model)

    # Solve using EGM
    # supply an asset grid
    a_grid = np.linspace(0.01, 6.0, 200)
    dr = egm(model, a_grid=a_grid, verbose=True)
    print("EGM solution complete.")

    # Now let's do a quick simulation
    sim = simulate(model, dr, N=5, T=30)
    plt.figure(figsize=(8,4))
    for i in range(5):
        plt.plot(sim.sel(N=i, V='w'), label=f"Sim {i}", alpha=0.7)
    plt.title("Wealth over time (first 5 simulations)")
    plt.ylabel("w_t")
    plt.xlabel("t")
    plt.grid(True)

    # Plot consumption policy function
    plt.figure(figsize=(8,6))
    w_grid = np.linspace(0.01, 4.0, 100)
    tab = dr.eval_is(None, w_grid.reshape(-1,1))
    plt.plot(w_grid, tab[:,0], label='Consumption c(w)')
    plt.plot(w_grid, w_grid, '--', color='black', label='45° line')
    plt.xlabel('Wealth (w)')
    plt.ylabel('Consumption (c)')
    plt.title('Consumption Policy Function')
    plt.grid(True)
    plt.legend()

    # Show all figures at once at the end
    plt.show()

if __name__ == "__main__":
    main()