#!/usr/bin/env python
# coding: utf-8

# In[1]:
import matplotlib.pyplot as plt
from dolo import *
import os
from pathlib import Path
from dolo.algos.egm import egm
import numpy as np
from dolo.numeric.grids import CartesianGrid

def set_cwd_to_script_or_notebook():
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
        
        # Set the CWD
        os.chdir(notebook_dir)
        print("Current working directory set to:", os.getcwd())
    except Exception as e:
        print(f"Failed to set the current working directory: {e}")

# Call the function
set_cwd_to_script_or_notebook()
dolo.__file__

# In[2]:
model = yaml_import("../models/consumption_savings_iid_egm.yaml")
# In[3]:
import numpy as np
# The only change: specify maxit=1 to solve only one iteration

# Decision rule for the final period: consume everything
def dr0(i, s):
    # i: exogenous state index (unused since c=w regardless of shock)
    # s: array of wealth values where we need to evaluate the policy
    return s  # final period rule: consume all wealth

# Create grid for post-decision states (required by egm.py)
a_grid = np.linspace(0.1, 10, 10)

# Solve backward one period using EGM:
# - Uses dr0 (final period policy c(w)=w) to solve for previous period
# - Returns EGMResult object containing only decision rules, not value functions
# - Requires maxit >= 1 for algorithm to work
sol = egm(model, dr0=dr0, a_grid=a_grid, verbose=True, maxit=1)

"""
The EGMResult object returned by egm() has the following attributes:

@dataclass
class EGMResult:
    dr: DecisionRule       # Policy function mapping states to controls
                          # Callable as dr(s) or dr(i,s) where:
                          # - s: array of state values
                          # - i: optional exogenous state index
                          
    iterations: int       # Number of iterations performed
                         # Here always 1 since maxit=1
                         
    dprocess: object     # Discretized process for exogenous shocks
                         # Contains transition probabilities and nodes
                         
    a_converged: bool    # Whether policy function converged
                         # Here False since we only do 1 iteration
                         
    a_tol: float        # Tolerance level used for convergence check
                         # Default 1e-6 in egm()
                         
    err: float          # Final error in policy function
                         # Measures change in last iteration
"""

dr = sol.dr  # Extract decision rule from EGMResult object

def egm_mdp(model, dr0=None, verbose=False, details=True, a_grid=None, 
            η_tol=1e-6, maxit=1000, grid=None, dp=None, **mdp_options):
    """
    Extended EGM solver that adds MDP features to the base EGM algorithm.
    
    First calls the original egm() function, then adds:
    - Value function computation
    - Additional MDP-specific outputs
    """
    
    # First get standard EGM solution
    egm_result = egm(model, dr0, verbose, details, a_grid, η_tol, maxit, grid, dp)
    
    # Add MDP extensions here
    # ...
    
    return egm_result  # or return enhanced result object
