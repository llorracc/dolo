"""
Steady state solvers for dynamic economic models.

This module provides methods for finding deterministic steady states of economic models.
Key features:
- Newton-based solvers for finding equilibrium points
- Support for models with complementarity conditions
- Handles both static and dynamic steady states
- Options for controlling convergence and accuracy
"""

from dolo.compiler.model import Model  # For model class definition
from typing import Dict, List  # For type hints
import numpy as np  # For numerical operations

from dolo.compiler.misc import CalibrationDict  # For handling model calibration


def residuals(model: Model, calib=None) -> Dict[str, List[float]]:  # Compute steady-state equation residuals
    """
    Computes residuals of steady-state equations for a given model and calibration.
    
    Evaluates both transition and arbitrage equations at the steady state to verify
    that states are constant and controls satisfy optimality conditions.
    
    Parameters
    ----------
    model : Model
        The model to compute residuals for
    calib : dict, optional
        Calibration dictionary containing steady-state values. If None, uses
        model's current calibration.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'transition': Residuals of state transition equations
        - 'arbitrage': Residuals of arbitrage equations
        
    Notes
    -----
    A perfect steady state will have all residuals equal to zero, indicating
    that states are constant and controls are optimal.
    """

    if calib is None:  # Use model's calibration if none provided
        calib = model.calibration  # Get default calibration

    res = dict()  # Initialize residuals dictionary

    m = calib["exogenous"]  # Get steady-state exogenous values
    s = calib["states"]  # Get steady-state state values
    x = calib["controls"]  # Get steady-state control values
    p = calib["parameters"]  # Get model parameters
    f = model.functions["arbitrage"]  # Get arbitrage equations
    g = model.functions["transition"]  # Get transition equations

    res["transition"] = g(m, s, x, m, p) - s  # Check if states are constant
    res["arbitrage"] = f(m, s, x, m, s, x, p)  # Check if controls satisfy optimality

    return res  # Return dictionary of residuals


def find_steady_state(model: Model, *, m=None):  # Find steady state given exogenous shock
    """
    Finds the deterministic steady state of a model for given exogenous values.
    
    Uses numerical root finding to solve for state and control values that satisfy
    both the transition and arbitrage equations in steady state.
    
    Parameters
    ----------
    model : Model
        The model to solve for steady state
    m : array, optional
        Values for exogenous variables. If None, uses model's calibrated values.
        
    Returns
    -------
    CalibrationDict
        Dictionary containing the steady state values for all variables:
        - exogenous: Given or calibrated exogenous values
        - states: Solved steady state values
        - controls: Solved control values
        - parameters: Model parameters
        
    Notes
    -----
    The algorithm:
    1. Constructs an objective function combining transition and arbitrage residuals
    2. Uses scipy.optimize.root to find values making residuals zero
    3. Takes initial guess from model's current calibration
    4. Returns solution as a calibration dictionary
    
    The steady state equations solved are:
    - s = g(m, s, x, m, p)  # State transition
    - 0 = f(m, s, x, m, s, x, p)  # Arbitrage conditions
    where s are states, x are controls, m are exogenous variables,
    and p are parameters.
    """

    n_s = len(model.calibration["states"])  # Get number of state variables
    n_x = len(model.calibration["controls"])  # Get number of control variables

    if m is None:  # If no exogenous values provided
        m = model.calibration["exogenous"]  # Use calibrated values
    p = model.calibration["parameters"]  # Get model parameters

    def fobj(v):  # Objective function for root finder
        s = v[:n_s]  # Extract state variables
        x = v[n_s:]  # Extract control variables
        d = dict(states=s, controls=x, exogenous=m, parameters=p)  # Pack into dictionary
        res = residuals(model, d)  # Compute steady-state residuals
        return np.concatenate([res["transition"], res["arbitrage"]])  # Combine all residuals

    calib = model.calibration  # Get current calibration
    x0 = np.concatenate([calib["states"], calib["controls"]])  # Initial guess from calibration
    import scipy.optimize  # For numerical root finding

    sol = scipy.optimize.root(fobj, x0)  # Solve for steady state
    res = sol.x  # Extract solution vector

    d = dict(exogenous=m, states=res[:n_s], controls=res[n_s:], parameters=p)  # Pack solution
    return CalibrationDict(model.symbols, d)  # Return as calibration dictionary


def find_deterministic_equilibrium(model, verbose=True):  # Find deterministic steady state
    """
    Find the deterministic steady state of a model.
    
    Solves for state and control values that satisfy both transition and arbitrage
    equations in the absence of shocks. Uses the steady state as a starting point
    for finding the deterministic equilibrium.
    
    Parameters
    ----------
    model : Model
        The model to solve for equilibrium
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    dict
        Dictionary containing steady state values for:
        - states
        - controls
        - auxiliaries
        - values
        
    Notes
    -----
    The algorithm proceeds in two steps:
    1. Find steady state values for states and controls
    2. Compute auxiliary variables consistent with the steady state
    
    The steady state is found by solving the system of equations:
    - s = g(m, s, x, m, p)  # State transition
    - 0 = f(m, s, x, m, s, x, p)  # Arbitrage conditions
    where m are exogenous variables, s are states, x are controls,
    and p are parameters.
    """
