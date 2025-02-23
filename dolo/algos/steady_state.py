# Module for finding steady states of dynamic models
from dolo.compiler.model import Model
from typing import Dict, List
import numpy as np

from dolo.compiler.misc import CalibrationDict


def residuals(model: Model, calib=None) -> Dict[str, List[float]]:
    # Calculate residuals at steady state using model's transition and arbitrage functions

    if calib is None:
        calib = model.calibration

    res = dict()

    m = calib["exogenous"]      # Get exogenous variables
    s = calib["states"]         # Get state variables
    x = calib["controls"]       # Get control variables
    p = calib["parameters"]     # Get model parameters
    f = model.functions["arbitrage"]    # Get arbitrage function
    g = model.functions["transition"]   # Get transition function

    # Compute residuals for transition and arbitrage equations
    res["transition"] = g(m, s, x, m, p) - s
    res["arbitrage"] = f(m, s, x, m, s, x, p)

    return res


def find_steady_state(model: Model, *, m=None):
    # Find steady state by solving for states and controls that zero the residuals

    n_s = len(model.calibration["states"])     # Number of state variables
    n_x = len(model.calibration["controls"])   # Number of control variables

    if m is None:
        m = model.calibration["exogenous"]     # Use calibrated exogenous if not provided
    p = model.calibration["parameters"]        # Get model parameters

    def fobj(v):
        # Objective function that returns concatenated residuals
        s = v[:n_s]                           # Extract state variables
        x = v[n_s:]                           # Extract control variables
        d = dict(states=s, controls=x, exogenous=m, parameters=p)
        res = residuals(model, d)
        return np.concatenate([res["transition"], res["arbitrage"]])

    calib = model.calibration
    x0 = np.concatenate([calib["states"], calib["controls"]])  # Initial guess from calibration
    import scipy.optimize

    # Find root of residual equations
    sol = scipy.optimize.root(fobj, x0)
    res = sol.x

    # Return results as calibration dictionary
    d = dict(exogenous=m, states=res[:n_s], controls=res[n_s:], parameters=p)
    return CalibrationDict(model.symbols, d)
