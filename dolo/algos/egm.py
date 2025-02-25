import yaml
from dolo.numeric.decision_rule import DecisionRule
import numpy as np
from interpolation.splines import eval_linear
from dolo.compiler.model import Model
from .results import EGMResult


def egm(
    model: Model,
    dr0: DecisionRule = None,                       # Initial guess for decision rule
    verbose: bool = False,                          # Whether to print iteration info
    details: bool = True,                           # Whether to return detailed results
    a_grid=None,                                    # Grid for post-decision states
    η_tol=1e-6,                                     # Convergence tolerance
    maxit=1000,                                     # Maximum iterations
    grid=None,                                      # Optional state space grid
    dp=None,                                        # Optional discretized process
):
    """
    a_grid: (numpy-array) vector of points used to discretize poststates; must be increasing
    """

    assert len(model.symbols["states"]) == 1        # Only one state variable allowed
    assert (
        len(model.symbols["controls"]) == 1         # Only one control variable allowed
    )  # we probably don't need this restriction

    from dolo.numeric.processes import IIDProcess

    iid_process = isinstance(model.exogenous, IIDProcess)  # Check if shocks are IID

    def vprint(t):                                  # Helper function for verbose output
        if verbose:
            print(t)

    p = model.calibration["parameters"]             # Get model parameters

    if grid is None and dp is None:
        grid, dp = model.discretize()               # Get default discretization

    s = grid["endo"].nodes                          # Get endogenous grid nodes

    funs = model.__original_gufunctions__           # Get model functions
    h = funs["expectation"]                         # Expectation function
    gt = funs["half_transition"]                    # State transition function
    τ = funs["direct_response_egm"]                 # EGM policy function
    aτ = funs["reverse_state"]                      # State update function
    lb = funs["arbitrage_lb"]                       # Lower bound function
    ub = funs["arbitrage_ub"]                       # Upper bound function

    if dr0 is None:
        x0 = model.calibration["controls"]          # Get initial controls
        dr0 = lambda i, s: x0[None, :].repeat(s.shape[0], axis=0)  # Constant policy

    n_m = dp.n_nodes                                # Number of exogenous states
    n_x = len(model.symbols["controls"])            # Number of controls

    if a_grid is None:
        raise Exception("You must supply a grid for the post-states.")

    assert a_grid.ndim == 1                         # Grid must be 1-dimensional
    a = a_grid[:, None]                             # Reshape grid for broadcasting
    N_a = a.shape[0]                                # Number of grid points

    N = s.shape[0]                                  # Number of state points

    n_h = len(model.symbols["expectations"])        # Number of expectation terms

    xa = np.zeros((n_m, N_a, n_x))                 # Policy on post-decision grid
    sa = np.zeros((n_m, N_a, 1))                   # States on post-decision grid
    xa0 = np.zeros((n_m, N_a, n_x))                # Previous policy
    sa0 = np.zeros((n_m, N_a, 1))                  # Previous states

    z = np.zeros((n_m, N_a, n_h))                  # Expectation terms

    if verbose:
        headline = "|{0:^4} | {1:10} |".format("N", " Error")
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    for it in range(0, maxit):                      # Main iteration loop

        if it == 0:
            drfut = dr0                             # Use initial guess first time

        else:
            def drfut(i, ss):                       # Future decision rule
                if iid_process:
                    i = 0                           # Only one exogenous state for IID
                m = dp.node(i)                      # Get exogenous state
                l_ = lb(m, ss, p)                   # Get lower bound
                u_ = ub(m, ss, p)                   # Get upper bound
                x = eval_linear((sa0[i, :, 0],), xa0[i, :, 0], ss)[:, None]  # Interpolate policy
                x = np.minimum(x, u_)               # Apply upper bound
                x = np.maximum(x, l_)               # Apply lower bound
                return x

        z[:, :, :] = 0                             # Reset expectations

        for i_m in range(n_m):                      # Loop over exogenous states
            m = dp.node(i_m)                        # Get current exogenous state
            for i_M in range(dp.n_inodes(i_m)):     # Loop over future states
                w = dp.iweight(i_m, i_M)            # Get transition probability
                M = dp.inode(i_m, i_M)              # Get future exogenous state
                S = gt(m, a, M, p)                  # Get future endogenous state
                X = drfut(i_M, S)                   # Get future controls
                z[i_m, :, :] += w * h(M, S, X, p)   # Update expectations
            xa[i_m, :, :] = τ(m, a, z[i_m, :, :], p)  # Compute optimal policy
            sa[i_m, :, :] = aτ(m, a, xa[i_m, :, :], p)  # Update state

        if it > 1:
            η = abs(xa - xa0).max() + abs(sa - sa0).max()  # Compute error
        else:
            η = 1                                   # Skip error first iteration

        vprint("|{0:4} | {1:10.3e} |".format(it, η))

        if η < η_tol:                               # Check convergence
            break

        sa0[...] = sa                               # Store current states
        xa0[...] = xa                               # Store current policy

    # resample the result on the standard grid
    # confusingly, what dolo calls "endo_grid" is what the EGM paper calls an exogenous grid
    # in dolo, it is endogenous wrt the draws of the exogenous variable(s) (say, the income shock)
    endo_grid = grid["endo"]                        # Get decision state grid
    exo_grid = grid["exo"]                          # Get shock grid

    # Create decision rule by interpolating among the points in the 
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")  

    mdr.set_values(                                 # Set policy values
        np.concatenate([drfut(i, s)[None, :, :] for i in range(n_m)], axis=0)
    )

    sol = EGMResult(                                # Create result object
        mdr,                                        # Decision rule
        it,                                         # Number of iterations
        dp,                                         # Discretized process
        (η < η_tol),                               # Whether converged
        η_tol,                                     # Tolerance level
        η                                          # Final error
    )

    return sol
