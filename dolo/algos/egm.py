# Explanation of changes from egm_EconForge.py:
# 
# The main motivation for these changes is to ensure consistent behavior between
# time_iteration.py and egm.py algorithms when using a fixed number of iterations.
# In particular, we want solutions for any given number of maxit to be the same 
# distance from the terminal decision rule in both algorithms.
#
# Key changes:
# 1. Define drfut function once outside the loop instead of redefining it each iteration
# 2. Introduce drcurr to track the current decision rule, initialized to dr0
# 3. For the first iteration, use the initial guess (dr0)
# 4. Always update drcurr to drfut after each iteration (includes first iteration)
#      # Despite its name, 'drfut' actually returns the "current" decision rule after
#      # we update sa0 and xa0 to the current iteration's values. This ensures
#      # mdr.set_values sets the final result to the most recently computed rule.
# 5. Rearrange the state updates to be before the convergence check
# 6. Added special handling for maxit=0 to return the initial guess unchanged
# 7. Added trace parameter to record decision rules at each iteration
# 8. Consolidated grid initialization to reduce code duplication
#
# These changes allow us to:
# - Get consistent results between time_iteration.py and egm.py for the same maxit
# - Properly handle edge cases (maxit=0, maxit=1)
# - Track the evolution of decision rules across iterations

# Commenting scheme used in this file:
# - Lines starting with "##" are from egm_EconForge.py but have been commented out
# - Lines with "# newline" comments are completely new additions to this file
# - Lines with "# modified" comments are modified versions of lines from egm_EconForge.py
# - Lines with "# moved" comments exist in egm_EconForge.py but have been relocated

import yaml
from dolo.numeric.decision_rule import DecisionRule
import numpy as np
from interpolation.splines import eval_linear
from dolo.compiler.model import Model
from .results import EGMResult
import copy                                      # Add this import                     # newline


def egm(
    model: Model,
    dr0: DecisionRule = None, # Initial guess for decision rule
    verbose: bool = False,    # Whether to print iteration info
    details: bool = True,     # Whether to return detailed results
    trace: bool = False,      # Whether to record iteration history                    # newline
    a_grid=None,              # Grid for post-decision states
    η_tol=1e-6,               # Convergence tolerance
    maxit=1000,               # Maximum iterations
    grid=None,                # Optional state space grid
    dp=None,                  # Optional discretized process
):
    """
    a_grid: (numpy-array) vector of points used to discretize poststates; must be increasing
    trace: (bool) if True, decision rules at each iteration are stored                 # newline
    """
                                                                                       # newline 
    # Initialize trace collection                                                      # newline 
    trace_details = [] if trace else None                                              # newline

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

##    for it in range(0, maxit):
##
##            drfut = dr0
##
##        else:

# changed indentation of this block because it was moved out of the loop               # newline 
    def drfut(i, ss):
        if iid_process:
            i = 0
        m = dp.node(i)
        l_ = lb(m, ss, p)
        u_ = ub(m, ss, p)
        x = eval_linear((sa0[i, :, 0],), xa0[i, :, 0], ss)[:, None]
        x = np.minimum(x, u_)
        x = np.maximum(x, l_)
        return x

    # Initialize drcurr to the initial guess                                           # newline
    drcurr = dr0                                                                       # newline 
                                                                                       # newline
    # Initialize grid objects once for all code paths                                  # newline
    endo_grid = grid["endo"]                                                           # newline
    exo_grid = grid["exo"]                                                             # newline 
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")        # newline
                                                                                       # newline
    # Special case for maxit=0: return the initial guess directly                      # newline
    if maxit == 0:                                                                     # newline
        # Sample dr0 onto the standard grid                                            # newline
        mdr.set_values(                                                                # newline 
            np.concatenate([dr0(i, s)[None, :, :] for i in range(n_m)], axis=0)        # newline
        )                                                                              # newline
                                                                                       # newline
        # Add initial state to trace if enabled                                        # newline 
        if trace:                                                                      # newline
            trace_details.append({"dr": copy.deepcopy(mdr)})                           # newline
                                                                                       # newline
        sol = EGMResult(                                                               # newline
            mdr,           # Decision rule                                             # newline
            0,             # Iterations                                                # newline
            dp,            # Discretized process                                       # newline
            True,          # Converged (since no iterations requested)                 # newline
            η_tol,         # Tolerance level                                           # newline
            0.0,           # Final error (no iterations, so no error)                  # newline
            trace_details, # Add trace to result                                       # newline
        )                                                                              # newline
                                                                                       # newline    
        return sol                                                                     # newline
                                                                                       # newline 
    # For maxit > 0, run the algorithm normally                                        # newline 
    it = 0                                                                             # newline
    η = 1                                                                              # newline 
                                                                                       # newline
    # Create initial mdr to store in trace                                             # newline
    if trace:                                                                          # newline
        # Set mdr to initial decision rule                                             # newline
        mdr.set_values(                                                                # newline 
            np.concatenate([drcurr(i, s)[None, :, :] for i in range(n_m)], axis=0)     # newline 
        )                                                                              # newline 
        trace_details.append({"dr": copy.deepcopy(mdr)})                               # newline
                                                                                       # newline 
    for it in range(0, maxit):                       # moved                           # newline
        # Use drcurr for the first iteration, drfut for subsequent iterations          # newline
        decision_rule = drcurr if it == 0 else drfut  # moved                          # newline
        
        z[:, :, :] = 0
        
        for i_m in range(n_m):
            m = dp.node(i_m)
            for i_M in range(dp.n_inodes(i_m)):
                w = dp.iweight(i_m, i_M)
                M = dp.inode(i_m, i_M)
                S = gt(m, a, M, p)
                print(it, i_m, i_M)
                X = decision_rule(i_M, S)  # Use the selected decision rule; modified  # newline
                z[i_m, :, :] += w * h(M, S, X, p)
            xa[i_m, :, :] = τ(m, a, z[i_m, :, :], p)
            sa[i_m, :, :] = aτ(m, a, xa[i_m, :, :], p)
        
        # Compute error and check convergence
        if it > 1:
            η = abs(xa - xa0).max() + abs(sa - sa0).max()
        else:
            η = 1
            
        vprint("|{0:4} | {1:10.3e} |".format(it, η))
##        if η < η_tol:
##            break
        sa0[...] = sa
        xa0[...] = xa
        
        # Always update drcurr to drfut, even on the first iteration                   # newline 
        drcurr = drfut                              # Modified to always update        # newline
                                                                                       # newline        
##    # resample the result on the standard grid
        # Add current state to trace if enabled                                        # newline 
        if trace:                                                                      # newline 
            mdr.set_values(                                                            # newline 
                np.concatenate([drcurr(i, s)[None, :, :] for i in range(n_m)], axis=0) # newline 
            )                                                                          # newline 
            trace_details.append({"dr": copy.deepcopy(mdr)})                           # newline
            # newline
        if η < η_tol:  # moved                                                         # newline 
            break      # moved                                                         # newline 

    # Use drcurr for the final decision rule                                           # newline 
    mdr.set_values(
        np.concatenate([drcurr(i, s)[None, :, :] for i in range(n_m)], axis=0)# modified # newline
    )

##    sol = EGMResult(mdr, it, dp, (η < η_tol), η_tol, η)

    sol = EGMResult(   # Create result object
        mdr,           # Decision rule
        it,            # Number of iterations
        dp,            # Discretized process
        (η < η_tol),   # Whether converged
        η_tol,         # Tolerance level
        η,             # Final error
        trace_details, # Add trace to result
    )

    return sol
