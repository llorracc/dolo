import yaml                                        # model files in YAML format
from dolo.numeric.decision_rule import DecisionRule  # policy functions
import numpy as np                                 # numerical computations
from interpolation.splines import eval_linear      # policy interpolation
from dolo.compiler.model import Model             # model object representation
from .results import EGMResult                    # EGM solution results
from numba import njit # imports numba's njit decorator for just-in-time compilation


def egm(
    model: Model,                                 # dolo model to solve
    dr0: DecisionRule = None,                     # initial decision rule
    verbose: bool = False,                        # iteration info flag
    details: bool = True,                         # detailed results flag
    a_grid=None,                                  # post-decision state grid
    η_tol=1e-6,                                   # convergence tolerance
    maxit=1000,                                   # max iterations
    grid=None,                                    # pre-specified grid
    dp=None,                                      # discrete process
    return_internals=False,                        # if True, return internal functions and state
):
    """
    Endogenous Grid Method (EGM) solver for models with one state and one control.
    
    Implements the EGM algorithm as described in Carroll (2006) to solve for optimal
    policy functions in dynamic models. Used extensively in examples/notebooks_py/
    consumption_savings_iid_egm.py and consumption_savings_iid_egm_Tm1.py.

    Parameters
    ----------
    model : Model
        Dolo model with one state and one control variable
    dr0 : DecisionRule, optional
        Initial guess for decision rule
    verbose : bool, default=False
        If True, print convergence information
    details : bool, default=True
        If True, return detailed solution information
    a_grid : array, optional
        Grid for post-decision states, must be increasing
    η_tol : float, default=1e-6
        Tolerance for convergence criterion
    maxit : int, default=1000
        Maximum number of iterations
    grid : dict, optional
        Pre-specified grid for states
    dp : DiscreteProcess, optional
        Pre-specified discrete process for shocks
    return_internals : bool, default=False
        If True, return internal functions and state along with result

    Returns
    -------
    EGMResult
        Contains the solution including decision rule and convergence details
    """

    assert len(model.symbols["states"]) == 1      # single endogenous state
    assert (
        len(model.symbols["controls"]) == 1
    )  # get the control bounds object from the model
    from dolo.numeric.processes import IIDProcess  # IID shocks
    iid_process = isinstance(model.exogenous, IIDProcess)  # check for IID process

    def vprint(t):                                # conditional printing helper
        if verbose:                               # verbose mode check
            print(t)                              # message output

    p = model.calibration["parameters"]
    
    if grid is None and dp is None:              # no grid supplied
        grid, dp = model.discretize()             # default discretization

    s = grid["endo"].nodes                       # decision-perch endogenous grid nodes

    funs = model.__original_gufunctions__        # compiled functions
    h = funs["expectation"]                      # expectations
    gt = funs["half_transition"]                 # state transition
    τ = funs["direct_response_egm"]              # optimal choice given post-state
    aτ = funs["reverse_state"]                   # dcsn state from choice and cntn
    lb = funs["arbitrage_lb"]                    # lower bounds
    ub = funs["arbitrage_ub"]                    # upper bounds

    if dr0 is None:                              # no initial decision rule
        x0 = model.calibration["controls"]        # calibrated controls
        dr0 = lambda i, s: x0[None, :].repeat(s.shape[0], axis=0)  # constant policy
    n_m = dp.n_nodes                             # exogenous nodes count
    n_x = len(model.symbols["controls"])         # controls count

    if a_grid is None:                           # missing required grid
        raise Exception("You must supply a grid for the post-states.")

    assert a_grid.ndim == 1                      # 1D grid check
    a = a_grid[:, None]                          # grid reshape
    N_a = a.shape[0]                             # grid points count

    N = s.shape[0]                               # state space size

    # precompute next period states for all possible combinations of current states and exogenous shocks
    n_h = len(model.symbols["expectations"])     # expectation terms count

    xa = np.zeros((n_m, N_a, n_x))              # policy array
    sa = np.zeros((n_m, N_a, 1))                # endogenous state array
    xa0 = np.zeros((n_m, N_a, n_x))             # next choices (persist for further iterations)
    sa0 = np.zeros((n_m, N_a, 1))               # next endogenous states (corresponding to choices)

    z = np.zeros((n_m, N_a, n_h))               # expectations storage

    if verbose:                                  # verbose mode
        headline = "|{0:^4} | {1:10} |".format("N", " Error")  # header format
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    for it in range(0, maxit):                   # iteration loop

    # evaluate expectations based on next values (uppercase = next decision-perch state; 'a' = current post-decision)
        if it == 0:                              # first iteration
            drfut = dr0
        else:                                    # later iterations
            def drfut(i, ss):                     # decision rule (ss = next decision-perch state)
                if iid_process:                   # IID case
                    i = 0
                m = dp.node(i)                    # exogenous state
                l_ = lb(m, ss, p)                 # lower bound
                u_ = ub(m, ss, p)                 # upper bound
                x = eval_linear((sa0[i, :, 0],), xa0[i, :, 0], ss)[:, None]  # policy interpolation
                x = np.minimum(x, u_)             # upper bound
                x = np.maximum(x, l_)             # lower bound
                return x                          # bounded policy

        z[:, :, :] = 0                           # expectations reset
        for i_m in range(n_m):                   # current exog state (matters for transition if not IID)
            m = dp.node(i_m)                      # current exogenous state
            for i_M in range(dp.n_inodes(i_m)):  # next exogenous states loop
                w = dp.iweight(i_m, i_M)          # transition weight
                M = dp.inode(i_m, i_M)            # next exogenous state
                S = gt(m, a, M, p)                # next endogenous state
                vprint(f"  (it,i_m,i_M) = ({it},{i_m},{i_M})") 
                X = drfut(i_M, S)                 # next decision
                z[i_m, :, :] += w * h(M, S, X, p)  # expectations update
            xa[i_m, :, :] = τ(m, a, z[i_m, :, :], p)  # choice given post-state
            sa[i_m, :, :] = aτ(m, a, xa[i_m, :, :], p)  # decision state via reverse_state
        if it > 1:                               # past first iteration
            η = abs(xa - xa0).max() + abs(sa - sa0).max()  # error measure
        else:                                    # first iteration
            η = 1

        vprint("|{0:4} | {1:10.3e} |".format(it, η))  # iteration progress

        if η < η_tol:                            # convergence check
            break

        sa0[...] = sa                            # next endogenous states update
        xa0[...] = xa                            # next policies update

    # evaluate current rewards based on current states and optimal controls
    endo_grid = grid["endo"]                     # endogenous grid
    exo_grid = grid["exo"]                       # exogenous grid
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")
    mdr.set_values(
        np.concatenate([drfut(i, s)[None, :, :] for i in range(n_m)], axis=0)
    ) # next choices as a function of next decision-perch state

    sol = EGMResult(mdr, it, dp, (η < η_tol), η_tol, η)
    
    if return_internals:
        internals = {
            # Functions
            'h': h,              # expectations function
            'gt': gt,            # state transition
            'τ': τ,              # optimal choice given post-state
            'aτ': aτ,            # decision state from choice
            'lb': lb,            # lower bounds
            'ub': ub,            # upper bounds
            # State
            'xa': xa,            # policy array
            'sa': sa,            # endogenous state array
            'z': z,              # expectations storage
            # Grids and process
            'grid': grid,        # state grids
            'a_grid': a_grid,    # post-decision grid
            'dp': dp            # discretized process
        }
        return sol, internals
    
    return sol

@njit # compiles the function using numba for performance
def egm_step(iA, a_grid, z_grid, probs, r, beta, interp_func, interp_func_deriv, val_func):
    """
    One step of the Endogenous Grid Method for discrete choice problems.

    This function performs a single iteration of the EGM algorithm for problems with both continuous and discrete choices.
    It updates the value function and policy function for a given discrete choice.

    Parameters:
        iA (int): Index of the discrete choice.
        a_grid (ndarray): Grid for the endogenous state variable (e.g., assets).
        z_grid (ndarray): Grid for the exogenous state variable (e.g., productivity).
        probs (ndarray): Transition probabilities for the exogenous state.
        r (float): Interest rate.
        beta (float): Discount factor.
        interp_func (callable): Function to interpolate the value function.  See, e.g., `dolo.numeric.interpolation.multilinear.multilinear_interpolation`.
        interp_func_deriv (callable): Function to compute the derivative of the value function. See, e.g., `dolo.numeric.interpolation.multilinear.multilinear_interpolation_derivative`.
        val_func (ndarray): Current value function.

    Returns:
        tuple: A tuple containing:
            - c_star (ndarray): Optimal consumption policy.
            - a_star (ndarray): Optimal asset policy.
            - v_star (ndarray): Updated value function.
    """

    n_z = z_grid.shape[0] # get the number of exogenous state grid points
    n_a = a_grid.shape[0] # get the number of endogenous state grid points

    c_star = np.empty((n_z, n_a)) # initialize an array to store optimal consumption
    a_star = np.empty((n_z, n_a)) # initialize an array to store optimal asset choices
    v_star = np.empty((n_z, n_a)) # initialize an array to store the updated value function

    for iz, z in enumerate(z_grid): # loop over exogenous state grid points

        for ia, a in enumerate(a_grid): # loop over endogenous state grid points

            ev_next = 0.0 # initialize expected value for the next period
            dev_next = 0.0 # initialize derivative of expected value for the next period

            for izp in range(n_z): # loop over next period's exogenous state grid points

                zp = z_grid[izp] # get next period's exogenous state value

                vp = interp_func(iA, zp, a, val_func) # interpolate the value function at (iA, zp, a)
                dvp = interp_func_deriv(iA, zp, a, val_func) # compute derivative of value function at (iA, zp, a)

                ev_next += vp * probs[iz, izp] # accumulate expected value
                dev_next += dvp * probs[iz, izp] # accumulate derivative of expected value

            c_star[iz, ia] = dev_next ** (-1.0) # compute optimal consumption using the inverse of the derivative
            a_star[iz, ia] = (c_star[iz, ia] + a - z) / (1 + r) # compute optimal asset choice using the budget constraint
            v_star[iz, ia] = (
                c_star[iz, ia] ** (1) - 1.0 + beta * ev_next
            )  # compute the updated value function

    return c_star, a_star, v_star # return optimal consumption, asset choice, and updated value function

@njit # compiles the function using numba for performance
def egm_egm(it_inf_bounds, a_grid, z_grid, probs, r, beta, interp_func, interp_func_deriv, val_func):
    """
    Endogenous Grid Method algorithm for discrete choice problems.

    This function implements the EGM algorithm to solve for the optimal policy and value functions
    in a dynamic programming problem with discrete choices.

    Parameters:
        it_inf_bounds (tuple): Tuple of lower and upper bounds for the infinite iteration.
        a_grid (ndarray): Grid for the endogenous state variable (e.g., assets).
        z_grid (ndarray): Grid for the exogenous state variable (e.g., productivity).
        probs (ndarray): Transition probabilities for the exogenous state.
        r (float): Interest rate.
        beta (float): Discount factor.
        interp_func (callable): Function to interpolate the value function. See, e.g., `dolo.numeric.interpolation.multilinear.multilinear_interpolation`.
        interp_func_deriv (callable): Function to compute the derivative of the value function.  See, e.g., `dolo.numeric.interpolation.multilinear.multilinear_interpolation_derivative`.
        val_func (ndarray): Initial value function.

    Returns:
        tuple: A tuple containing:
            - c_star (ndarray): Optimal consumption policy.
            - a_star (ndarray): Optimal asset policy.
            - v_star (ndarray): Updated value function.

    """

    n_z = z_grid.shape[0] # get the number of exogenous state grid points
    n_a = a_grid.shape[0] # get the number of endogenous state grid points
    nA = it_inf_bounds[1] # get the upper bound for the infinite iteration (number of discrete choices)

    c_star = np.empty((nA, n_z, n_a)) # initialize an array to store optimal consumption for each discrete choice
    a_star = np.empty((nA, n_z, n_a)) # initialize an array to store optimal asset choices for each discrete choice
    v_star = np.empty((nA, n_z, n_a)) # initialize an array to store the updated value function for each discrete choice

    for iA in range(nA): # loop over discrete choices

        c_stari, a_stari, v_stari = egm_step( # call the egm_step function for the current discrete choice
            iA, a_grid, z_grid, probs, r, beta, interp_func, interp_func_deriv, val_func
        )
        c_star[iA, :, :] = c_stari # store the optimal consumption for the current discrete choice
        a_star[iA, :, :] = a_stari # store the optimal asset choice for the current discrete choice
        v_star[iA, :, :] = v_stari # store the updated value function for the current discrete choice

    return c_star, a_star, v_star # return optimal consumption, asset choice, and updated value function for all discrete choices
