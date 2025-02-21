from .egm import egm  # Import the base EGM solver
from dolo.compiler.model import Model             # model object representation
from dolo.numeric.decision_rule import DecisionRule  # policy functions
from .results import EGMResult                    # EGM solution results
from typing import Tuple, Dict, Any              # type hints

def egm_mdp(
    model: Model,                                # dolo model to solve
    dr0: DecisionRule = None,                    # initial decision rule
    verbose: bool = False,                       # iteration info flag
    details: bool = True,                        # detailed results flag
    a_grid = None,                              # post-decision state grid
    η_tol: float = 1e-6,                        # convergence tolerance
    maxit: int = 1000,                          # max iterations
    grid = None,                                # pre-specified grid
    dp = None,                                  # discrete process
) -> Tuple[EGMResult, Dict[str, Any]]:
    """
    MDP-enhanced version of EGM solver that exposes internal functions and state.
    
    Takes exactly the same arguments as egm() but returns both the solution
    and the internal objects used in the computation.
    
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
    
    Returns
    -------
    tuple
        (EGMResult, dict of internal objects) where internal objects include:
        Functions:
            h: expectations function
            gt: state transition
            τ: optimal choice given post-state
            aτ: decision state from choice
            lb: lower bounds
            ub: upper bounds
        State:
            xa: policy array
            sa: endogenous state array
            z: expectations storage
        Grids and process:
            grid: state grids
            a_grid: post-decision grid
            dp: discretized process
    """
    
    return egm(model, dr0, verbose, details, a_grid, η_tol, 
              maxit, grid, dp, return_internals=True) 

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
        interp_func (callable): Function to interpolate the value function.
        interp_func_deriv (callable): Function to compute the derivative of the value function.
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
        interp_func (callable): Function to interpolate the value function.
        interp_func_deriv (callable): Function to compute the derivative of the value function.
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

def time_iteration(model, initial_guess=None, tol=1e-6, max_iter=1000, verbose=False):
    """
    Time iteration algorithm for solving dynamic programming problems with discrete and continuous choices.

    This function implements the time iteration algorithm to find the optimal policy and value functions
    for a given dynamic programming model. It uses the Endogenous Grid Method (EGM) to handle continuous choices
    and incorporates discrete choices.

    Parameters:
        model (object): The dynamic programming model.
        initial_guess (ndarray, optional): Initial guess for the value function. Defaults to None.
        tol (float, optional): Tolerance level for convergence. Defaults to 1e-6.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        verbose (bool, optional): Whether to print iteration information. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - c_star (ndarray): Optimal consumption policy.
            - a_star (ndarray): Optimal asset policy.
            - v_star (ndarray): Optimal value function.
    """

    a_grid = model.get_grid() # get the grid for the endogenous state variable from the model
    z_grid, probs = model.get_exogenous_process() # get the grid and transition probabilities for the exogenous process
    z_grid = z_grid.nodes # extract the nodes from the exogenous grid
    probs = probs.T # transpose the transition probabilities matrix
    r = model.get_r() # get the interest rate from the model
    beta = model.get_beta() # get the discount factor from the model

    nA = model.n_choices # get the number of discrete choices from the model
    it_inf_bounds = [0, nA] # define the bounds for the infinite iteration (all discrete choices)

    if initial_guess is None: # if no initial guess is provided
        val_func = np.zeros((nA, z_grid.shape[0], a_grid.shape[0])) # initialize the value function with zeros
    else: # if an initial guess is provided
        val_func = initial_guess # use the provided initial guess

    interp_func = multilinear_interpolation # set the interpolation function to multilinear interpolation
    interp_func_deriv = multilinear_interpolation_derivative # set the derivative function to multilinear interpolation derivative

    for i in range(max_iter): # loop until convergence or maximum iterations reached

        c_star, a_star, v_star = egm_egm( # call the egm_egm function to perform one step of the EGM algorithm
            it_inf_bounds, a_grid, z_grid, probs, r, beta, interp_func, interp_func_deriv, val_func
        )

        err = abs(v_star - val_func).max() # compute the maximum absolute difference between the updated and current value functions

        if verbose: # if verbose is True
            print(f"Iteration {i}: error = {err}") # print the iteration number and the error

        if err < tol: # if the error is less than the tolerance level
            break # exit the loop (convergence achieved)

        val_func = v_star # update the value function

    return c_star, a_star, v_star # return the optimal consumption, asset choice, and value function 