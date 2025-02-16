# -*- coding: utf-8 -*-
import numpy as np # import numpy for numerical operations
from dolo.numeric.interpolation.tree import eval_linear_interp # import linear interpolation function from tree module
from dolo.numeric.misc import cartesian # import cartesian product function from misc module
from dolo.numeric.optimize.ncpsolve import ncpsolve # import ncpsolve function for solving nonlinear complementarity problems

def egm(model, v, x0=None, initial_point=None, verbose=False, hook=None, backmapping='linear'):
    """
    Solves for the optimal control using the Endogenous Grid Method (EGM).

    This function implements the Endogenous Grid Method to solve for the optimal control policy in dynamic programming problems.
    It takes a model, a value function, and optional parameters for initialization and verbosity.

    Parameters:
        model (Model): The dolo model object.
        v (ndarray): Value function at the current iteration (defined on grid 'model.grid').
        x0 (ndarray, optional): Initial guess for the control variables. Defaults to None.
        initial_point (ndarray, optional): Initial point for the solver. Defaults to None.
        verbose (bool, optional): If True, print verbose output during iteration. Defaults to False.
        hook (callable, optional): Hook function to be called after each iteration. Defaults to None.
        backmapping (str, optional): Backmapping method ('linear' or 'cubic'). Defaults to 'linear'.

    Returns:
        tuple: A tuple containing:
            - g (ndarray): Policy function (control variables as a function of state variables).
            - V (ndarray): Updated value function.

    Example:
        >>> # Assuming 'model' is a dolo model and 'v_old' is the previous value function
        >>> g_new, v_new = egm(model, v_old)
    """

    if verbose: # if verbose flag is True
        print('Starting EGM iteration') # print a message indicating the start of EGM iteration

    grid = model.grid # get the grid object from the model
    nodes = grid.nodes # get the grid nodes (state values)
    n_states = grid.n_nodes # get the number of grid nodes (number of states)
    shape = grid.shape # get the shape of the grid

    controls_bounds = model.controls_bounds # get the control bounds object from the model
    exo_grid = model.exogenous_grid # get the exogenous grid object from the model
    n_exo = exo_grid.n_nodes # get the number of exogenous grid nodes
    exo_nodes = exo_grid.nodes # get the exogenous grid nodes (exogenous values)

    f = model.f_arbitrage # get the arbitrage function (Euler equation) from the model
    g = model.f_transition # get the transition function (state update rule) from the model
    u = model.f_rewards # get the reward function from the model
    h = model.f_expectations # get the expectations function from the model (can be None)

    x_min = controls_bounds.lower(nodes) # get the lower bounds for control variables at each grid node
    x_max = controls_bounds.upper(nodes) # get the upper bounds for control variables at each grid node

    if x0 is None: # if no initial guess for controls is provided
        x0 = controls_bounds.initial_value(nodes) # use the initial value function from control bounds as initial guess

    if initial_point is None: # if no initial point for the solver is provided
        initial_point = x0 # set the initial point to the initial guess x0

    x_controls_i = np.ascontiguousarray(x0) # convert initial guess to a contiguous array for performance

    V = np.zeros(shape) # initialize the value function array with zeros, same shape as the grid

    # precompute next period states for all possible combinations of current states and exogenous shocks
    next_nodes_exo = cartesian(nodes, exo_nodes) # create cartesian product of current state nodes and exogenous shock nodes
    m_plus_exo = next_nodes_exo[:, :grid.n_dim] # extract next period state nodes (m') from the cartesian product
    e_exo = next_nodes_exo[:, grid.n_dim:] # extract exogenous shock nodes (e') from the cartesian product

    # evaluate transition function for all combinations to get next period states (s')
    s_plus_exo = g(m_exo=nodes, s_exo=nodes, x_exo=x_controls_i, e_exo=e_exo, m_plus_exo=m_plus_exo) # calculate next period states s' = g(m, s, x, e, m') using transition function

    # evaluate value function at next period states (v(s')) using interpolation
    v_plus_exo = eval_linear_interp(grid, v.flatten(), s_plus_exo) # interpolate the value function v on the next period states s'

    # evaluate expectations based on next period value function and states
    if h is not None: # check if an expectations function 'h' is defined in the model
        z_plus_exo = h(m_exo=nodes, s_exo=nodes, x_exo=x_controls_i, e_exo=e_exo, m_plus_exo=m_plus_exo, s_plus_plus_exo=s_plus_exo) # calculate expectations z' = h(m, s, x, e, m', s'') using the expectations function
        vf = lambda x: f(x, m=nodes, s=nodes, v=z_plus_exo) # define the vector function 'vf' for the NCP solver, using expectations z'
    else: # if no expectations function 'h' is defined
        vf = lambda x: f(x, m=nodes, s=nodes, v=v_plus_exo) # define the vector function 'vf' for the NCP solver, using interpolated value function v(s')

    # solve the non-linear complementarity problem to find optimal controls (x)
    results = ncpsolve(vf, x_controls_i, x_min.flatten(), x_max.flatten(), initial_point=initial_point.flatten(), verbose=False, solver='brentq') # solve the NCP using ncpsolve function to find optimal controls x

    x_controls_i = results['x'].reshape(x0.shape) # reshape the solution 'x' from the NCP solver to the original shape of controls

    # evaluate current rewards based on current states and optimal controls
    rewards = u(m=nodes, s=nodes, x=x_controls_i, e=np.zeros((n_states, 0))) # calculate current period rewards u = u(m, s, x, e)

    # evaluate expectations (if defined) or use interpolated value function for expectations
    if h is not None: # check again if an expectations function 'h' is defined
        expectations = z_plus_exo.mean(axis=1) # if defined, expectations are the mean of z' over exogenous shocks (E[z'])
    else: # if no expectations function is defined
        expectations = v_plus_exo.mean(axis=1) # if not defined, expectations are the mean of interpolated value function v(s') over exogenous shocks (E[v(s')])

    V[:] = rewards + model.beta * expectations.reshape(shape) # update the value function V = u + beta * E[v(s')] or V = u + beta * E[z']

    if hook: # if a hook function is provided
        hook(V, x_controls_i) # call the hook function with current value function and controls

    return x_controls_i, V # return the optimal controls and the updated value function
```