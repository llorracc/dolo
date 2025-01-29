"""
Value function iteration solvers for dynamic economic models.

This module implements value function iteration methods for solving dynamic 
optimization problems. Key features:
- Standard value function iteration
- Modified policy iteration variants
- Support for discrete and continuous state spaces
- Handling of occasionally binding constraints
- Options for controlling convergence and accuracy
"""

import time  # For timing iterations
import numpy as np  # For numerical array operations
import numpy  # For numerical computations
import scipy.optimize  # For optimization routines

from dolo.compiler.model import Model  # For model class definition
from dolo.numeric.processes import DiscretizedIIDProcess  # For discretizing stochastic processes
from dolo.numeric.decision_rule import DecisionRule, ConstantDecisionRule  # For policy function representation
from dolo.numeric.grids import Grid, CartesianGrid, SmolyakGrid, UnstructuredGrid  # For state space discretization
from dolo.misc.itprinter import IterationsPrinter  # For iteration output formatting
from dolo.numeric.optimize.newton import newton  # For numerical optimization
from dolo.numeric.optimize.ncpsolve import ncpsolve  # For complementarity problems


def constant_policy(model: Model) -> ConstantDecisionRule:
    """
    Create a constant decision rule using calibrated control values.
    
    This provides an initial guess for value iteration by using the
    steady state values from the model's calibration block.
    
    Parameters
    ----------
    model : Model
        Model with calibrated control values
        
    Returns
    -------
    ConstantDecisionRule
        Decision rule that returns calibrated controls for any state
    """
    return ConstantDecisionRule(model.calibration["controls"])  # Return calibrated controls


from .results import AlgoResult, ValueIterationResult


def value_iteration(
    model: Model,
    *,
    verbose: bool = False,  #
    details: bool = True,  #
    tol: float = 1e-6,     # Convergence tolerance
    maxit: int = 500,      # Maximum iterations
    maxit_howard: int = 20 # Maximum Howard improvement steps
) -> ValueIterationResult:
    """
    Solve for optimal value function and policy using value function iteration.
    
    Implements the value iteration algorithm described in value_iteration.md.
    The method works by:
    1. Starting with an initial guess for the value function
    2. Computing optimal controls by maximizing Bellman equation
    3. Updating value function until convergence
    
    The implementation requires the following model components from model_specification.md:
    - Transition equations (g): How states evolve
    - Felicity function (u): Per-period rewards
    - Value updating (v): Bellman equation structure
    - Control bounds: Feasible control ranges
    
    Example usage from rbc.yaml:
    ```yaml
    equations:
        transition:
            - k[t] = (1-δ)*k[t-1] + i[t-1]
        felicity:
            - u[t] = c[t]^(1-γ)/(1-γ)
        value:
            - v[t] = u[t] + β*v[t+1]
        controls_lb:
            - i[t] >= 0
    ```
    
    Parameters
    ----------
    model : Model
        Model to solve, containing transition, reward and constraint functions
    verbose : bool, default=False
        Whether to print iteration progress
    details : bool, default=True
        Return detailed results object vs just decision rules
    tol : float, default=1e-6
        Convergence tolerance for value function
    maxit : int, default=500
        Maximum number of iterations
    maxit_howard : int, default=20
        Maximum policy improvement steps between value iterations
        
    Returns
    -------
    ValueIterationResult
        Object containing:
        - mdr: Optimal policy function
        - mdrv: Optimal value function
        - iterations: Number of iterations
        - convergence info: Error metrics and tolerances
    """

    transition = model.functions["transition"]  # Get state transition function
    felicity = model.functions["felicity"]  # Get instantaneous reward function
    controls_lb = model.functions["controls_lb"]  # Get control lower bounds
    controls_ub = model.functions["controls_ub"]  # Get control upper bounds

    parms = model.calibration["parameters"]  # Get model parameters
    discount = model.calibration["beta"]  # Get discount factor

    x0 = model.calibration["controls"]  # Get initial control values
    m0 = model.calibration["exogenous"]  # Get initial exogenous state
    s0 = model.calibration["states"]  # Get initial endogenous state
    r0 = felicity(m0, s0, x0, parms)  # Compute initial reward

    process = model.exogenous  # Get exogenous process specification

    grid, dprocess = model.discretize()  # Create discretized state space
    endo_grid = grid["endo"]  # Get endogenous state grid
    exo_grid = grid["exo"]  # Get exogenous state grid

    n_ms = dprocess.n_nodes  # Number of exogenous states
    n_mv = dprocess.n_inodes(0)  # Number of integration nodes

    mdrv = DecisionRule(exo_grid, endo_grid)  # Create value function approximation
    s = mdrv.endo_grid.nodes  # Get state grid points
    N = s.shape[0]  # Number of grid points
    n_x = len(x0)  # Number of control variables
    mdr = constant_policy(model)  # Get initial policy function

    controls_0 = np.zeros((n_ms, N, n_x))  # Initialize policy array
    for i_ms in range(n_ms):  # For each exogenous state
        controls_0[i_ms, :, :] = mdr.eval_is(i_ms, s)  # Evaluate initial policy

    values_0 = np.zeros((n_ms, N, 1))  # Initialize value function array

    mdr = DecisionRule(exo_grid, endo_grid)  # Create new decision rule

    # Initialize iteration variables
    it = 0  # Iteration counter
    err_v = 100  # Value function error
    err_v_0 = 0  # Previous value error
    gain_v = 0.0  # Value error reduction
    err_x = 100  # Policy error
    err_x_0 = 0  # Previous policy error
    tol_x = 1e-5  # Policy tolerance
    tol_v = 1e-7  # Value tolerance

    itprint = IterationsPrinter(  # Setup iteration printer
        ("N", int),  # Iteration number
        ("Error_V", float),  # Value error
        ("Gain_V", float),  # Value improvement
        ("Error_x", float),  # Policy error
        ("Gain_x", float),  # Policy improvement
        ("Eval_n", int),  # Number of evaluations
        ("Time", float),  # Iteration time
        verbose=verbose,
    )
    itprint.print_header("Start value function iterations.")  # Print header

    while (it < maxit) and (err_v > tol or err_x > tol_x):  # Main iteration loop

        t_start = time.time()  # Start timing
        it += 1  # Increment counter

        mdr.set_values(controls_0)  # Update policy function
        if it > 2:  # After initial iterations
            ev = evaluate_policy(model, mdr, dr0=mdrv, verbose=False, details=True)  # Evaluate with warm start
        else:  # First iterations
            ev = evaluate_policy(model, mdr, verbose=False, details=True)  # Evaluate from scratch

        mdrv = ev.solution  # Get value function
        for i_ms in range(n_ms):  # For each exogenous state
            values_0[i_ms, :, :] = mdrv.eval_is(i_ms, s)  # Evaluate value function

        values = values_0.copy()  # Copy current values
        controls = controls_0.copy()  # Copy current controls

        for i_m in range(n_ms):  # For each exogenous state
            m = dprocess.node(i_m)  # Get exogenous value
            for n in range(N):  # For each grid point
                s_ = s[n, :]  # Get state
                x = controls[i_m, n, :]  # Get control
                lb = controls_lb(m, s_, parms)  # Get lower bound
                ub = controls_ub(m, s_, parms)  # Get upper bound
                bnds = [e for e in zip(lb, ub)]  # Create bounds list

                def valfun(xx):  # Objective function
                    """Negative of value from choosing controls xx at state (m,s)"""
                    return -choice_value(  # Negative for maximization
                        transition,
                        felicity,
                        i_m,
                        s_,
                        xx,
                        mdrv,
                        dprocess,
                        parms,
                        discount,
                    )[0]

                res = scipy.optimize.minimize(valfun, x, bounds=bnds)  # Optimize policy
                controls[i_m, n, :] = res.x  # Store optimal control
                values[i_m, n, 0] = -valfun(x)  # Store optimal value

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()  # Compute policy error
        err_v = abs(values - values_0).max()  # Compute value error
        t_end = time.time()  # End timing
        elapsed = t_end - t_start  # Calculate iteration time

        values_0 = values  # Update value function
        controls_0 = controls  # Update policy function

        gain_x = err_x / err_x_0  # Calculate policy improvement
        gain_v = err_v / err_v_0  # Calculate value improvement

        err_x_0 = err_x  # Store policy error
        err_v_0 = err_v  # Store value error

        itprint.print_iteration(  # Print iteration results
            N=it,  # Iteration number
            Error_V=err_v,  # Value error
            Gain_V=gain_v,  # Value improvement
            Error_x=err_x,  # Policy error
            Gain_x=gain_x,  # Policy improvement
            Eval_n=ev.iterations,  # Number of evaluations
            Time=elapsed,  # Iteration time
        )

    itprint.print_finished()  # Print completion message

    mdr = DecisionRule(exo_grid, endo_grid)  # Create final decision rule

    mdr.set_values(controls)  # Set optimal policy
    mdrv.set_values(values_0)  # Set optimal value function

    if not details:  # If only basic output requested
        return mdr, mdrv  # Return policy and value functions
    else:  # If detailed output requested
        return ValueIterationResult(  # Return full results
            mdr,  # Optimal policy function
            mdrv,  # Optimal value function
            it,  # Number of iterations
            dprocess,  # Discretized process
            err_x < tol_x,  # Policy convergence flag
            tol_x,  # Policy tolerance
            err_x,  # Final policy error
            err_v < tol_v,  # Value convergence flag
            tol_v,  # Value tolerance
            err_v,  # Final value error
            None,  # No log
            None,  # No trace
        )


def choice_value(
    transition, 
    felicity, 
    i_ms: int, 
    s: np.ndarray, 
    x: np.ndarray, 
    drv: DecisionRule,
    dprocess: DiscretizedIIDProcess,
    parms: dict,
    beta: float
) -> float:
    """
    Compute value of choosing control x at state (m,s).
    
    This implements the right-hand side of the Bellman equation:
    v(m,s) = u(m,s,x) + β*E[v(M,S)|m,s,x]
    
    Parameters
    ----------
    transition : callable
        State transition function g(m,s,x,M)
    felicity : callable
        Per-period reward function u(m,s,x)
    i_ms : int
        Index of current exogenous state
    s : ndarray
        Current endogenous state vector
    x : ndarray
        Control vector to evaluate
    drv : DecisionRule
        Current value function approximation
    dprocess : DiscretizedIIDProcess
        Discretized exogenous process
    parms : dict
        Model parameters
    beta : float
        Time discount factor
        
    Returns
    -------
    float
        Total value (current reward + discounted future value)
        
    Notes
    -----
    The total value is computed as:
    V(s,x) = r(s,x) + β * E[V(g(s,x,ε))]
    where:
    - r(s,x) is the current reward (felicity)
    - g(s,x,ε) is the state transition
    - V(s) is the value function
    - β is the discount factor
    - E[] is expectation over future shocks ε
    """

    m = dprocess.node(i_ms)  # Get current exogenous state
    cont_v = 0.0  # Initialize continuation value

    for I_ms in range(dprocess.n_inodes(i_ms)):  # For each future state
        M = dprocess.inode(i_ms, I_ms)  # Get future exogenous
        prob = dprocess.iweight(i_ms, I_ms)  # Get transition probability
        S = transition(m, s, x, M, parms)  # Get future state
        V = drv(I_ms, S)[0]  # Get future value
        cont_v += prob * V  # Add weighted future value

    return felicity(m, s, x, parms) + beta * cont_v  # Return total value


class EvaluationResult:  # Container for policy evaluation results
    """
    Container for results from policy evaluation.
    
    Attributes
    ----------
    solution : DecisionRule
        Value function for given policy
    iterations : int
        Number of iterations taken
    tol : float
        Tolerance level used
    error : float
        Final error achieved
    """
    def __init__(self, solution, iterations, tol, error):  # Initialize result object
        self.solution = solution  # Store optimal solution
        self.iterations = iterations  # Store iteration count
        self.tol = tol  # Store tolerance used
        self.error = error  # Store final error


def evaluate_policy(  # Evaluate value function for given policy
    model,  # Model to evaluate
    mdr,  # Decision rule to evaluate
    tol=1e-8,  # Convergence tolerance
    maxit=2000,  # Maximum iterations
    grid={},  # Optional custom grid
    verbose=True,  # Whether to print progress
    dr0=None,  # Initial value function guess
    hook=None,  # Optional callback
    integration_orders=None,  # Integration accuracy
    details=False,  # Whether to return details
    interp_method="cubic",  # Interpolation method
):
    """
    Compute value function corresponding to a given policy function.
    
    For a given policy function π(s), this computes the corresponding
    value function v(s) by iterating on the Bellman equation:
    v(s) = u(s,π(s)) + β*E[v(g(s,π(s),ε))]
    
    This is used as a subroutine in value_iteration() to evaluate
    candidate policies. The evaluation uses the model's value updating
    equation as specified in the yaml file.
    
    Parameters
    ----------
    model : Model
        Model with properly defined functions
    mdr : DecisionRule
        Policy function to evaluate
    tol : float, default=1e-8
        Convergence tolerance
    maxit : int, default=2000
        Maximum iterations
    grid : dict, default={}
        Optional custom grid specification
    verbose : bool, default=True
        Print iteration details
    dr0 : DecisionRule, optional
        Initial guess for value function
    hook : callable, optional
        Function called after each iteration
    integration_orders : dict, optional
        Custom integration orders for expectations
    details : bool, default=False
        Return detailed results
    interp_method : str, default='cubic'
        Interpolation method for value function
        
    Returns
    -------
    EvaluationResult
        Value function and convergence information
    """

    process = model.exogenous  # Get exogenous process
    grid, dprocess = model.discretize()  # Discretize state space
    endo_grid = grid["endo"]  # Get endogenous grid
    exo_grid = grid["exo"]  # Get exogenous grid

    n_ms = dprocess.n_nodes  # Number of exogenous states
    n_mv = dprocess.n_inodes(0)  # Number of integration nodes

    x0 = model.calibration["controls"]  # Get initial controls
    v0 = model.calibration["values"]  # Get initial values
    parms = model.calibration["parameters"]  # Get parameters
    n_x = len(x0)  # Number of controls
    n_v = len(v0)  # Number of values
    n_s = len(model.symbols["states"])  # Number of states

    if dr0 is not None:  # If initial guess provided
        mdrv = dr0  # Use provided guess
    else:  # Otherwise
        mdrv = DecisionRule(exo_grid, endo_grid, interp_method=interp_method)  # Create new rule

    s = mdrv.endo_grid.nodes  # Get grid nodes
    N = s.shape[0]  # Number of grid points

    if isinstance(mdr, np.ndarray):  # If policy is array
        controls = mdr  # Use directly
    else:  # If policy is decision rule
        controls = np.zeros((n_ms, N, n_x))  # Initialize controls array
        for i_m in range(n_ms):  # For each exogenous state
            controls[i_m, :, :] = mdr.eval_is(i_m, s)  # Evaluate policy

    values_0 = np.zeros((n_ms, N, n_v))  # Initialize values array
    if dr0 is None:  # If no initial guess
        for i_m in range(n_ms):  # For each exogenous state
            values_0[i_m, :, :] = v0[None, :]  # Use calibrated values
    else:  # If initial guess provided
        for i_m in range(n_ms):  # For each exogenous state
            values_0[i_m, :, :] = dr0.eval_is(i_m, s)  # Use guess values

    val = model.functions["value"]  # Get value function
    g = model.functions["transition"]  # Get transition function

    sh_v = values_0.shape  # Store value shape

    err = 10  # Initialize error
    inner_maxit = 50  # Maximum inner iterations
    it = 0  # Initialize counter

    if verbose:  # If progress output requested
        headline = "|{0:^4} | {1:10} | {2:8} | {3:8} |".format(  # Format header
            "N", " Error", "Gain", "Time"  # Column names
        )
        stars = "-" * len(headline)  # Create separator line
        print(stars)  # Print top border
        print(headline)  # Print header
        print(stars)  # Print bottom border

    t1 = time.time()  # Start total timing

    err_0 = np.nan  # Initialize previous error

    verbit = verbose == "full"  # Set verbosity level

    while err > tol and it < maxit:  # Main iteration loop

        it += 1  # Increment counter

        t_start = time.time()  # Start iteration timing

        mdrv.set_values(values_0.reshape(sh_v))  # Update value function
        values = update_value(  # Compute updated values
            val, g, s, controls, values_0, mdr, mdrv, dprocess, parms
        ).reshape((-1, n_v))
        err = abs(values.reshape(sh_v) - values_0).max()  # Compute error

        err_SA = err / err_0  # Compute error reduction
        err_0 = err  # Store error for next iteration

        values_0 = values.reshape(sh_v)  # Update values

        t_finish = time.time()  # End iteration timing
        elapsed = t_finish - t_start  # Calculate iteration time

        if verbose:  # If progress output requested
            print(  # Print iteration results
                "|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |".format(
                    it, err, err_SA, elapsed
                )
            )

    t2 = time.time()  # End total timing

    if verbose:  # If progress output requested
        print(stars)  # Print separator
        print("Elapsed: {} seconds.".format(t2 - t1))  # Print total time
        print(stars)  # Print separator

    if not details:  # If only basic output requested
        return mdrv  # Return value function
    else:  # If detailed output requested
        return EvaluationResult(mdrv, it, tol, err)  # Return full results


def update_value(
    val: callable,
    g: callable,
    s: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    dr: DecisionRule,
    drv: DecisionRule,
    dprocess: DiscretizedIIDProcess,
    parms: dict
) -> np.ndarray:
    """
    Update value function by computing expectations over future states.
    
    This implements the value updating step in policy evaluation:
    v_new(m,s) = E[val(m,s,x,v,M,S,X,V)|m,s]
    where:
    - (m,s) is current state
    - x = π(m,s) is current policy
    - v is current value
    - M is next exogenous state
    - S = g(m,s,x,M) is next endogenous state
    - X = π(M,S) is next policy
    - V = v(M,S) is next value
    
    Parameters
    ----------
    val : callable
        Value function to evaluate
    g : callable
        State transition function from model
    s : ndarray
        Grid of state points
    x : ndarray
        Current policy values on grid
    v : ndarray
        Current value function on grid
    dr : DecisionRule
        Current policy function
    drv : DecisionRule
        Current value function
    dprocess : DiscretizedIIDProcess
        Discretized exogenous process
    parms : dict
        Model parameters
        
    Returns
    -------
    ndarray
        Updated value function values on grid
    """

    N = s.shape[0]  # Number of grid points
    n_s = s.shape[1]  # Number of states

    n_ms = dprocess.n_nodes  # Number of exogenous states
    n_mv = dprocess.n_inodes(0)  # Number of integration nodes

    res = np.zeros_like(v)  # Initialize results array

    for i_ms in range(n_ms):  # For each exogenous state

        m = dprocess.node(i_ms)[None, :].repeat(N, axis=0)  # Get current exogenous

        xm = x[i_ms, :, :]  # Get current controls
        vm = v[i_ms, :, :]  # Get current values

        for I_ms in range(n_mv):  # For each future state

            M = dprocess.inode(i_ms, I_ms)[None, :].repeat(N, axis=0)  # Get future exogenous
            prob = dprocess.iweight(i_ms, I_ms)  # Get transition probability

            S = g(m, s, xm, M, parms)  # Get future state
            XM = dr.eval_ijs(i_ms, I_ms, S)  # Get future controls
            VM = drv.eval_ijs(i_ms, I_ms, S)  # Get future values
            rr = val(m, s, xm, vm, M, S, XM, VM, parms)  # Compute value

            res[i_ms, :, :] += prob * rr  # Add weighted value

    return res  # Return updated values
