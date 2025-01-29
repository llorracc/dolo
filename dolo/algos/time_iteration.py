"""Time Iteration Algorithm

This module implements the time iteration method for solving dynamic economic models.

The algorithm iterates backwards in time to find a stationary solution to the model's
optimality conditions (arbitrage equations).
Key features:
- Handles both unconstrained and constrained optimization problems
- Supports models with discrete exogenous states (Markov chains)
- Uses cubic spline interpolation for continuous state variables
- Implements Newton-based nonlinear equation solver
For more details, see:
- model_specification.md: Core model structure and equation definitions
- finite_iteration.md: Theory behind time iteration methods
- processes.md: Handling of exogenous processes and discretization
The algorithm solves models specified according to model_specification.md, which defines:
- State variables (s): Endogenous states like capital, debt
- Control variables (x): Choice variables like consumption, labor
- Exogenous processes (m): Shocks like productivity, preferences
- Transition equations: How states evolve s' = g(m,s,x,m')
- Arbitrage equations: Optimality conditions f(m,s,x,m',s',x') = 0
The solution is a policy function x = φ(m,s) satisfying these equations.
"""

import numpy  # For numerical computations
from dolo import dprint  # For debug printing
from dolo.compiler.model import Model  # Base model class
from dolo.numeric.processes import DiscretizedIIDProcess  # For discretizing shocks
from dolo.numeric.decision_rule import DecisionRule  # For policy functions
from dolo.numeric.grids import CartesianGrid  # For state space discretization
from dolo.algos.steady_state import find_deterministic_equilibrium


def residuals_simple(f, g, s, x, dr, dprocess, parms):  # Compute arbitrage equation residuals
    """Compute residuals of the arbitrage equations.
    
    For each exogenous state and each point on the endogenous grid, computes
    the residual of the arbitrage equation by:
    1. Getting next period's states using the transition function g
    2. Evaluating next period's controls using the decision rule dr
    3. Computing the arbitrage equation residual using f
    4. Taking expectations over future exogenous states
    
    Parameters
    ----------
    f : callable
        Arbitrage equation function
    g : callable
        State transition function from model
    s : ndarray
        Current endogenous state values (N, n_s)
    x : ndarray
        Current control values (n_ms, N, n_x)
    dr : DecisionRule
        Current decision rule
    dprocess : Process
        Discretized process for exogenous states
    parms : ndarray
        Model parameters
        
    Returns
    -------
    ndarray
        Residuals of arbitrage equations (n_ms, N, n_x)
    """

    N = s.shape[0]  # Number of grid points
    n_s = s.shape[1]  # Number of state variables

    res = numpy.zeros_like(x)  # Initialize residuals array

    for i_ms in range(dprocess.n_nodes):  # Loop over exogenous states

        # solving on grid for markov index i_ms
        m = numpy.tile(dprocess.node(i_ms), (N, 1))  # Current exogenous state
        xm = x[i_ms, :, :]  # Current controls

        for I_ms in range(dprocess.n_inodes(i_ms)):  # Loop over future states
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N, 1))  # Next period exogenous
            prob = dprocess.iweight(i_ms, I_ms)  # Transition probability
            S = g(m, s, xm, M, parms)  # Next period states
            XM = dr.eval_ijs(i_ms, I_ms, S)  # Next period controls
            rr = f(m, s, xm, M, S, XM, parms)  # Arbitrage equation residuals
            res[i_ms, :, :] += prob * rr  # Add weighted residuals

    return res  # Return total residuals


from .results import TimeIterationResult, AlgoResult  # For returning solution results
from dolo.misc.itprinter import IterationsPrinter  # For iteration output
import copy  # For deep copying objects


def time_iteration(
    model: Model,  # Model to solve
    *,  #
    dr0: DecisionRule = None,  # Initial guess for decision rule
    verbose: bool = True,  # Whether to print progress
    details: bool = True,  # Whether to return detailed results
    ignore_constraints: bool = False,  # Whether to ignore bounds
    trace: bool = False,  # Whether to store iteration history
    dprocess=None,  # Optional custom discretization
    maxit=1000,  # Maximum iterations
    inner_maxit=10,  # Maximum iterations for inner solver
    tol=1e-6,  # Convergence tolerance
    hook=None,  # Optional callback function
    interp_method="cubic",  # Interpolation method
    # obsolete
    with_complementarities=None,  # Deprecated option
) -> TimeIterationResult:
    """Find a global solution for ``model`` using backward time-iteration.

    This algorithm iterates on the residuals of the arbitrage equations until convergence.
    At each iteration:
    1. Given current decision rule dr_t for controls
    2. For each state (m,s):
        a. Compute next period states S using transition g
        b. Get next period controls X = dr_t(S) 
        c. Solve arbitrage equation f(m,s,x,M,S,X) = 0 for x
    3. Update dr_t+1 with new optimal controls x
    4. Repeat until successive rules converge

    The algorithm handles complementarity conditions (inequality constraints) by:
    1. Checking if controls hit bounds lb ≤ x ≤ ub
    2. Using an appropriate nonlinear solver (ncpsolve)
    3. Modifying arbitrage equations at the bounds

    Parameters
    ----------
    model : Model
        Model to solve
    dr0 : DecisionRule, optional
        Initial guess for decision rule. If None, uses calibrated values
    verbose : bool, default=True
        Whether to print iteration progress
    details : bool, default=True
        Whether to return detailed solution results
    ignore_constraints : bool, default=False
        Whether to ignore complementarity constraints
    trace : bool, default=False
        Whether to store iteration history
    dprocess : Process, optional
        Custom discretization process. If None, uses model's default
    maxit : int, default=1000
        Maximum number of outer loop iterations
    inner_maxit : int, default=10
        Maximum number of inner loop iterations
    tol : float, default=1e-6
        Convergence tolerance for successive approximations
    hook : callable, optional
        Function called after each iteration for debugging
    interp_method : str, default='cubic'
        Interpolation method for decision rule
    with_complementarities : bool, optional
        Deprecated. Use ignore_constraints instead
        
    Returns
    -------
    TimeIterationResult or DecisionRule
        If details=True, returns TimeIterationResult with:
        - Converged decision rule
        - Number of iterations
        - Complementarity info
        - Convergence status
        If details=False, returns just the decision rule
    """

    # deal with obsolete options
    if with_complementarities is not None:
        # TODO warn
        pass
    else:
        with_complementarities = not ignore_constraints  # Set complementarity flag based on constraints

    if trace:  # Initialize trace storage if requested
        trace_details = []
    else:
        trace_details = None

    from dolo import dprint  # Import debug printing utility

    def vprint(t):  # Helper function for verbose output
        if verbose:
            print(t)

    grid, dprocess_ = model.discretize()  # Get discretized state space

    if dprocess is None:  # Use default discretization if none provided
        dprocess = dprocess_

    n_ms = dprocess.n_nodes  # Number of exogenous states
    n_mv = dprocess.n_inodes(0)  # Number of integration nodes

    x0 = model.calibration["controls"]  # Get initial control values
    parms = model.calibration["parameters"]  # Get model parameters
    n_x = len(x0)  # Number of control variables
    n_s = len(model.symbols["states"])  # Number of state variables

    endo_grid = grid["endo"]  # Grid for endogenous states
    exo_grid = grid["exo"]  # Grid for exogenous states

    mdr = DecisionRule(  # Create decision rule object
        exo_grid, endo_grid, 
        dprocess=dprocess,
        interp_method=interp_method
    )

    s = mdr.endo_grid.nodes  # Get grid nodes
    N = s.shape[0]  # Number of grid points

    controls_0 = numpy.zeros((n_ms, N, n_x))  # Initialize control values
    if dr0 is None:  # If no initial guess provided
        controls_0[:, :, :] = x0[None, None, :]  # Use calibrated controls
    else:
        if isinstance(dr0, AlgoResult):  # Extract decision rule if needed
            dr0 = dr0.dr
        try:
            for i_m in range(n_ms):  # Try evaluating on grid
                controls_0[i_m, :, :] = dr0(i_m, s)
        except Exception:
            for i_m in range(n_ms):  # Fall back to direct evaluation
                m = dprocess.node(i_m)  # Get exogenous state
                controls_0[i_m, :, :] = dr0(m, s)  # Evaluate initial guess

    f = model.functions["arbitrage"]  # Get arbitrage equations
    g = model.functions["transition"]  # Get transition equations

    if "arbitrage_lb" in model.functions and with_complementarities == True:  # Handle bounds
        lb_fun = model.functions["arbitrage_lb"]  # Lower bound function
        ub_fun = model.functions["arbitrage_ub"]  # Upper bound function
        lb = numpy.zeros_like(controls_0) * numpy.nan  # Initialize lower bounds
        ub = numpy.zeros_like(controls_0) * numpy.nan  # Initialize upper bounds
        for i_m in range(n_ms):  # Compute bounds at each point
            m = dprocess.node(i_m)[None, :]  # Get exogenous state
            p = parms[None, :]  # Get parameters
            m = numpy.repeat(m, N, axis=0)  # Repeat for each grid point
            p = numpy.repeat(p, N, axis=0)  # Repeat parameters

            lb[i_m, :, :] = lb_fun(m, s, p)  # Evaluate lower bounds
            ub[i_m, :, :] = ub_fun(m, s, p)  # Evaluate upper bounds

    else:
        with_complementarities = False  # Disable bounds if not provided

    sh_c = controls_0.shape  # Store shape for reshaping

    controls_0 = controls_0.reshape((-1, n_x))  # Flatten controls for solver

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction  # Import solvers
    from dolo.numeric.optimize.ncpsolve import ncpsolve  # Import complementarity solver

    err = 10  # Initialize error measure
    it = 0  # Initialize iteration counter
    err_0 = numpy.nan  # Previous error

    if with_complementarities:  # If using bounds
        lb = lb.reshape((-1, n_x))  # Reshape bounds to match controls
        ub = ub.reshape((-1, n_x))  # Reshape bounds to match controls

    itprint = IterationsPrinter(  # Setup iteration printer
        ("N", int),  # Iteration number
        ("Error", float),  # Current error
        ("Gain", float),  # Error reduction
        ("Time", float),  # Iteration time
        ("nit", int),  # Inner iterations
        verbose=verbose,
    )
    itprint.print_header("Start Time Iterations.")  # Print header

    import time  # For timing iterations

    t1 = time.time()  # Start timing

    err_0 = numpy.nan  # Initialize previous error
    verbit = verbose == "full"  # Set verbosity level

    while err > tol and it < maxit:  # Main iteration loop

        it += 1  # Increment counter

        t_start = time.time()  # Start iteration timer

        mdr.set_values(controls_0.reshape(sh_c))  # Update decision rule

        if trace:  # Store iteration history if requested
            trace_details.append({"dr": copy.deepcopy(mdr)})

        fn = lambda x: residuals_simple(  # Define residual function
            f, g, s, x.reshape(sh_c), mdr, dprocess, parms
        ).reshape((-1, n_x))
        dfn = SerialDifferentiableFunction(fn)  # Make function differentiable

        res = fn(controls_0)  # Compute residuals

        if hook:  # Call hook if provided
            hook()

        if with_complementarities:  # Solve with bounds if needed
            [controls, nit] = ncpsolve(
                dfn, lb, ub, controls_0, 
                verbose=verbit, 
                maxit=inner_maxit
            )
        else:  # Solve without bounds
            # Use Newton solver for unconstrained case
            [controls, nit] = newton(
                dfn, controls_0,
                verbose=verbit,
                maxit=inner_maxit
            )

        err = abs(controls - controls_0).max()  # Compute error

        err_SA = err / err_0  # Compute error reduction
        err_0 = err  # Store error for next iteration

        controls_0 = controls  # Update controls

        t_finish = time.time()  # End iteration timer
        elapsed = t_finish - t_start  # Compute iteration time

        # Print iteration info
        itprint.print_iteration(
            N=it,           # Iteration number
            Error=err_0,    # Current error
            Gain=err_SA,    # Error reduction
            Time=elapsed,   # Iteration time
            nit=nit         # Inner iterations
        )

    controls_0 = controls.reshape(sh_c)  # Reshape solution

    mdr.set_values(controls_0)  # Set final values
    if trace:  # Store final iteration if tracing
        trace_details.append({"dr": copy.deepcopy(mdr)})

    itprint.print_finished()  # Print completion message

    if not details:  # Return just decision rule if no details wanted
        return mdr

    return TimeIterationResult(  # Return full results
        mdr,  # Decision rule
        it,  # Iterations
        with_complementarities,  # Whether bounds were used
        dprocess,  # Discretization process
        err < tol,  # Whether converged
        tol,  # Tolerance used
        err,  # Final error
        None,  # Log (not used)
        trace_details,  # Iteration history
    )
