"""Time Iteration Algorithm"""

import numpy
from dolo import dprint
from dolo.compiler.model import Model
from dolo.numeric.processes import DiscretizedIIDProcess
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.grids import CartesianGrid


def residuals_simple(f, g, s, x, dr, dprocess, parms):
    # Compute residuals for time iteration using simple evaluation (snt3p5)

    N = s.shape[0]                                # Number of grid points
    n_s = s.shape[1]                             # Number of state variables

    res = numpy.zeros_like(x)                     # Initialize residuals array

    for i_ms in range(dprocess.n_nodes):          # Loop over exogenous states

        # solving on grid for markov index i_ms
        m = numpy.tile(dprocess.node(i_ms), (N, 1))
        xm = x[i_ms, :, :]

        for I_ms in range(dprocess.n_inodes(i_ms)):
            M = numpy.tile(dprocess.inode(i_ms, I_ms), (N, 1))
            prob = dprocess.iweight(i_ms, I_ms)
            S = g(m, s, xm, M, parms)
            XM = dr.eval_ijs(i_ms, I_ms, S)
            rr = f(m, s, xm, M, S, XM, parms)
            res[i_ms, :, :] += prob * rr

    return res


from .results import TimeIterationResult, AlgoResult
from dolo.misc.itprinter import IterationsPrinter
import copy


def time_iteration(
    model: Model,
    *,  #
    dr0: DecisionRule = None,  #                    # Initial guess for decision rule
    verbose: bool = True,  #                        # Whether to print iteration info
    details: bool = True,  #                        # Whether to return detailed results
    ignore_constraints: bool = False,  #            # Whether to ignore complementarity constraints
    trace: bool = False,  #                         # Whether to store iteration trace
    dprocess=None,                                  # Discretized exogenous process
    maxit=1000,                                     # Maximum outer iterations
    inner_maxit=10,                                 # Maximum inner iterations
    tol=1e-6,                                       # Convergence tolerance
    hook=None,                                      # Optional callback function
    interp_method="cubic",                          # Interpolation method for decision rule
    # obsolete
    with_complementarities=None,
) -> TimeIterationResult:
    """Finds a global solution for ``model`` using backward time-iteration.


    This algorithm iterates on the residuals of the arbitrage equations

    Parameters
    ----------
    model : Model
        model to be solved
    verbose : bool
        if True, display iterations
    dr0 : decision rule
        initial guess for the decision rule
    with_complementarities : bool (True)
        if False, complementarity conditions are ignored
    maxit: maximum number of iterations
    inner_maxit: maximum number of iteration for inner solver
    tol: tolerance criterium for successive approximations
    hook: Callable
        function to be called within each iteration, useful for debugging purposes


    Returns
    -------
    decision rule :
        approximated solution
    """

    # Handle legacy option for complementarity constraints (snt3p5)
    if with_complementarities is not None:
        # TODO warn
        pass
    else:
        with_complementarities = not ignore_constraints

    if trace:
        trace_details = []                          # Initialize trace storage
    else:
        trace_details = None

    def vprint(t):                                  # Helper function for verbose output
        if verbose:
            print(t)

    grid, dprocess_ = model.discretize()            # Discretize state space and shocks (snt3p5)

    if dprocess is None:
        dprocess = dprocess_                        # Use default discretization

    n_ms = dprocess.n_nodes                         # Number of exogenous states
    n_mv = dprocess.n_inodes(0)                     # Number of integration nodes

    x0 = model.calibration["controls"]              # Get initial controls from calibration (snt3p5)
    parms = model.calibration["parameters"]         # Get model parameters
    n_x = len(x0)                                   # Number of controls
    n_s = len(model.symbols["states"])             # Number of states

    endo_grid = grid["endo"]                       # Grid for endogenous states (snt3p5)
    exo_grid = grid["exo"]                         # Grid for exogenous states

    mdr = DecisionRule(                            # Create interpolated decision rule (snt3p5)
        exo_grid, endo_grid, dprocess=dprocess, interp_method=interp_method
    )

    s = mdr.endo_grid.nodes                        # Get grid nodes for states
    N = s.shape[0]                                 # Number of grid points

    controls_0 = numpy.zeros((n_ms, N, n_x))        # Initialize policy function array (snt3p5)
    if dr0 is None:
        controls_0[:, :, :] = x0[None, None, :]     # Use constant initial guess from calibration (snt3p5)
    else:
        if isinstance(dr0, AlgoResult):
            dr0 = dr0.dr                            # Extract decision rule from result
        try:
            for i_m in range(n_ms):
                controls_0[i_m, :, :] = dr0(i_m, s)  # Evaluate initial guess on grid
        except Exception:
            for i_m in range(n_ms):
                m = dprocess.node(i_m)
                controls_0[i_m, :, :] = dr0(m, s)    # Alternative evaluation method

    f = model.functions["arbitrage"]                # Get optimality conditions function (snt3p5)
    g = model.functions["transition"]               # Get state transition function

    if "arbitrage_lb" in model.functions and with_complementarities == True:
        lb_fun = model.functions["arbitrage_lb"]    # Get lower bound function for constraints (snt3p5)
        ub_fun = model.functions["arbitrage_ub"]    # Get upper bound function for constraints
        lb = numpy.zeros_like(controls_0) * numpy.nan  # Initialize bounds arrays
        ub = numpy.zeros_like(controls_0) * numpy.nan
        for i_m in range(n_ms):
            m = dprocess.node(i_m)[None, :]         # Get exogenous state
            p = parms[None, :]                      # Get parameters
            m = numpy.repeat(m, N, axis=0)          # Repeat for each grid point
            p = numpy.repeat(p, N, axis=0)

            lb[i_m, :, :] = lb_fun(m, s, p)        # Compute lower bounds at each point (snt3p5)
            ub[i_m, :, :] = ub_fun(m, s, p)        # Compute upper bounds at each point

    else:
        with_complementarities = False              # Disable complementarities if no bounds

    sh_c = controls_0.shape                        # Store shape for reshaping

    controls_0 = controls_0.reshape((-1, n_x))      # Flatten controls for optimization (snt3p5)

    from dolo.numeric.optimize.newton import newton, SerialDifferentiableFunction
    from dolo.numeric.optimize.ncpsolve import ncpsolve

    err = 10                                       # Initialize error measure
    it = 0                                         # Initialize iteration counter

    if with_complementarities:
        lb = lb.reshape((-1, n_x))                 # Reshape bounds for optimization
        ub = ub.reshape((-1, n_x))

    itprint = IterationsPrinter(                   # Setup iteration printer for progress (snt3p5)
        ("N", int),
        ("Error", float),
        ("Gain", float),
        ("Time", float),
        ("nit", int),
        verbose=verbose,
    )
    itprint.print_header("Start Time Iterations.")

    import time

    t1 = time.time()                              # Start timing iterations (snt3p5)

    err_0 = numpy.nan                             # Initialize previous error

    verbit = verbose == "full"                    # Set verbose flag for solver

    while err > tol and it < maxit:               # Main iteration loop until convergence (snt3p5)

        it += 1                                   # Increment counter

        t_start = time.time()                     # Start iteration timer

        mdr.set_values(controls_0.reshape(sh_c))    # Update decision rule with current controls (snt3p5)

        if trace:
            trace_details.append({"dr": copy.deepcopy(mdr)})  # Store current state

        fn = lambda x: residuals_simple(           # Define residual function for solver (snt3p5)
            f, g, s, x.reshape(sh_c), mdr, dprocess, parms
        ).reshape((-1, n_x))
        dfn = SerialDifferentiableFunction(fn)     # Create differentiable function

        res = fn(controls_0)                       # Compute current residuals

        if hook:
            hook()                                 # Call hook if provided

        if with_complementarities:
            [controls, nit] = ncpsolve(            # Solve with complementarity constraints (snt3p5)
                dfn, lb, ub, controls_0, verbose=verbit, maxit=inner_maxit
            )
        else:
            [controls, nit] = newton(              # Solve without constraints using Newton method (snt3p5)
                dfn, controls_0, verbose=verbit, maxit=inner_maxit
            )

        err = abs(controls - controls_0).max()     # Compute maximum absolute error (snt3p5)

        err_SA = err / err_0                       # Compute successive approximation ratio
        err_0 = err                                # Store current error

        controls_0 = controls                      # Update controls for next iteration (snt3p5)

        t_finish = time.time()
        elapsed = t_finish - t_start               # Compute iteration time

        itprint.print_iteration(                   # Print iteration info
            N=it, Error=err_0, Gain=err_SA, Time=elapsed, nit=nit
        )

    controls_0 = controls.reshape(sh_c)            # Reshape solution to original dimensions (snt3p5)

    mdr.set_values(controls_0)                     # Update decision rule with final solution
    if trace:
        trace_details.append({"dr": copy.deepcopy(mdr)})  # Store final state

    itprint.print_finished()                       # Print completion message

    if not details:
        return mdr                                 # Return decision rule only
    else:
        return TimeIterationResult(                # Return detailed results object (snt3p5)
            mdr,                                   # Decision rule
            it,                                    # Number of iterations
            with_complementarities,                # Whether complementarities were used
            dprocess,                              # Discretized process
            err < tol,                            # Whether convergence was achieved
            tol,                                  # Tolerance level
            err,                                  # Final error
            None,                                 # Log (not implemented)
            trace_details,                        # Trace of iterations
        )
