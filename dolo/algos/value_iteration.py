# Value iteration algorithm for solving dynamic models using Bellman equation (snt3p5)

import time
import numpy as np
import numpy
import scipy.optimize

from dolo.compiler.model import Model

from dolo.numeric.processes import DiscretizedIIDProcess

# from dolo.numeric.decision_rules_markov import MarkovDecisionRule, IIDDecisionRule
from dolo.numeric.decision_rule import DecisionRule, ConstantDecisionRule
from dolo.numeric.grids import Grid, CartesianGrid, SmolyakGrid, UnstructuredGrid
from dolo.misc.itprinter import IterationsPrinter


def constant_policy(model):
    # Create decision rule that returns constant calibrated controls
    return ConstantDecisionRule(model.calibration["controls"])


from .results import AlgoResult, ValueIterationResult


def value_iteration(
    model: Model,
    *,
    verbose: bool = False,  #
    details: bool = True,  #
    tol=1e-6,
    maxit=500,
    maxit_howard=20,
) -> ValueIterationResult:
    """
    Solve for the value function and associated Markov decision rule by iterating over
    the value function.

    Parameters:
    -----------
    model :
        model to be solved
    dr :
        decision rule to evaluate

    Returns:
    --------
    mdr : Markov decision rule
        The solved decision rule/policy function
    mdrv: decision rule
        The solved value function
    """

    transition = model.functions["transition"]        # Get state transition function
    felicity = model.functions["felicity"]           # Get instantaneous utility function
    controls_lb = model.functions["controls_lb"]     # Get control lower bounds
    controls_ub = model.functions["controls_ub"]     # Get control upper bounds

    parms = model.calibration["parameters"]          # Get model parameters
    discount = model.calibration["beta"]             # Get discount factor

    # Initialize from calibrated steady state values (snt3p5)
    x0 = model.calibration["controls"]
    m0 = model.calibration["exogenous"]
    s0 = model.calibration["states"]
    r0 = felicity(m0, s0, x0, parms)

    process = model.exogenous

    # Setup discretized state space grids (snt3p5)
    grid, dprocess = model.discretize()
    endo_grid = grid["endo"]
    exo_grid = grid["exo"]

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    mdrv = DecisionRule(exo_grid, endo_grid)        # Initialize value function approximation (snt3p5)

    s = mdrv.endo_grid.nodes                        # Get endogenous grid nodes
    N = s.shape[0]                                  # Number of grid points
    n_x = len(x0)                                   # Number of controls

    mdr = constant_policy(model)                    # Initialize with constant policy

    controls_0 = np.zeros((n_ms, N, n_x))           # Initialize control values
    for i_ms in range(n_ms):
        controls_0[i_ms, :, :] = mdr.eval_is(i_ms, s)  # Evaluate policy at each state

    values_0 = np.zeros((n_ms, N, 1))               # Initialize value function
    # for i_ms in range(n_ms):
    #     values_0[i_ms, :, :] = mdrv(i_ms, grid)

    mdr = DecisionRule(exo_grid, endo_grid)         # Create new decision rule
    # mdr.set_values(controls_0)

    # Value function iteration loop
    it = 0                                          # Iteration counter
    err_v = 100                                     # Value function error
    err_v_0 = 0                                     # Previous value error
    gain_v = 0.0                                    # Value error reduction
    err_x = 100                                     # Policy error
    err_x_0 = 0                                     # Previous policy error
    tol_x = 1e-5                                    # Policy tolerance
    tol_v = 1e-7                                    # Value tolerance

    itprint = IterationsPrinter(                    # Setup iteration printer
        ("N", int),
        ("Error_V", float),
        ("Gain_V", float),
        ("Error_x", float),
        ("Gain_x", float),
        ("Eval_n", int),
        ("Time", float),
        verbose=verbose,
    )
    itprint.print_header("Start value function iterations.")

    while (it < maxit) and (err_v > tol or err_x > tol_x):

        t_start = time.time()                       # Start iteration timer
        it += 1

        mdr.set_values(controls_0)                  # Update policy function
        if it > 2:
            ev = evaluate_policy(model, mdr, dr0=mdrv, verbose=False, details=True)  # Evaluate policy with previous value
        else:
            ev = evaluate_policy(model, mdr, verbose=False, details=True)  # Initial policy evaluation

        mdrv = ev.solution                          # Get updated value function
        for i_ms in range(n_ms):
            values_0[i_ms, :, :] = mdrv.eval_is(i_ms, s)  # Evaluate at each state

        values = values_0.copy()                    # Copy for updates
        controls = controls_0.copy()                # Copy for updates

        for i_m in range(n_ms):                     # Loop over exogenous states (snt3p5)
            m = dprocess.node(i_m)                  # Get exogenous state
            for n in range(N):                      # Loop over endogenous states
                s_ = s[n, :]                        # Get endogenous state
                x = controls[i_m, n, :]             # Get controls
                lb = controls_lb(m, s_, parms)      # Get control bounds
                ub = controls_ub(m, s_, parms)
                bnds = [e for e in zip(lb, ub)]     # Create bounds list

                def valfun(xx):                     # Value function for optimization (snt3p5)
                    return -choice_value(
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

                res = scipy.optimize.minimize(valfun, x, bounds=bnds)  # Optimize controls
                controls[i_m, n, :] = res.x         # Store optimal controls
                values[i_m, n, 0] = -valfun(x)      # Store optimal value

        # compute error, update value and dr
        err_x = abs(controls - controls_0).max()    # Maximum policy change
        err_v = abs(values - values_0).max()        # Maximum value change
        t_end = time.time()
        elapsed = t_end - t_start

        values_0 = values                           # Update stored values
        controls_0 = controls                       # Update stored controls

        gain_x = err_x / err_x_0                    # Policy error reduction
        gain_v = err_v / err_v_0                    # Value error reduction

        err_x_0 = err_x                            # Store policy error
        err_v_0 = err_v                            # Store value error

        itprint.print_iteration(                    # Print iteration info
            N=it,
            Error_V=err_v,
            Gain_V=gain_v,
            Error_x=err_x,
            Gain_x=gain_x,
            Eval_n=ev.iterations,
            Time=elapsed,
        )

    itprint.print_finished()

    mdr = DecisionRule(exo_grid, endo_grid)         # Create final decision rule

    mdr.set_values(controls)                        # Set optimal controls
    mdrv.set_values(values_0)                       # Set optimal values

    if not details:
        return mdr, mdrv                            # Return policy and value functions
    else:
        return ValueIterationResult(                # Return detailed results object (snt3p5)
            mdr,  #:AbstractDecisionRule
            mdrv,  #:AbstractDecisionRule
            it,  #:Int
            dprocess,  #:AbstractDiscretizedProcess
            err_x < tol_x,  #:Bool
            tol_x,  #:Float64
            err_x,  #:Float64
            err_v < tol_v,  #:Bool
            tol_v,  #:Float64
            err_v,  #:Float64
            None,  # log:     #:ValueIterationLog
            None,  # trace:   #:Union{Nothing,IterationTrace
        )


def choice_value(transition, felicity, i_ms, s, x, drv, dprocess, parms, beta):
    # Compute value of choice x at state (i_ms,s) with continuation value from drv

    m = dprocess.node(i_ms)                         # Get exogenous state
    cont_v = 0.0                                    # Initialize continuation value
    for I_ms in range(dprocess.n_inodes(i_ms)):    # Loop over future states
        M = dprocess.inode(i_ms, I_ms)             # Get next exogenous state
        prob = dprocess.iweight(i_ms, I_ms)        # Get transition probability
        S = transition(m, s, x, M, parms)          # Get next endogenous state
        V = drv(I_ms, S)[0]                        # Get continuation value
        cont_v += prob * V                         # Add weighted value
    return felicity(m, s, x, parms) + beta * cont_v  # Return total value


class EvaluationResult:
    def __init__(self, solution, iterations, tol, error):
        self.solution = solution                    # Store solution
        self.iterations = iterations                # Store iteration count
        self.tol = tol                             # Store tolerance
        self.error = error                         # Store final error


def evaluate_policy(
    model,
    mdr,
    tol=1e-8,
    maxit=2000,
    grid={},
    verbose=True,
    dr0=None,
    hook=None,
    integration_orders=None,
    details=False,
    interp_method="cubic",
):
    """Compute value function corresponding to policy ``dr``

    Parameters:
    -----------

    model:
        "dtcscc" model. Must contain a 'value' function.

    mdr:
        decision rule to evaluate

    Returns:
    --------

    decision rule:
        value function (a function of the space similar to a decision rule
        object)

    """

    process = model.exogenous                       # Get exogenous process
    grid, dprocess = model.discretize()             # Discretize state space
    endo_grid = grid["endo"]                       # Get endogenous grid
    exo_grid = grid["exo"]                         # Get exogenous grid

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    x0 = model.calibration["controls"]             # Get initial controls
    v0 = model.calibration["values"]               # Get initial values
    parms = model.calibration["parameters"]        # Get model parameters
    n_x = len(x0)                                  # Number of controls
    n_v = len(v0)                                  # Number of value components
    n_s = len(model.symbols["states"])             # Number of states

    if dr0 is not None:
        mdrv = dr0                                 # Use provided value function
    else:
        mdrv = DecisionRule(exo_grid, endo_grid, interp_method=interp_method)  # Create new value function

    s = mdrv.endo_grid.nodes                       # Get grid nodes
    N = s.shape[0]                                 # Number of grid points

    if isinstance(mdr, np.ndarray):
        controls = mdr                             # Use provided control values
    else:
        controls = np.zeros((n_ms, N, n_x))        # Initialize control values
        for i_m in range(n_ms):
            controls[i_m, :, :] = mdr.eval_is(i_m, s)  # Evaluate policy at each state

    values_0 = np.zeros((n_ms, N, n_v))           # Initialize value function
    if dr0 is None:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = v0[None, :]      # Use initial values
    else:
        for i_m in range(n_ms):
            values_0[i_m, :, :] = dr0.eval_is(i_m, s)  # Use provided values

    val = model.functions["value"]                 # Get value function
    g = model.functions["transition"]              # Get transition function

    sh_v = values_0.shape                          # Store value shape

    err = 10                                       # Initialize error
    inner_maxit = 50                              # Max inner iterations
    it = 0                                        # Iteration counter

    if verbose:
        headline = "|{0:^4} | {1:10} | {2:8} | {3:8} |".format(  # Setup progress display
            "N", " Error", "Gain", "Time"
        )
        stars = "-" * len(headline)
        print(stars)
        print(headline)
        print(stars)

    t1 = time.time()                              # Start timing

    err_0 = np.nan                                # Previous error

    verbit = verbose == "full"                    # Verbose output flag

    while err > tol and it < maxit:               # Main iteration loop

        it += 1

        t_start = time.time()                     # Start iteration timer

        mdrv.set_values(values_0.reshape(sh_v))   # Update value function
        values = update_value(                     # Compute new values
            val, g, s, controls, values_0, mdr, mdrv, dprocess, parms
        ).reshape((-1, n_v))
        err = abs(values.reshape(sh_v) - values_0).max()  # Compute error

        err_SA = err / err_0                      # Error reduction rate
        err_0 = err                               # Store error

        values_0 = values.reshape(sh_v)           # Update stored values

        t_finish = time.time()
        elapsed = t_finish - t_start              # Iteration time

        if verbose:
            print(                                # Print progress
                "|{0:4} | {1:10.3e} | {2:8.3f} | {3:8.3f} |".format(
                    it, err, err_SA, elapsed
                )
            )

    # values_0 = values.reshape(sh_v)

    t2 = time.time()

    if verbose:
        print(stars)
        print("Elapsed: {} seconds.".format(t2 - t1))
        print(stars)

    if not details:
        return mdrv                               # Return value function
    else:
        return EvaluationResult(mdrv, it, tol, err)  # Return detailed results


def update_value(val, g, s, x, v, dr, drv, dprocess, parms):
    # Update value function using Bellman operator

    N = s.shape[0]                                # Number of grid points
    n_s = s.shape[1]                              # Number of states

    n_ms = dprocess.n_nodes  # number of exogenous states
    n_mv = dprocess.n_inodes(0)  # this assume number of integration nodes is constant

    res = np.zeros_like(v)                        # Initialize result array

    for i_ms in range(n_ms):                      # Loop over current states

        m = dprocess.node(i_ms)[None, :].repeat(N, axis=0)  # Get exogenous state

        xm = x[i_ms, :, :]                        # Get controls
        vm = v[i_ms, :, :]                        # Get values

        for I_ms in range(n_mv):                  # Loop over future states

            M = dprocess.inode(i_ms, I_ms)[None, :].repeat(N, axis=0)  # Next exogenous state
            prob = dprocess.iweight(i_ms, I_ms)   # Transition probability

            S = g(m, s, xm, M, parms)             # Next endogenous state
            XM = dr.eval_ijs(i_ms, I_ms, S)       # Next controls
            VM = drv.eval_ijs(i_ms, I_ms, S)      # Next values
            rr = val(m, s, xm, vm, M, S, XM, VM, parms)  # Evaluate value function

            res[i_ms, :, :] += prob * rr          # Add weighted value

    return res                                    # Return updated values
