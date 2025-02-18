import yaml  # For reading model files in YAML format
from dolo.numeric.decision_rule import DecisionRule  # For representing policy functions
import numpy as np  # For numerical computations
from interpolation.splines import eval_linear  # For interpolating policy functions
from dolo.compiler.model import Model  # For model object representation
from .results import EGMResult  # For storing EGM solution results


def egm(
    model: Model,  # The dolo model to solve
    dr0: DecisionRule = None,  # Initial guess for decision rule
    verbose: bool = False,  # Whether to print iteration info
    details: bool = True,  # Whether to return detailed results
    a_grid=None,  # Grid for post-decision states
    η_tol=1e-6,  # Convergence tolerance
    maxit=1000,  # Maximum number of iterations
    grid=None,  # Optional pre-specified grid
    dp=None,  # Optional discrete process
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

    Returns
    -------
    EGMResult
        Contains the solution including decision rule and convergence details
    """

    assert len(model.symbols["states"]) == 1  # Verify single state variable
    assert (
        len(model.symbols["controls"]) == 1
    )  # get the control bounds object from the model
    from dolo.numeric.processes import IIDProcess  # For handling IID shocks
    iid_process = isinstance(model.exogenous, IIDProcess)  # Check if shocks are IID

    def vprint(t):  # Helper function for verbose printing
        if verbose:  # Only print if verbose mode is on
            print(t)  # Print the message

    p = model.calibration["parameters"]  # Get calibrated parameters

    if grid is None and dp is None:  # If no grid provided
        grid, dp = model.discretize()  # Create default discretization

    s = grid["endo"].nodes  # Get nodes for endogenous states

    funs = model.__original_gufunctions__  # Get compiled functions
    h = funs["expectation"]  # Function for expectations
    gt = funs["half_transition"]  # Function for state transition
    τ = funs["direct_response_egm"]  # Function for optimal response
    aτ = funs["reverse_state"]  # Function for state updating
    lb = funs["arbitrage_lb"]  # Lower bound function
    ub = funs["arbitrage_ub"]  # Upper bound function

    if dr0 is None:  # If no initial decision rule
        x0 = model.calibration["controls"]  # Use calibrated controls
        dr0 = lambda i, s: x0[None, :].repeat(s.shape[0], axis=0)  # Create constant policy
    n_m = dp.n_nodes  # Number of exogenous nodes
    n_x = len(model.symbols["controls"])  # Number of control variables

    if a_grid is None:  # Check for required grid
        raise Exception("You must supply a grid for the post-states.")  # Raise informative error

    assert a_grid.ndim == 1  # Verify grid dimensionality
    a = a_grid[:, None]  # Reshape grid for broadcasting
    N_a = a.shape[0]  # Number of grid points

    N = s.shape[0]  # Size of state space

    # precompute next period states for all possible combinations of current states and exogenous shocks
    n_h = len(model.symbols["expectations"])  # Number of expectation terms

    xa = np.zeros((n_m, N_a, n_x))  # Initialize policy array
    sa = np.zeros((n_m, N_a, 1))  # Initialize state array
    xa0 = np.zeros((n_m, N_a, n_x))  # Previous iteration policies
    sa0 = np.zeros((n_m, N_a, 1))  # Previous iteration states

    z = np.zeros((n_m, N_a, n_h))  # Storage for expectations

    if verbose:  # If in verbose mode
        headline = "|{0:^4} | {1:10} |".format("N", " Error")  # Format header
        stars = "-" * len(headline)  # Create separator
        print(stars)  # Print top border
        print(headline)  # Print header
        print(stars)  # Print bottom border

    for it in range(0, maxit):  # Main iteration loop

    # evaluate expectations based on next period value function and states
        if it == 0:  # First iteration
            drfut = dr0  # Use initial guess
        else:  # Subsequent iterations
            def drfut(i, ss):  # Define future decision rule
                if iid_process:  # For IID case
                    i = 0  # Reset index
                m = dp.node(i)  # Get exogenous state
                l_ = lb(m, ss, p)  # Get lower bound
                u_ = ub(m, ss, p)  # Get upper bound
                x = eval_linear((sa0[i, :, 0],), xa0[i, :, 0], ss)[:, None]  # Interpolate policy
                x = np.minimum(x, u_)  # Apply upper bound
                x = np.maximum(x, l_)  # Apply lower bound
                return x  # Return bounded policy
        z[:, :, :] = 0  # Reset expectations
        for i_m in range(n_m):  # Loop over exogenous states
            m = dp.node(i_m)  # Get current node
            for i_M in range(dp.n_inodes(i_m)):  # Loop over future nodes
                w = dp.iweight(i_m, i_M)  # Get transition weight
                M = dp.inode(i_m, i_M)  # Get future node
                S = gt(m, a, M, p)  # Get future state
                print(it, i_m, i_M)  # Print progress
                X = drfut(i_M, S)  # Get future policy
                z[i_m, :, :] += w * h(M, S, X, p)  # Update expectations
            xa[i_m, :, :] = τ(m, a, z[i_m, :, :], p)  # Compute optimal policy
            sa[i_m, :, :] = aτ(m, a, xa[i_m, :, :], p)  # Update state
        if it > 1:  # After first iteration
            η = abs(xa - xa0).max() + abs(sa - sa0).max()  # Compute error
        else:  # First iteration
            η = 1  # Initialize error

        vprint("|{0:4} | {1:10.3e} |".format(it, η))  # Print iteration info

        if η < η_tol:  # Check convergence
            break  # Exit if converged

        sa0[...] = sa  # Update previous states
        xa0[...] = xa  # Update previous policies

    # evaluate current rewards based on current states and optimal controls
    endo_grid = grid["endo"]  # Get endogenous grid
    exo_grid = grid["exo"]  # Get exogenous grid
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")  # Create decision rule
    mdr.set_values(
        np.concatenate([drfut(i, s)[None, :, :] for i in range(n_m)], axis=0)
    )  # Set policy values

    sol = EGMResult(mdr, it, dp, (η < η_tol), η_tol, η)  # Create solution object

    return sol  # Return solution
