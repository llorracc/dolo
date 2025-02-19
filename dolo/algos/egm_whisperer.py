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