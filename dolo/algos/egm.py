import yaml
from dolo.numeric.decision_rule import DecisionRule  # Represents policy functions mapping states to controls
import numpy as np
from interpolation.splines import eval_linear  # For interpolating policy functions on endogenous grid
from dolo.compiler.model import Model  # Core model class that implements model_specification.md concepts
from .results import EGMResult  # Container for EGM solution results

def egm(
    model: Model,  # Economic model following model_specification.md format
    dr0: DecisionRule = None,  # Initial guess for policy function
    verbose: bool = False,  # Print convergence info
    details: bool = True,  # Print convergence info
    a_grid=None,
    η_tol=1e-6,
    maxit=1000,
    grid=None,
    dp=None,
) -> EGMResult:
    """
    Solves for optimal policy using the Endogenous Grid Method (EGM).
    
    This implements Carroll's (2006) EGM for solving consumption-savings problems.
    The method works by:
    1. Starting with a grid of end-of-period assets (endogenous grid)
    2. Computing optimal consumption for each gridpoint using Euler equation
    3. Recovering beginning-of-period assets (exogenous grid)
    
    The implementation follows the model specification in model_specification.md,
    particularly the sections on:
    - State variables (s) and controls (x)
    - Transition equations (g)
    - Expectation equations (h)
    - Direct response functions (d)
    
    Example usage from consumption_savings_iid_egm_mdp.yaml:
    ```yaml
    equations:
        transition:
            - w[t] = exp(y[t]) + (w[t-1]-c[t-1])*r
        expectation: |
            mr[t] = β*( c[t+1]/c[t] )^(-γ)*r
        direct_response_egm: |
            c[t] = c[t]*(mr[t])^(-1/γ)
    ```
    
    Parameters
    ----------
    model : Model
        Dolo model with properly defined transition, expectation and direct_response functions
    dr0 : DecisionRule, optional
        Initial guess for decision rule. If None, starts with default rule
    verbose : bool, default=True
        If True, prints convergence information
    details : bool, default=True
        If True, prints convergence information
    a_grid : array-like, optional
        Array-like object representing the endogenous grid
    η_tol : float, default=1e-6
        Stopping criterion for value function iteration
    maxit : int, default=1000
        Maximum number of iterations
    grid : array-like, optional
        Array-like object representing the exogenous grid
    dp : DiscretizedProcess, optional
        Discretized process object
        
    Returns
    -------
    EGMResult
        Object containing:
        - Decision rule (policy function)
        - Number of iterations
        - Whether convergence was achieved
        - Final change in policy function
    """

    # Discretize exogenous process (e.g. income shocks in consumption-savings)
    dp = model.exogenous.discretize()  # Converts continuous process to discrete grid
    
    # Get model dimensions from symbols defined in yaml
    n_m = dp.n_nodes  # Number of exogenous shock nodes (e.g. income states)
    n_s = len(model.symbols['states'])  # Number of state variables (e.g. assets)
    n_x = len(model.symbols['controls'])  # Number of control variables (e.g. consumption)

    # Get core model functions defined in yaml equations block
    gt = model.functions['transition']  # State transition (e.g. budget constraint)
    h = model.functions['expectation']  # Expectation function (e.g. Euler equation terms)
    τ = model.functions['direct_response']  # Policy function (e.g. consumption choice)
    aτ = model.functions['auxiliary_direct']  # Auxiliary policy (e.g. end-of-period assets)

    # Get calibrated parameters from yaml calibration block
    p = model.calibration['parameters']  # Model parameters (e.g. β, γ, r)
    
    # Setup computational grids following domain block in yaml
    endo_grid = model.get_grid()  # Grid for endogenous states (e.g. asset holdings)
    exo_grid = dp.grid  # Grid for exogenous states (e.g. income levels)
    
    # Initialize arrays for EGM iteration
    z = np.zeros((n_m, n_s, n_x))  # Expected marginal values (e.g. expected marginal utility)
    xa = np.zeros((n_m, n_s, n_x))  # Policy on endogenous grid (e.g. consumption)
    sa = np.zeros((n_m, n_s, n_x))  # States on endogenous grid (e.g. assets)

    # Initialize convergence tracking
    it = 0  # Iteration counter
    η = 1.0  # Maximum policy function change
    
    # Get initial guess or create new decision rule
    if dr0 is None:
        drfut = DecisionRule(exo_grid, endo_grid, dprocess=dp)  # Default rule
    else:
        drfut = dr0  # Use provided guess

    # Main EGM iteration loop
    while it < maxit and η > η_tol:
        
        xa_old = xa.copy()  # Store previous policy for convergence check
        
        # Loop over exogenous shock nodes (e.g. income states)
        for i_m in range(n_m):
            m = dp.node(i_m)  # Current exogenous state (e.g. income)
            
            # Loop over possible future shock realizations
            for i_M in range(dp.n_inodes(i_m)):
                w = dp.iweight(i_m, i_M)  # Probability of future shock
                M = dp.inode(i_m, i_M)  # Future exogenous state
                
                # Get next period's states using transition function
                S = gt(m, sa, M, p)  # Next period state variables
                
                if verbose:
                    print(f"Iteration {it}, Node {i_m}, Future {i_M}")
                    
                # Get future period's policy from current guess
                X = drfut(i_M, S)  # Future period controls
                
                # Update expected marginal values using expectation function
                z[i_m, :, :] += w * h(M, S, X, p)  # Accumulate expectations
            
            # Compute optimal policy using direct response function
            xa[i_m, :, :] = τ(m, sa, z[i_m, :, :], p)  # Get policy
            
            # Update endogenous states using auxiliary function
            sa[i_m, :, :] = aτ(m, sa, xa[i_m, :, :], p)
            
        # Check convergence using maximum policy function change
        η = np.max(np.abs(xa - xa_old))
        
        if verbose:
            print(f"Iteration {it}: max policy change = {η}")
            
        it += 1

    # Create final decision rule object with cubic interpolation
    mdr = DecisionRule(exo_grid, endo_grid, dprocess=dp, interp_method="cubic")
    
    # Set policy values in decision rule
    mdr.set_values(
        np.concatenate([drfut(i, s)[None, :, :] for i in range(n_m)], axis=0)
    )

    # Return results with solution and diagnostics
    return EGMResult(
        mdr,  # Final decision rule (policy function)
        it,   # Number of iterations
        dp,   # Discretized process
        (η < η_tol),  # Convergence achieved
        η_tol,  # Tolerance used
        η      # Final policy change
    ) 