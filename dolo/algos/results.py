from dataclasses import dataclass  # For clean class definitions with type hints

class AlgoResult:
    """
    Base class for all algorithm results in dolo.
    
    Each solution method (as described in docs/*.md) returns a specific result class
    that inherits from this base class. The result classes contain:
    1. The computed decision rules (policy functions)
    2. Convergence information
    3. Solution diagnostics
    """
    pass


@dataclass
class TimeIterationResult(AlgoResult):
    """
    Result from time iteration algorithm for solving dynamic models.
    
    Time iteration solves for policy functions by iterating on the model's
    arbitrage equations as described in time_iteration.md. Used for models
    with occasionally binding constraints.
    
    Example usage from rbc.yaml:
    ```yaml
    equations:
        arbitrage:
            - 1 - β*(c[t]/c[t+1])*(1-δ + rk[t+1])  | 0 <= i[t] <= inf
    ```
    """
    dr: object                # Decision rule mapping states to controls (from dolo.numeric.decision_rule)
    iterations: int           # Number of iterations until convergence/max iterations
    complementarities: bool   # Whether model has complementarity conditions (inequality constraints)
    dprocess: object         # Discretized exogenous process (from dolo.numeric.processes)
    x_converged: bool        # Whether policy function converged within tolerance
    x_tol: float            # Tolerance level for convergence criterion
    err: float              # Final error in policy function between iterations
    log: object             # Detailed log of convergence progress
    trace: object           # Optional trace of policy function evolution


@dataclass 
class EGMResult(AlgoResult):
    """
    Result from Endogenous Grid Method (EGM) algorithm.
    
    EGM solves consumption-savings problems by exploiting the structure
    of the Euler equation, as implemented in algos/egm.py. The method
    requires models to specify:
    - Transition equations (budget constraint)
    - Expectation equations (Euler equation terms)
    - Direct response functions (consumption function)
    
    Example usage from consumption_savings_iid.yaml:
    ```yaml
    equations:
        transition:
            - w[t] = (w[t-1]-c[t-1])*r + y[t]
        expectation:
            - z[t] = β*(c[t+1]/c[t])^(-γ)*r
    ```
    """
    dr: object              # Decision rule for consumption policy (from dolo.numeric.decision_rule)
    iterations: int         # Number of iterations until convergence/max iterations
    dprocess: object        # Discretized income process (from dolo.numeric.processes)
    a_converged: bool       # Whether consumption policy converged within tolerance
    a_tol: float           # Tolerance level for convergence criterion
    err: float             # Final change in consumption policy between iterations


@dataclass
class ValueIterationResult(AlgoResult):
    """
    Result from value function iteration algorithm.
    
    Solves dynamic programming problems by iterating on the value function
    as described in value_iteration.md. Requires models to specify:
    - Transition equations (state evolution)
    - Utility function (per-period rewards)
    - Value updating equation (Bellman equation)
    
    Example usage from rbc.yaml:
    ```yaml
    equations:
        utility:
            - r[t] = c[t]^(1-γ)/(1-γ)
        value:
            - v[t] = r[t] + β*v[t+1]
    ```
    """
    dr: object             # Decision rule for optimal controls (from dolo.numeric.decision_rule)
    drv: object           # Value function approximation (from dolo.numeric.decision_rule)
    iterations: int        # Number of iterations until convergence/max iterations
    dprocess: object       # Discretized exogenous process (from dolo.numeric.processes)
    x_converged: object    # Whether policy function converged within tolerance
    x_tol: float          # Tolerance for policy function convergence
    x_err: float          # Final change in policy between iterations
    v_converged: bool     # Whether value function converged within tolerance
    v_tol: float         # Tolerance for value function convergence
    v_err: float         # Final change in value function between iterations
    log: object          # Detailed log of convergence progress
    trace: object        # Optional trace of value/policy evolution


@dataclass
class ImprovedTimeIterationResult(AlgoResult):
    """
    Result from improved time iteration algorithm with endogenous grid.
    
    Combines time iteration with EGM ideas as described in finite_iteration.md.
    Particularly useful for models with:
    - Occasionally binding constraints
    - Forward-looking expectations
    - Multiple control variables
    
    Example from rbc.yaml with labor-leisure choice:
    ```yaml
    equations:
        arbitrage:
            - 1 - β*(c[t]/c[t+1])*(1-δ + rk[t+1])  | 0 <= i[t] <= inf
            - w[t] - χ*n[t]^η*c[t]                  | 0 <= n[t] <= 1
    ```
    """
    dr: object           # Decision rule for optimal controls (from dolo.numeric.decision_rule)
    N: int              # Number of iterations performed
    f_x: float          # Forward difference in controls (for convergence check)
    d_x: float          # Backward difference in controls (for convergence check)
    x_converged: bool   # Whether solution converged within tolerance
    complementarities: bool  # Whether model has complementarity conditions
    radius: float       # Trust region radius for policy updates
    trace_data: object  # Detailed trace of iteration process
    L: object          # Additional convergence diagnostics


@dataclass
class PerturbationResult(AlgoResult):
    """
    Result from perturbation method (local approximation around steady state).
    
    Implements perturbation analysis as described in perturbation.md.
    Provides a local approximation by:
    1. Computing steady state
    2. Taking derivatives of model equations
    3. Solving linear system for policy derivatives
    
    Example from rbc.yaml linearized around steady state:
    ```yaml
    steady_state:
        - 1 - β*(1-δ + α*(k/n)^(α-1))
        - χ*n^η*c - (1-α)*(k/n)^α
        - y - k^α * n^(1-α)
        - i - δ*k
        - y - c - i
    ```
    """
    dr: object                    # Decision rule (Taylor expansion) (from dolo.numeric.decision_rule)
    generalized_eigenvalues: object  # System eigenvalues for stability analysis
    stable: bool                  # Whether solution is stable (max eigenvalue < 1)
    determined: bool              # Whether solution is well-determined (unique)
    unique: bool                  # Whether solution is unique (no indeterminacy)
