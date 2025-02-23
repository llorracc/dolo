@dataclass
class AlgoResult:
    """
    Base class for algorithm results.
    
    Provides common interface and functionality for results
    returned by various solution algorithms.
    
    Attributes
    ----------
    solution_type : str
        Type of solution method used
    iterations : int
        Number of iterations taken
    complementarities : bool
        Whether solution handles complementarities
    """
    pass

@dataclass
class PerturbationResult(AlgoResult):
    """
    Result from perturbation method (local approximation around steady state).
    
    Implements perturbation analysis as described in perturbation.md.
    Provides a local approximation by:
    1. Computing steady state
    2. Taking derivatives of model equations
    3. Solving linear system for policy derivatives
    
    Attributes
    ----------
    dr : object
        Decision rule (Taylor expansion) from dolo.numeric.decision_rule
    generalized_eigenvalues : object
        System eigenvalues for stability analysis
    stable : bool
        Whether solution is stable (max eigenvalue < 1)
    determined : bool
        Whether solution is well-determined (unique)
    unique : bool
        Whether solution is unique (no indeterminacy)
        
    Notes
    -----
    The perturbation method:
    1. Linearizes model equations around steady state
    2. Solves resulting linear system for policy derivatives
    3. Checks Blanchard-Kahn conditions for stability
    4. Optionally computes higher-order terms
    
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
    dr: object                    # Decision rule (Taylor expansion)
    generalized_eigenvalues: object  # System eigenvalues for stability analysis
    stable: bool                  # Whether solution is stable
    determined: bool              # Whether solution is well-determined
    unique: bool                  # Whether solution is unique

@dataclass
class TimeIterationResult(AlgoResult):
    """
    Result from time iteration method.
    
    Contains solution and diagnostics from time iteration algorithm
    for solving dynamic models.
    
    Attributes
    ----------
    dr : DecisionRule
        Converged decision rule
    iterations : int
        Number of iterations taken
    complementarities : bool
        Whether solution handles complementarities
    dprocess : Process
        Discretized process for exogenous states
    
    Notes
    -----
    The time iteration method:
    1. Discretizes state space and exogenous process
    2. Iterates on Euler equation residuals
    3. Updates policy function until convergence
    4. Handles occasionally binding constraints
    """
    pass

@dataclass
class ImprovedTimeIterationResult(TimeIterationResult):
    """
    Result from improved time iteration method.
    
    Extends TimeIterationResult with additional diagnostics from
    the improved algorithm using Newton-like updates.
    
    Attributes
    ----------
    dr : DecisionRule
        Converged decision rule
    iterations : int
        Number of iterations taken
    complementarities : bool
        Whether solution handles complementarities
    dprocess : Process
        Discretized process for exogenous states
    spectral_radius : float
        Spectral radius at solution
    
    Notes
    -----
    The improved algorithm:
    1. Uses Newton-like updates for faster convergence
    2. Provides spectral radius for stability analysis
    3. Supports various interpolation methods
    4. Handles complementarity conditions
    """
    pass 