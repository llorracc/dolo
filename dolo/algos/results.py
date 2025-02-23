# Base class for algorithm results
class AlgoResult:
    pass


from dataclasses import dataclass


@dataclass
class TimeIterationResult(AlgoResult):
    # Result class for time iteration algorithm
    dr: object              # Decision rule object
    iterations: int         # Number of iterations performed
    complementarities: bool # Whether complementarity conditions were enforced
    dprocess: object       # Discretized process object
    x_converged: bool      # Whether controls converged
    x_tol: float          # Tolerance level for controls
    err: float            # Final error value
    log: object           # Log of iterations
    trace: object         # Detailed trace of iterations


@dataclass
class EGMResult(AlgoResult):
    # Result class for endogenous grid method
    dr: object           # Decision rule object
    iterations: int      # Number of iterations performed
    dprocess: object     # Discretized process object
    a_converged: bool    # Whether assets converged
    a_tol: float
    a_tol: float        # Tolerance level for assets
    # log: object  # TimeIterationLog
    err: float          # Final error value


@dataclass
class ValueIterationResult(AlgoResult):
    # Result class for value iteration algorithm
    dr: object          # Decision rule object
    drv: object         # Value function decision rule
    iterations: int     # Number of iterations performed
    dprocess: object    # Discretized process object
    x_converged: object # Whether controls converged
    x_tol: float       # Tolerance level for controls
    x_err: float       # Error in controls
    v_converged: bool   # Whether value function converged
    v_tol: float       # Tolerance level for value function
    v_err: float       # Error in value function
    log: object        # Log of iterations
    trace: object      # Detailed trace of iterations


@dataclass
class ImprovedTimeIterationResult(AlgoResult):
    # Result class for improved time iteration algorithm
    dr: object          # Decision rule object
    N: int             # Number of grid points
    f_x: float         # Function value at solution
    d_x: float         # Derivative value at solution
    x_converged: bool  # Whether controls converged
    complementarities: bool  # Whether complementarity conditions were enforced
    radius: float      # Trust region radius
    trace_data: object # Trace of algorithm progress
    L: object         # Linear operator


@dataclass
class PerturbationResult(AlgoResult):
    # Result class for perturbation analysis
    dr: object                  # Decision rule (bi-taylor expansion)
    generalized_eigenvalues: object  # Eigenvalues of linearized system
    stable: bool               # Whether largest eigenvalue < 1
    determined: bool           # Whether next eigenvalue > largest + epsilon
    unique: bool              # Whether next eigenvalue > 1
