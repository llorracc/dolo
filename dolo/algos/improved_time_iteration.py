"""
Improved time iteration solver for dynamic economic models.

This module implements an enhanced version of the time iteration algorithm that:
- Uses more sophisticated update methods for faster convergence
- Handles occasionally binding constraints more robustly
- Provides better error control and diagnostics
- Supports adaptive step sizes and dampening

The improved algorithm generally converges faster than standard time iteration
while maintaining numerical stability.
"""

from .bruteforce_lib import *  # For brute force solution methods
from .invert import *  # For matrix inversion utilities

from dolo.compiler.model import Model  # For model class definition
from dolo.numeric.decision_rule import DecisionRule  # For policy function representation
from dolo.misc.itprinter import IterationsPrinter  # For iteration output formatting
from numba import jit  # For function optimization
import numpy  # For numerical computations
import time  # For timing operations
import scipy.sparse.linalg  # For sparse linear algebra operations

from operator import mul  # For multiplication operator
from functools import reduce  # For reducing sequences

from dolo.numeric.optimize.newton import SerialDifferentiableFunction  # For automatic differentiation


def prod(l):  # Compute product of list elements
    """
    Compute the product of elements in a list.
    
    Uses functools.reduce with multiplication operator to efficiently
    compute the product of all elements in a sequence.
    
    Parameters
    ----------
    l : list or sequence
        Sequence of numbers to multiply together
        
    Returns
    -------
    number
        Product of all elements in the sequence
        
    Examples
    --------
    >>> prod([2, 3, 4])
    24
    """
    return reduce(mul, l)  # Use reduce with multiplication


from math import sqrt  # For numerical operations
from numba import jit  # For function optimization
import time  # For timing operations
from numpy import array, zeros  # For array operations

import time  # For timing operations


@jit  # Optimize with numba
def inplace(Phi, J):  # Multiply tensors in-place
    """
    Multiply tensors in-place for efficient memory usage.
    
    Performs element-wise multiplication of a 3D tensor Phi with a 5D tensor J,
    storing the result in J. This is used for efficiently updating Jacobian
    matrices during the iteration process.
    
    Parameters
    ----------
    Phi : array
        3D tensor of shape (a, c, d)
    J : array
        5D tensor of shape (a, b, c, d, e) that will be modified in-place
        
    Notes
    -----
    Uses numba @jit decorator for performance optimization.
    The operation performed is:
        J[i_a, i_b, i_c, i_d, i_e] *= Phi[i_a, i_c, i_d]
    for all valid indices.
    """
    a, b, c, d, e = J.shape  # Get tensor dimensions
    for i_a in range(a):  # Loop over first dimension
        for i_b in range(b):  # Loop over second dimension
            for i_c in range(c):  # Loop over third dimension
                for i_d in range(d):  # Loop over fourth dimension
                    for i_e in range(e):  # Loop over fifth dimension
                        J[i_a, i_b, i_c, i_d, i_e] *= Phi[i_a, i_c, i_d]  # Multiply elements


def smooth(res, dres, jres, dx, pos=1.0):  # Smooth residuals for complementarity problems
    """
    Apply smoothing to residuals and derivatives for complementarity problems.
    
    Uses a smooth approximation to handle inequality constraints in the model.
    The smoothing helps avoid numerical issues at constraint boundaries while
    maintaining differentiability.
    
    Parameters
    ----------
    res : array
        Residuals to smooth
    dres : array
        Derivatives of residuals
    jres : array
        Jacobian of residuals
    dx : array
        Distance to constraint boundary
    pos : float, default=1.0
        Direction of smoothing (+1 for upper bounds, -1 for lower bounds)
        
    Returns
    -------
    H : array
        Smoothed residuals
    H_x : array
        Smoothed derivatives
    jres : array
        Smoothed Jacobian
        
    Notes
    -----
    Uses the Fischer-Burmeister smoothing function to approximate
    complementarity conditions while maintaining differentiability.
    """
    from numpy import sqrt  # For numerical operations
    dinf = dx > 100000  # Check for infinite values
    n_m, N, n_x = res.shape  # Get dimensions
    sq = sqrt(res**2 + (dx) ** 2)  # Compute smoothing term
    H = res + (dx) - sq  # Apply smoothing
    Phi_a = 1 - res / sq  # Compute first derivative
    Phi_b = 1 - (dx) / sq  # Compute second derivative
    
    # Handle infinite cases
    H[dinf] = res[dinf]  # Use residual directly
    Phi_a[dinf] = 1.0  # Set derivative to 1
    Phi_b[dinf] = 0.0  # Set derivative to 0
    
    # Compute derivatives
    H_x = Phi_a[:, :, :, None] * dres  # First derivative term
    for i_x in range(n_x):  # For each control
        H_x[:, :, i_x, i_x] += Phi_b[:, :, i_x] * pos  # Add second derivative term
    
    # H_xt = Phi_a[:,None,:,:,None]*jres
    inplace(Phi_a, jres)  # Update Jacobian in-place
    return H, H_x, jres  # Return smoothed values and derivatives


def smooth_nodiff(res, dx):  # Smooth residuals without derivatives
    """
    Apply smoothing to residuals without computing derivatives.
    
    A simplified version of the smooth() function that only computes
    smoothed residuals, without derivatives or Jacobians. Used in line
    search where derivatives are not needed.
    
    Parameters
    ----------
    res : array
        Residuals to smooth
    dx : array
        Distance to constraint boundary
        
    Returns
    -------
    array
        Smoothed residuals
        
    Notes
    -----
    Uses the same Fischer-Burmeister smoothing function as smooth(),
    but avoids computing derivatives for efficiency.
    """
    from numpy import sqrt  # For numerical operations

    n_m, N, n_x = res.shape  # Get dimensions
    dinf = dx > 100000  # Check for infinite values
    sq = sqrt(res**2 + (dx) ** 2)  # Compute smoothing term
    H = res + (dx) - sq  # Apply smoothing
    H[dinf] = res[dinf]  # Handle infinite cases
    return H  # Return smoothed values


@jit  # Optimize with numba
def ssmul(A, B):  # Simple serial matrix multiplication
    """
    Perform simple serial matrix multiplication optimized with numba.
    
    Implements a basic matrix multiplication algorithm for 3D tensors
    without using vectorized operations. This can be more efficient for
    small matrices due to reduced overhead.
    
    Parameters
    ----------
    A : array
        3D tensor of shape (N, a, b)
    B : array
        2D matrix of shape (N, b)
        
    Returns
    -------
    array
        Result matrix of shape (N, a)
        
    Notes
    -----
    Uses @jit decorator for performance optimization.
    Implements the operation:
        O[n,k] = sum(A[n,k,l] * B[n,l] for l in range(b))
    """
    # simple serial_mult (matrix times vector)
    N, a, b = A.shape  # Get matrix dimensions
    NN, b = B.shape  # Get vector dimensions
    O = numpy.zeros((N, a))  # Initialize output array
    for n in range(N):  # For each row
        for k in range(a):  # For each output element
            for l in range(b):  # For each inner dimension
                O[n, k] += A[n, k, l] * B[n, l]  # Accumulate product
    return O  # Return result


@jit  # Optimize with numba
def ssmul_inplace(A, B, O):  # In-place serial matrix multiplication
    """
    Perform in-place serial matrix multiplication optimized with numba.
    
    Similar to ssmul() but modifies an existing output array instead of
    allocating a new one. This can be more efficient when the output
    array can be reused.
    
    Parameters
    ----------
    A : array
        3D tensor of shape (N, a, b)
    B : array
        2D matrix of shape (N, b)
    O : array
        Pre-allocated output array of shape (N, a) to store result
        
    Returns
    -------
    array
        Modified output array O containing the matrix product
        
    Notes
    -----
    Uses @jit decorator for performance optimization.
    Implements the operation:
        O[n,k] += sum(A[n,k,l] * B[n,l] for l in range(b))
    modifying O in-place.
    """
    # simple serial_mult (matrix times vector)
    N, a, b = A.shape  # Get matrix dimensions
    NN, b = B.shape  # Get vector dimensions
    for n in range(N):  # For each row
        for k in range(a):  # For each output element
            for l in range(b):  # For each inner dimension
                O[n, k] += A[n, k, l] * B[n, l]  # Accumulate product
    return O  # Return modified array


def d_filt_dx(π, M_ij, S_ij, n_m, N, n_x, dumdr):  # Filter derivatives
    """
    Filter derivatives through transition matrices and policy evaluation.
    
    [M @ π](m,s) = E[M(m,M) @ π(M,g(m,s,x,M))]
    
    where E[·] is computed by discretizing the continuous state space
    and taking weighted sums over the discrete points.
    
    Usage Contexts
    -------------
    1. Policy Updates (invert_jac):
       - M(m,M) = (∂f/∂x)^{-1} @ ∂f/∂X
       - π is the policy update
       - Implements the iteration xₖ₊₁ = M @ xₖ + x₀
    
    2. Spectral Radius (radius_jac):
       - Same M(m,M) as above
       - π is the current eigenvector estimate
       - Used in power iteration vₖ₊₁ = M @ vₖ
    
    Parameters
    ----------
    π : array
        Policy function to filter, shape (n_m, N, n_x)
    M_ij : array
        Transition matrices for each state combination
    S_ij : array
        Future state values
    n_m : int
        Number of exogenous states
    N : int
        Number of grid points
    n_x : int
        Number of control variables
    dumdr : DecisionRule
        Decision rule for policy evaluation
        
    Returns
    -------
    array
        Filtered policy function, same shape as input
        
    Notes
    -----
    This is a core operation in the improved time iteration algorithm,
    implementing the filtering step that propagates policy updates.
    """
    n_m, n_im = M_ij.shape[:2]  # Get dimensions
    dumdr.set_values(π)  # Update decision rule
    i = 0  # Initialize indices
    j = 0
    for i in range(n_m):  # For each exogenous state
        π[i, :, :] = 0  # Reset values
        for j in range(n_im):  # For each future state
            A = M_ij[i, j, :, :, :]  # Get transition matrix
            B = dumdr.eval_ijs(i, j, S_ij[i, j, :, :])  # Evaluate policy
            π[i, :, :] += ssmul(A, B)  # Accumulate filtered values
    return π  # Return filtered derivatives


from scipy.sparse.linalg import LinearOperator


class Operator(LinearOperator):  # Linear operator for solving system
    """
    Linear operator for solving policy iteration equations.
    
    Implements a specialized linear operator that applies the policy iteration
    mapping to a policy function. This operator is used in the improved time
    iteration algorithm to solve for policy updates more efficiently.
    
    Parameters
    ----------
    M_ij : array
        Transition matrices for each state and future state combination
    S_ij : array
        Future state values for each state combination
    dumdr : DecisionRule
        Decision rule for evaluating policies
        
    Attributes
    ----------
    n_m : int
        Number of exogenous states
    N : int
        Number of grid points
    n_x : int
        Number of control variables
    counter : int
        Number of operator applications
    addid : bool
        Whether to add identity matrix for preconditioning
        
    Notes
    -----
    The operator implements matrix-vector multiplication through policy
    evaluation and filtering. It is used with iterative solvers like GMRES
    to find policy updates efficiently.
    """

    def __init__(self, M_ij, S_ij, dumdr):  # Initialize operator
        """
        Initialize the linear operator for policy iteration.
        
        Parameters
        ----------
        M_ij : array
            Transition matrices for each state and future state combination
        S_ij : array
            Future state values for each state combination
        dumdr : DecisionRule
            Decision rule for evaluating policies
            
        Notes
        -----
        Sets up internal attributes including:
        - Matrix dimensions from M_ij shape
        - Counter for operator applications
        - Flag for identity matrix addition
        - Data type for numerical operations
        """
        self.M_ij = M_ij  # Store transition matrices
        self.S_ij = S_ij  # Store future states
        self.n_m = M_ij.shape[0]  # Number of exogenous states
        self.N = M_ij.shape[2]  # Number of grid points
        self.n_x = M_ij.shape[3]  # Number of controls
        self.dumdr = dumdr  # Store decision rule
        self.dtype = numpy.dtype("float64")  # Set data type
        self.counter = 0  # Initialize counter
        self.addid = False  # Identity addition flag

    @property
    def shape(self):  # Get operator dimensions
        """
        Get the shape of the linear operator as a matrix.
        
        Returns the dimensions of the operator when viewed as a square matrix.
        The size is (n_m * N * n_x, n_m * N * n_x) where:
        - n_m is the number of exogenous states
        - N is the number of grid points
        - n_x is the number of control variables
        
        Returns
        -------
        tuple
            (total_size, total_size) where total_size = n_m * N * n_x
        """
        nn = self.n_m * self.N * self.n_x  # Total size of flattened system
        return (nn, nn)  # Return square matrix shape

    def _matvec(self, x):  # Matrix-vector multiplication
        """
        Implement matrix-vector multiplication for the linear operator.
        
        Required method for scipy.sparse.linalg.LinearOperator interface.
        Applies the operator to a flattened vector by:
        1. Reshaping input to policy function form
        2. Applying the policy iteration mapping
        3. Optionally subtracting from identity for preconditioning
        4. Flattening result back to vector form
        
        Parameters
        ----------
        x : array
            Input vector of size (n_m * N * n_x)
            
        Returns
        -------
        array
            Result vector of size (n_m * N * n_x)
        """
        self.counter += 1  # Increment operation counter
        xx = x.reshape((self.n_m, self.N, self.n_x))  # Reshape input to 3D
        yy = self.apply(xx)  # Apply operator
        if self.addid:  # If using preconditioned system
            yy = xx - yy  # Subtract from identity
        return yy.ravel()  # Return flattened result

    def apply(self, π, inplace=False):  # Apply operator to policy
        """
        Apply the policy iteration operator to a policy function.
        
        Applies the operator by evaluating the policy at future states and
        filtering through transition matrices. Can optionally modify the
        input policy in-place.
        
        Parameters
        ----------
        π : array
            Policy function to apply operator to, shape (n_m, N, n_x)
        inplace : bool, default=False
            Whether to modify input array in-place
            
        Returns
        -------
        array
            Result of applying operator, same shape as input
            
        Notes
        -----
        The operator combines policy evaluation with filtering through
        transition matrices to implement one step of policy iteration.
        """
        M_ij = self.M_ij  # Get transition matrices
        S_ij = self.S_ij  # Get future states
        n_m = self.n_m  # Number of exogenous states
        N = self.N  # Number of grid points
        n_x = self.n_x  # Number of controls
        dumdr = self.dumdr  # Get decision rule
        if not inplace:  # If not modifying input
            π = π.copy()  # Make a copy
        return d_filt_dx(π, M_ij, S_ij, n_m, N, n_x, dumdr)  # Apply filter

    def as_matrix(self):  # Convert operator to explicit matrix
        """
        Convert the linear operator to an explicit matrix representation.
        
        Constructs the full matrix representation by applying the operator
        to unit vectors. This is mainly used for debugging and analysis
        since the explicit matrix can be very large.
        
        Returns
        -------
        array
            Full matrix representation of the operator with shape
            (n_m * N * n_x, n_m * N * n_x)
            
        Notes
        -----
        This method should only be used for small problems as it requires
        storing the full matrix which can be memory intensive. For large
        problems, use the matrix-free operator interface instead.
        """
        arg = np.zeros((self.n_m, self.N, self.n_x))  # Initialize input
        larg = arg.ravel()  # Flatten input
        N = len(larg)  # Get total size
        J = numpy.zeros((N, N))  # Initialize Jacobian matrix
        for i in range(N):  # For each column
            if i > 0:  # After first column
                larg[i - 1] = 0.0  # Reset previous entry
            larg[i] = 1.0  # Set current entry
            J[:, i] = self.apply(arg).ravel()  # Apply operator and store
        return J  # Return full matrix


def invert_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):  # Invert Jacobian matrix
    """
    Iteratively invert Jacobian matrix for improved time iteration.
    
    Uses a fixed point iteration method to invert the Jacobian matrix without
    explicitly forming the full matrix. This is more memory efficient than
    direct inversion for large problems.
    
    Parameters
    ----------
    res : array
        Current residuals
    dres : array
        Derivatives with respect to current controls
    jres : array
        Jacobian with respect to future controls
    fut_S : array
        Future state values
    dumdr : DecisionRule
        Decision rule for policy evaluation
    tol : float, default=1e-10
        Convergence tolerance
    maxit : int, default=1000
        Maximum iterations
    verbose : bool, default=False
        Whether to print iteration progress
        
    Returns
    -------
    tot : array
        Solution of the linear system
    nn : int
        Number of iterations taken
    lam : float
        Final eigenvalue estimate
        
    Notes
    -----
    The method uses a power iteration approach to solve the linear system,
    which is particularly efficient when the Jacobian has good spectral properties.
    """

    n_m = res.shape[0]  # Number of exogenous states
    N = res.shape[1]  # Number of grid points
    n_x = res.shape[2]  # Number of controls

    err0 = 0.0  # Initialize error
    ddx = solve_gu(dres.copy(), res.copy())  # Get initial direction

    lam = -1.0  # Initialize eigenvalue
    lam_max = -1.0  # Initialize maximum eigenvalue
    err_0 = abs(ddx).max()  # Compute initial error

    tot = ddx.copy()  # Initialize total update
    if verbose:  # If verbose output requested
        print("Starting inversion")  # Print start message
    for nn in range(maxit):  # Main inversion loop
        ddx = d_filt_dx(ddx, jres, fut_S, n_m, N, n_x, dumdr)  # Apply filter
        err = abs(ddx).max()  # Compute error
        lam = err / err_0  # Compute eigenvalue
        lam_max = max(lam_max, lam)  # Update maximum eigenvalue
        if verbose:  # If verbose output requested
            print("- {} | {} | {}".format(err, lam, lam_max))  # Print progress
        tot += ddx  # Accumulate update
        err_0 = err  # Update error
        if err < tol:  # If converged
            break  # Exit loop

    # tot += ddx*lam/(1-lam)
    return tot, nn, lam  # Return solution and stats


def radius_jac(res, dres, jres, fut_S, dumdr, tol=1e-10, maxit=1000, verbose=False):  # Compute spectral radius
    """
    Compute spectral radius of the Jacobian operator using power iteration.
    
    Uses power iteration to estimate the largest eigenvalue of the Jacobian
    operator, which determines the convergence properties of the algorithm.
    
    Parameters
    ----------
    res : array
        Current residuals
    dres : array
        Derivatives with respect to current controls
    jres : array
        Jacobian with respect to future controls
    fut_S : array
        Future state values
    dumdr : DecisionRule
        Decision rule for policy evaluation
    tol : float, default=1e-10
        Convergence tolerance
    maxit : int, default=1000
        Maximum iterations
    verbose : bool, default=False
        Whether to print iteration progress
        
    Returns
    -------
    tuple
        Contains:
        - lam : float
            Final eigenvalue estimate
        - lam_max : float
            Maximum eigenvalue seen during iteration
        - lambdas : list
            History of eigenvalue estimates
        
    Notes
    -----
    The spectral radius determines the local convergence rate of the
    time iteration algorithm. A radius less than 1 indicates local convergence.
    """

    from numpy import sqrt  # For norm computation

    n_m = res.shape[0]  # Number of exogenous states
    N = res.shape[1]  # Number of grid points
    n_x = res.shape[2]  # Number of controls

    err0 = 0.0  # Initialize error

    norm2 = lambda m: sqrt((m**2).sum())  # Define L2 norm

    import numpy.random  # For random initialization

    π = (numpy.random.random(res.shape) * 2 - 1) * 1  # Random initial vector
    π /= norm2(π)  # Normalize vector

    verbose = True  # Enable verbose output
    lam = 1.0  # Initialize eigenvalue
    lam_max = 0.0  # Initialize maximum eigenvalue

    lambdas = []  # Store eigenvalue history
    if verbose:  # If verbose output requested
        print("Starting inversion")  # Print start message
    for nn in range(maxit):  # Power iteration loop
        # operations are made in place in ddx
        # π = (numpy.random.random(res.shape)*2-1)*1
        # π /= norm2(π)
        π[:, :, :] /= lam  # Normalize by eigenvalue
        π = d_filt_dx(π, jres, fut_S, n_m, N, n_x, dumdr)  # Apply operator
        lam = norm2(π)  # Compute new eigenvalue
        lam_max = max(lam_max, lam)  # Update maximum eigenvalue
        if verbose:  # If verbose output requested
            print("- {} | {}".format(lam, lam_max))  # Print progress
        lambdas.append(lam)  # Store eigenvalue
    return (lam, lam_max, lambdas)  # Return eigenvalue info


from dolo import dprint
from .results import AlgoResult, ImprovedTimeIterationResult


def improved_time_iteration(
    model: Model,
    *,
    dr0: DecisionRule = None,
    verbose: bool = True,
    details: bool = True,
    ignore_constraints=False,
    method="jac",
    dprocess=None,
    interp_method="cubic",
    mu=2,
    maxbsteps=10,
    tol=1e-8,
    smaxit=500,
    maxit=1000,
    compute_radius=False,
    invmethod="iti",
    complementarities=None
) -> ImprovedTimeIterationResult:
    """
    Solve model using improved time iteration algorithm.
    
    An enhanced version of standard time iteration that uses more sophisticated
    numerical methods for solving the nonlinear equations at each iteration.
    
    Parameters
    ----------
    model : Model
        Model to solve
    dr0 : DecisionRule, optional
        Initial guess for decision rule
    verbose : bool, default=True
        Whether to print iteration progress
    details : bool, default=True
        Whether to return detailed solution results
    ignore_constraints : bool, default=False
        Whether to ignore complementarity constraints
    method : str, default='jac'
        Solution method for linear systems
    dprocess : Process, optional
        Custom discretization process
    interp_method : str, default='cubic'
        Interpolation method for decision rules
    mu : float, default=2
        Damping parameter for iterations
    maxbsteps : int, default=10
        Maximum number of backsteps in line search
    tol : float, default=1e-8
        Convergence tolerance
    smaxit : int, default=500
        Maximum secondary iterations
    maxit : int, default=1000
        Maximum main iterations
    compute_radius : bool, default=False
        Whether to compute convergence radius
    invmethod : str, default='iti'
        Method for inverting Jacobian ('iti' or 'gmres')
        
    Returns
    -------
    ImprovedTimeIterationResult
        Object containing:
        - Decision rule
        - Number of iterations
        - Final errors
        - Convergence info
        - Linear operator
    """

    # Handle deprecated option
    if complementarities is not None:  # If old option used
        pass  # TODO: add warning
    else:
        complementarities = not ignore_constraints  # Set from new option

    def vprint(*args, **kwargs):  # Helper for verbose output
        if verbose:  # If verbose mode enabled
            print(*args, **kwargs)  # Print message

    itprint = IterationsPrinter(  # Setup iteration printer
        ("N", int),  # Iteration number
        ("f_x", float),  # Function value
        ("d_x", float),  # Step size
        ("Time_residuals", float),  # Time for residual computation
        ("Time_inversion", float),  # Time for matrix inversion
        ("Time_search", float),  # Time for line search
        ("Lambda_0", float),  # Initial lambda value
        ("N_invert", int),  # Number of inversions
        ("N_search", int),  # Number of line searches
        verbose=verbose,  # Whether to print output
    )
    itprint.print_header("Start Improved Time Iterations.")  # Print header

    f = model.functions["arbitrage"]  # Get arbitrage equations
    g = model.functions["transition"]  # Get transition equations
    x_lb = model.functions["arbitrage_lb"]  # Get control lower bounds
    x_ub = model.functions["arbitrage_ub"]  # Get control upper bounds

    parms = model.calibration["parameters"]  # Get model parameters

    grid, dprocess_ = model.discretize()  # Discretize state space

    if dprocess is None:  # If no custom discretization
        dprocess = dprocess_  # Use default discretization

    endo_grid = grid["endo"]  # Get endogenous grid
    exo_grid = grid["exo"]  # Get exogenous grid

    n_m = max(dprocess.n_nodes, 1)  # Number of exogenous states
    n_s = len(model.symbols["states"])  # Number of state variables

    if interp_method in ("cubic", "linear"):  # If using standard interpolation
        ddr = DecisionRule(exo_grid,
                          endo_grid,
                          dprocess=dprocess,
                          interp_method=interp_method)
        ddr_filt = DecisionRule(exo_grid,
                               endo_grid,
                               dprocess=dprocess,
                               interp_method=interp_method)
    else:
        raise Exception("Unsupported interpolation method.")  # Invalid method

    s = endo_grid.nodes  # Get grid nodes
    N = s.shape[0]  # Number of grid points
    n_x = len(model.symbols["controls"])  # Number of controls
    x0 = (model.calibration["controls"][None, None,]
          .repeat(n_m, axis=0)
          .repeat(N, axis=1))

    if dr0 is not None:  # If initial guess provided
        for i_m in range(n_m):  # For each exogenous state
            x0[i_m, :, :] = dr0.eval_is(i_m, s)  # Evaluate initial guess
    ddr.set_values(x0)  # Set initial values

    steps = 0.5 ** numpy.arange(maxbsteps)  # Create backstep sequence

    lb = x0.copy()  # Initialize lower bounds
    ub = x0.copy()  # Initialize upper bounds
    for i_m in range(n_m):  # For each exogenous state
        m = dprocess.node(i_m)  # Get exogenous value
        lb[i_m, :] = x_lb(m, s, parms)  # Compute lower bounds
        ub[i_m, :] = x_ub(m, s, parms)  # Compute upper bounds

    x = x0  # Initialize current solution

    ddr.set_values(x)  # Update decision rule

    # both affect the precision

    # Memory allocation
    n_im = dprocess.n_inodes(0)  # we assume it is constant for now

    jres = numpy.zeros((n_m, n_im, N, n_x, n_x))  # Initialize Jacobian residuals
    S_ij = numpy.zeros((n_m, n_im, N, n_s))  # Initialize future states

    for it in range(maxit):  # Main iteration loop

        jres[...] = 0.0  # Reset Jacobian residuals
        S_ij[...] = 0.0  # Reset future states

        t1 = time.time()  # Start timing

        ddr.set_values(x)  # Update decision rule

        #
        # ub[ub>100000] = 100000
        # lb[lb<-100000] = -100000
        #
        # sh_x = x.shape
        # rr =euler_residuals(f,g,s,x,ddr,dp,parms, diff=False, with_jres=False,set_dr=True)
        # print(rr.shape)
        #
        # from iti.fb import smooth_
        # jj = smooth_(rr, x, lb, ub)
        #
        # print("Errors with complementarities")
        # print(abs(jj.max()))
        #
        # exit()
        #
        #

        # compute derivatives and residuals:
        # res: residuals
        # dres: derivatives w.r.t. x
        # jres: derivatives w.r.t. ~x
        # fut_S: future states

        from dolo.numeric.optimize.newton import SerialDifferentiableFunction

        sh_x = x.shape  # Get solution shape
        ff = SerialDifferentiableFunction(  # Create differentiable function
            lambda u: euler_residuals(f,g,s,u.reshape(sh_x),ddr,dprocess,parms,
                                    diff=False,with_jres=False,set_dr=False).reshape((-1,sh_x[2]))
        )
        res, dres = ff(x.reshape((-1, sh_x[2])))  # Compute residuals and derivatives
        res = res.reshape(sh_x)  # Reshape residuals
        dres = dres.reshape((sh_x[0], sh_x[1], sh_x[2], sh_x[2]))  # Reshape derivatives
        junk, jres, fut_S = euler_residuals(  # Get Jacobian and future states
            f,
            g,
            s,
            x,
            ddr,
            dprocess,
            parms,
            diff=False,
            with_jres=True,
            set_dr=False,
            jres=jres,
            S_ij=S_ij,
        )

        # Handle complementarity constraints
        if complementarities:  # If using bounds
            # if there are complementerities, we modify derivatives
            res, dres, jres = smooth(res, dres, jres, x - lb)  # Smooth lower bound
            res[...] *= -1  # Flip signs
            dres[...] *= -1  # Flip derivatives
            jres[...] *= -1  # Flip Jacobian
            res, dres, jres = smooth(res, dres, jres, ub - x, pos=-1.0)  # Smooth upper bound
            res[...] *= -1  # Flip signs back
            dres[...] *= -1  # Flip derivatives back
            jres[...] *= -1  # Flip Jacobian back

        err_0 = abs(res).max()  # Compute maximum residual

        jres[...] *= -1.0  # Prepare for solving

        # premultiply by A

        for i_m in range(n_m):  # For each exogenous state
            for j_m in range(n_im):  # For each future state
                M = jres[i_m, j_m, :, :, :]  # Get Jacobian block
                X = dres[i_m, :, :, :].copy()  # Get derivative block
                sol = solve_tensor(X, M)  # Solve linear system

        t2 = time.time()  # End timing block

        # Choose solution method
        if invmethod == "gmres":  # If using GMRES solver
            ddx = solve_gu(dres.copy(), res.copy())  # Get initial direction
            L = Operator(jres, fut_S, ddr_filt)  # Create linear operator
            n0 = L.counter  # Store initial count
            L.addid = True  # Enable preconditioning
            ttol = err_0 / 100  # Set solver tolerance
            sol = scipy.sparse.linalg.gmres(  # Solve linear system
                L, ddx.ravel(), tol=ttol, maxiter=1, restart=smaxit
            )
            lam0 = 0.01  # Set initial step size
            # operations are made in place in ddx
            # π = (numpy.random.random(res.shape)*2-1)*1
            # π /= norm2(π)
            nn = L.counter - n0  # Count iterations
            tot = sol[0].reshape(ddx.shape)  # Get solution
        else:  # If using direct inversion
            tot, nn, lam0 = invert_jac(  # Invert Jacobian
                res,
                dres,
                jres,
                fut_S,
                ddr_filt,
                tol=tol,
                maxit=smaxit,
                verbose=(verbose == "full"),
            )

        # lam, lam_max, lambdas = radius_jac(res,dres,jres,fut_S,tol=tol,maxit=1000,verbose=(verbose=='full'),filt=ddr_filt)

        # Perform line search with backsteps
        t3 = time.time()  # Start line search timing
        for i_bckstps, lam in enumerate(steps):  # Try different step sizes
            new_x = x - tot * lam  # Take trial step
            new_err = euler_residuals(  # Evaluate residuals at new point
                f, g, s, new_x, ddr, dprocess, parms, diff=False, set_dr=True
            )

            if complementarities:  # If using bounds
                new_err = smooth_nodiff(new_err, new_x - lb)  # Smooth lower bound
                new_err = smooth_nodiff(-new_err, ub - new_x)  # Smooth upper bound

            new_err = abs(new_err).max()  # Compute maximum error
            if new_err < err_0:  # If error decreased
                break  # Accept step

        err_2 = abs(tot).max()  # Compute step size
        t4 = time.time()  # End line search timing
        itprint.print_iteration(  # Print iteration info
            N=it,  # Iteration number
            f_x=err_0,  # Function value
            d_x=err_2,  # Step size
            Time_residuals=t2 - t1,  # Residual computation time
            Time_inversion=t3 - t2,  # Matrix inversion time
            Time_search=t4 - t3,  # Line search time
            Lambda_0=lam0,  # Initial lambda
            N_invert=nn,  # Number of inversions
            N_search=i_bckstps,  # Number of backsteps
        )
        if err_0 < tol:  # If converged
            break  # Exit iteration loop

        x = new_x  # Update solution

    ddr.set_values(x)  # Set final values

    itprint.print_finished()  # Print completion message

    if not details:  # If only basic output requested
        return ddr  # Return decision rule
    else:  # If detailed output requested
        ddx = solve_gu(dres.copy(), res.copy())  # Get final direction
        L = Operator(jres, fut_S, ddr_filt)  # Create linear operator

        if compute_radius:  # If computing convergence radius
            lam = scipy.sparse.linalg.eigs(L, k=1, return_eigenvectors=False)  # Compute eigenvalue
            lam = abs(lam[0])  # Get magnitude
        else:
            lam = np.nan  # No radius computed

        return ImprovedTimeIterationResult(  # Return full results
            ddr,  # Decision rule
            it,  # Iterations
            err_0,  # Final error
            err_2,  # Step size
            err_0 < tol,  # Convergence flag
            complementarities,  # Whether bounds used
            lam,  # Convergence radius
            None,  # No log
            L  # Linear operator
        )


def euler_residuals(  # Compute Euler equation residuals
    f,  # Arbitrage function
    g,  # Transition function
    s,  # State variables
    x,  # Control variables
    dr,  # Decision rule
    dp,  # Discretized process
    p_,  # Parameters
    diff=True,  # Whether to compute derivatives
    with_jres=False,  # Whether to compute Jacobian
    set_dr=True,  # Whether to update decision rule with current controls
    jres=None,  # Pre-allocated Jacobian
    S_ij=None,  # Pre-allocated future states
):
    """
    Compute residuals of Euler equations for improved time iteration.
    
    Evaluates the residuals of the Euler equations at each grid point and state,
    optionally computing derivatives for use in the iterative solver.
    
    Parameters
    ----------
    f : callable
        Arbitrage equation function
    g : callable
        State transition function
    s : array
        Current state values
    x : array
        Current control values
    dr : DecisionRule
        Current decision rule
    dp : Process
        Discretized exogenous process
    p_ : array
        Model parameters
    diff : bool, default=True
        Whether to compute derivatives
    with_jres : bool, default=False
        Whether to compute and return Jacobian
    set_dr : bool, default=True
        Whether to update decision rule with current controls
    jres : array, optional
        Pre-allocated array for Jacobian results
    S_ij : array, optional
        Pre-allocated array for future states
        
    Returns
    -------
    res : array
        Residuals of Euler equations
    jres : array, optional
        Jacobian of residuals w.r.t. future controls (if with_jres=True)
    S_ij : array, optional
        Future state values (if with_jres=True)
    """

    t1 = time.time()  # Start timing

    if set_dr:  # If updating decision rule
        dr.set_values(x)  # Set new values

    N = s.shape[0]  # Number of grid points
    n_s = s.shape[1]  # Number of states
    n_x = x.shape[2]  # Number of controls

    n_ms = max(dp.n_nodes, 1)  # number of markov states
    n_im = dp.n_inodes(0)  # Number of integration nodes

    res = numpy.zeros_like(x)  # Initialize residuals

    if with_jres:  # If computing Jacobian
        if jres is None:  # If not pre-allocated
            jres = numpy.zeros((n_ms, n_im, N, n_x, n_x))  # Allocate Jacobian
        if S_ij is None:  # If not pre-allocated
            S_ij = numpy.zeros((n_ms, n_im, N, n_s))  # Allocate future states

    for i_ms in range(n_ms):  # For each exogenous state
        m_ = dp.node(i_ms)  # Get current exogenous
        xm = x[i_ms, :, :]  # Get current controls
        for I_ms in range(n_im):  # For each future state
            M_ = dp.inode(i_ms, I_ms)  # Get future exogenous
            w = dp.iweight(i_ms, I_ms)  # Get transition weight
            S = g(m_, s, xm, M_, p_, diff=False)  # Get future state
            XM = dr.eval_ijs(i_ms, I_ms, S)  # Get future controls
            if with_jres:  # If computing Jacobian
                ff = SerialDifferentiableFunction(  # Create differentiable function
                    lambda u: f(m_, s, xm, M_, S, u, p_, diff=False)  # Arbitrage function
                )
                rr, rr_XM = ff(XM)  # Get residuals and derivatives

                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)  # Get residuals
                jres[i_ms, I_ms, :, :, :] = w * rr_XM  # Store weighted Jacobian
                S_ij[i_ms, I_ms, :, :] = S  # Store future states
            else:
                rr = f(m_, s, xm, M_, S, XM, p_, diff=False)  # Get residuals only
            res[i_ms, :, :] += w * rr  # Accumulate weighted residuals

    t2 = time.time()  # End timing

    if with_jres:  # If computing Jacobian
        return res, jres, S_ij  # Return residuals, Jacobian, and states
    else:
        return res  # Return only residuals


class EvaluationResult:  # Container for policy evaluation results
    """
    Container for results from policy evaluation.
    
    Stores the results of evaluating a policy function, including the
    solution, number of iterations taken, tolerance used, and final error.
    
    Parameters
    ----------
    solution : DecisionRule
        The evaluated policy function
    iterations : int
        Number of iterations taken
    tol : float
        Tolerance used for convergence
    error : float
        Final error achieved
        
    Attributes
    ----------
    solution : DecisionRule
        The evaluated policy function
    iterations : int
        Number of iterations taken
    tol : float
        Tolerance used for convergence
    error : float
        Final error achieved
    """
    def __init__(self, solution, iterations, tol, error):  # Initialize result object
        """
        Initialize an EvaluationResult object.
        
        Parameters
        ----------
        solution : DecisionRule
            The evaluated policy function
        iterations : int
            Number of iterations taken
        tol : float
            Tolerance used for convergence
        error : float
            Final error achieved
        """
        self.solution = solution  # Store optimal solution
        self.iterations = iterations  # Store iteration count
        self.tol = tol  # Store tolerance used
        self.error = error  # Store final error
