

from dolo.numeric.tensor import sdot, mdot  # For matrix operations in perturbation methods

import numpy as np  # For numerical array operations

TOL = 1e-10  # Tolerance for numerical convergence


# credits : second_order_solver is adapted from Sven Schreiber's port of Uhlig's Toolkit.
def second_order_solver(FF, GG, HH, eigmax=1.0 + 1e-6):  # Solves second-order perturbation equations

    # from scipy.linalg import qz
    from dolo.numeric.extern.qz import qzordered  # For generalized Schur decomposition

    from numpy import (
        array,
        mat,
        c_,
        r_,
        eye,
        zeros,
        real_if_close,
        diag,
        allclose,
        where,
        diagflat,
    )  # Import numpy functions for matrix operations
    from numpy.linalg import solve  # For solving linear systems

    Psi_mat = array(FF)  # Convert first-order derivatives to array
    Gamma_mat = array(-GG)  # Negative transition matrix
    Theta_mat = array(-HH)  # Second-order derivatives
    m_states = FF.shape[0]  # Number of state variables

    Xi_mat = r_[
        c_[Gamma_mat, Theta_mat], c_[eye(m_states), zeros((m_states, m_states))]
    ]  # Build augmented matrix for QZ decomposition

    Delta_mat = r_[
        c_[Psi_mat, zeros((m_states, m_states))],
        c_[zeros((m_states, m_states)), eye(m_states)],
    ]  # Build companion matrix

    [Delta_up, Xi_up, UUU, VVV, eigval] = qzordered(
        Delta_mat,
        Xi_mat,
    )  # Compute ordered QZ decomposition

    VVVH = VVV.T  # Transpose of right eigenvectors
    VVV_2_1 = VVVH[m_states : 2 * m_states, :m_states]  # Extract top-right block
    VVV_2_2 = VVVH[m_states : 2 * m_states, m_states : 2 * m_states]  # Extract bottom-right block
    UUU_2_1 = UUU[m_states : 2 * m_states, :m_states]  # Extract relevant block of U matrix
    PP = -solve(VVV_2_1, VVV_2_2)  # Compute policy function matrix

    # slightly different check than in the original toolkit:
    assert allclose(real_if_close(PP), PP.real)  # Verify solution is real-valued
    PP = PP.real  # Extract real part

    return [eigval, PP]  # Return eigenvalues and policy matrix


def solve_sylvester_vectorized(*args):  # Solve vectorized Sylvester equation
    from numpy import kron  # For Kronecker product operations
    from numpy.linalg import solve  # For linear system solution

    vec = lambda M: M.ravel()  # Helper to flatten matrix to vector
    n = args[0][0].shape[0]  # Get first dimension
    q = args[0][1].shape[0]  # Get second dimension
    K = vec(args[-1])  # Vectorize right-hand side
    L = sum([kron(A, B.T) for (A, B) in args[:-1]])  # Build coefficient matrix
    X = solve(L, -K)  # Solve linear system
    return X.reshape((n, q))  # Reshape solution to matrix form


def solve_sylvester(A, B, C, D, Ainv=None, method="linear"):  # Solve generalized Sylvester equation
    # Solves equation : A X + B X [C,...,C] + D = 0
    # where X is a multilinear function whose dimension is determined by D
    # inverse of A can be optionally specified as an argument

    n_d = D.ndim - 1  # Get number of dimensions
    n_v = C.shape[1]  # Get number of variables

    n_c = D.size // n_v**n_d  # Calculate size of control space

    DD = D.reshape(n_c, n_v**n_d)  # Reshape D matrix

    if n_d == 1:  # Handle 1D case
        CC = C
    else:  # Handle higher dimensional cases
        CC = np.kron(C, C)  # Compute Kronecker product
    for i in range(n_d - 2):  # Additional Kronecker products if needed
        CC = np.kron(CC, C)

    if method == "linear":  # Linear solution method
        I = np.eye(CC.shape[0])  # Identity matrix
        XX = solve_sylvester_vectorized((A, I), (B, CC), DD)  # Solve using vectorized method

    else:  # Use slycot solver
        # we use slycot by default
        import slycot  # Import specialized solver

        if Ainv != None:  # Use provided inverse if available
            Q = sdot(Ainv, B)  # Compute Q matrix
            S = sdot(Ainv, DD)  # Compute S matrix
        else:  # Compute inverse
            Q = np.linalg.solve(A, B)  # Solve for Q
            S = np.linalg.solve(A, DD)  # Solve for S

        n = n_c  # Number of equations
        m = n_v**n_d  # Size of solution

        XX = slycot.sb04qd(n, m, Q, CC, -S)  # Solve using slycot

    X = XX.reshape((n_c,) + (n_v,) * (n_d))  # Reshape solution

    return X  # Return solution matrix


class BKError(Exception):  # Custom exception for Blanchard-Kahn errors
    def __init__(self, type):
        self.type = type

    def __str__(self):
        return "Blanchard-Kahn error ({0})".format(self.type)  # Format error message


# def solve_tensor(A, B):  # Solve tensor equation
#     """
#     Solve a tensor equation of the form A X = B.
    
#     Solves systems of linear equations where the coefficient matrix A and
#     right-hand side B are 3-dimensional tensors. Each slice represents a
#     separate linear system.
    
#     Parameters
#     ----------
#     A : array_like
#         3D tensor of coefficient matrices (n_systems, n_eqs, n_vars)
#     B : array_like
#         3D tensor of right-hand sides (n_systems, n_eqs, n_rhs)
        
#     Returns
#     -------
#     array
#         Solution tensor X with shape (n_systems, n_vars, n_rhs)
        
#     Notes
#     -----
#     Uses numpy.linalg.solve to solve each system independently.
#     Assumes A is invertible for each slice.
#     """
#     from numpy.linalg import solve  # For linear system solution

#     n_systems = A.shape[0]  # Get number of systems
#     n_eqs = A.shape[1]  # Get number of equations
#     n_vars = A.shape[2]  # Get number of variables
#     n_rhs = B.shape[2]  # Get number of right-hand sides

#     X = np.zeros((n_systems, n_vars, n_rhs))  # Initialize solution tensor

#     for system in range(n_systems):  # Iterate over each system
#         for eq in range(n_eqs):  # Iterate over each equation
#             A_slice = A[system, eq, :, :]  # Get current slice of A
#             B_slice = B[system, eq, :]  # Get current slice of B
#             X[system, eq, :] = solve(A_slice, B_slice)  # Solve linear system

#     return X  # Return solution tensor
