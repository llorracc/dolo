from dolo.numeric.tensor import sdot, mdot                # Tensor operations for matrix products (snt3p5)

import numpy as np

TOL = 1e-10                                         # Numerical tolerance for computations (snt3p5)


# credits : second_order_solver is adapted from Sven Schreiber's port of Uhlig's Toolkit.
def second_order_solver(FF, GG, HH, eigmax=1.0 + 1e-6):
    """Solves second-order difference equations using QZ decomposition"""  # Add docstring summary (snt3p5)

    # from scipy.linalg import qz
    from dolo.numeric.extern.qz import qzordered    # QZ decomp with eigenvalue ordering (snt3p5)

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
    )
    from numpy.linalg import solve

    Psi_mat = array(FF)                            # Forward-looking coefficients (snt3p5)
    Gamma_mat = array(-GG)                         # Current state coefficients (snt3p5)
    Theta_mat = array(-HH)                         # Backward-looking coefficients (snt3p5)
    m_states = FF.shape[0]

    Xi_mat = r_[                                   # Build companion matrix for system (snt3p5)
        c_[Gamma_mat, Theta_mat], c_[eye(m_states), zeros((m_states, m_states))]
    ]

    Delta_mat = r_[                                # Build state transition matrix (snt3p5)
        c_[Psi_mat, zeros((m_states, m_states))],
        c_[zeros((m_states, m_states)), eye(m_states)],
    ]

    [Delta_up, Xi_up, UUU, VVV, eigval] = qzordered(  # Compute ordered QZ decomposition (snt3p5)
        Delta_mat,
        Xi_mat,
    )

    VVVH = VVV.T
    VVV_2_1 = VVVH[m_states : 2 * m_states, :m_states]  # Extract submatrices for solution (snt3p5)
    VVV_2_2 = VVVH[m_states : 2 * m_states, m_states : 2 * m_states]
    UUU_2_1 = UUU[m_states : 2 * m_states, :m_states]
    PP = -solve(VVV_2_1, VVV_2_2)                 # Compute policy function matrix (snt3p5)

    # slightly different check than in the original toolkit:
    assert allclose(real_if_close(PP), PP.real)    # Verify solution is real-valued (snt3p5)
    PP = PP.real

    return [eigval, PP]                            # Return eigenvalues and policy matrix (snt3p5)


def solve_sylvester_vectorized(*args):
    """Solves vectorized Sylvester equation using Kronecker products"""  # Add docstring summary (snt3p5)
    from numpy import kron
    from numpy.linalg import solve

    vec = lambda M: M.ravel()                      # Vectorization helper function (snt3p5)
    n = args[0][0].shape[0]
    q = args[0][1].shape[0]
    K = vec(args[-1])                             # Vectorize constant term (snt3p5)
    L = sum([kron(A, B.T) for (A, B) in args[:-1]])  # Build coefficient matrix using Kronecker products (snt3p5)
    X = solve(L, -K)                              # Solve vectorized system (snt3p5)
    return X.reshape((n, q))                       # Reshape solution to matrix form (snt3p5)


def solve_sylvester(A, B, C, D, Ainv=None, method="linear"):
    # Solves equation : A X + B X [C,...,C] + D = 0
    # where X is a multilinear function whose dimension is determined by D
    # inverse of A can be optionally specified as an argument

    n_d = D.ndim - 1                              # Number of C matrices in tensor product (snt3p5)
    n_v = C.shape[1]                              # Size of state vector (snt3p5)

    n_c = D.size // n_v**n_d                      # Size of control vector (snt3p5)

    #    import dolo.config
    #    opts = dolo.config.use_engine
    #    if opts['sylvester']:
    #        DD = D.flatten().reshape( n_c, n_v**n_d)
    #        [err,XX] = dolo.config.engine.engine.feval(2,'gensylv',n_d,A,B,C,-DD)
    #        X = XX.reshape( (n_c,)+(n_v,)*(n_d))

    DD = D.reshape(n_c, n_v**n_d)                 # Reshape D for vectorized operations (snt3p5)

    if n_d == 1:
        CC = C                                     # Single C matrix case (snt3p5)
    else:
        CC = np.kron(C, C)                        # Build repeated Kronecker product of C (snt3p5)
    for i in range(n_d - 2):
        CC = np.kron(CC, C)

    if method == "linear":
        I = np.eye(CC.shape[0])                   # Use vectorized solver for linear case (snt3p5)
        XX = solve_sylvester_vectorized((A, I), (B, CC), DD)

    else:
        # we use slycot by default
        import slycot                             # Use Slycot for more stable solution (snt3p5)

        if Ainv != None:
            Q = sdot(Ainv, B)                     # Use provided A inverse (snt3p5)
            S = sdot(Ainv, DD)
        else:
            Q = np.linalg.solve(A, B)             # Compute A inverse implicitly (snt3p5)
            S = np.linalg.solve(A, DD)

        n = n_c
        m = n_v**n_d

        XX = slycot.sb04qd(n, m, Q, CC, -S)      # Solve using Slycot's Sylvester solver (snt3p5)

    X = XX.reshape((n_c,) + (n_v,) * (n_d))      # Reshape solution to original dimensions (snt3p5)

    return X


class BKError(Exception):
    """Exception raised for Blanchard-Kahn condition violations"""  # Add docstring summary (snt3p5)
    def __init__(self, type):
        self.type = type                          # Type of BK condition violation (snt3p5)

    def __str__(self):
        return "Blanchard-Kahn error ({0})".format(self.type)  # Format error message (snt3p5)
