import numpy

from dolo.numeric.discretization.splines import UniformSplineBasis, TensorBase, B_prod

def construct_j2(jres):
    # Convert Jacobian residuals to sparse matrix format
    import numpy

    m_m, N, n_x, a, b = jres.shape                  # Get dimensions of Jacobian tensor
    assert a == n_m
    assert b == n_x
    big_N = n_m * N * n_x * n_m * N * n_x          # Total size of sparse matrix
    nnz = m_m * N * n_x * n_m * n_x                # Number of non-zero elements
    inds = numpy.zeros((nnz, 6), dtype=float)       # Indices array for non-zero elements
    vals = numpy.zeros(nnz, dtype=float)            # Values array for non-zero elements
    n = 0
    for i in range(n_m):
        for j in range(N):
            for k in range(n_x):
                for l in range(n_m):
                    for m in range(n_x):
                        inds[n, 0] = i               # Store row indices
                        inds[n, 1] = j
                        inds[n, 2] = k
                        inds[n, 3] = l               # Store column indices
                        inds[n, 4] = j
                        inds[n, 5] = m
                        vals[n] = jres[i, j, k, l, m]  # Store matrix values
                        n += 1
    return SparseTensor(inds, vals, (n_m, N, n_x, n_m, N, n_x))  # Return sparse tensor

class SmartJacobian:
    # Efficient Jacobian computation for dynamic programming problems
    def __init__(self, res, dres, jres, fut_S, grid):
        # Initialize with residuals, derivatives and grid
        self.res = res                              # Residuals array
        self.dres = dres                            # Derivatives wrt current states
        self.jres = jres                            # Derivatives wrt future states
        self.fut_S = fut_S                          # Future state values
        self.n_m = res.shape[0]                     # Number of exogenous states
        self.N = res.shape[1]                       # Number of grid points
        self.n_x = res.shape[2]                     # Number of controls
        self.grid = grid                            # Grid object
        self.__B__ = None                           # Cache for filter matrix

    def B(self):
        # Get filter matrix for sparse operations
        if self.__B__ is None:
            self.__B__ = self.get_filter()          # Compute filter if not cached
        return self.__B__

    def get_filter(self):
        # Construct filter matrix for sparse operations
        a = self.grid.a                             # Grid lower bounds
        b = self.grid.b                             # Grid upper bounds
        o = self.grid.orders                        # Grid points per dimension

        bases = [UniformSplineBasis(a[i], b[i], o[i]) for i in range(len(o))]  # Create spline bases
        tb = TensorBase(bases)                      # Create tensor product base
        self.tb = tb

        B = B_prod(bases)                           # Compute basis product matrix

        return B

    @property
    def jac_1(self):
        # Get first Jacobian matrix component
        n_ms, N, n_x = self.res.shape
        dres = self.dres
        bigN = n_ms * N * n_x
        DDRes = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for n in range(N):
            for i_m in range(n_ms):
                DDRes[i_m, n, :, i_m, n, :] = dres[i_m, n, :, :]  # Fill diagonal blocks
        jac1 = DDRes.reshape((bigN, bigN))         # Reshape to 2D matrix
        return jac1

    @property
    def j2_A(self):
        # Get second Jacobian matrix component A
        n_ms, N, n_x = self.res.shape
        jres = self.jres
        fut_S = self.fut_S
        bigN = n_ms * N * n_x

        JJRes = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for n in range(N):
            # for i_m in range(n_ms):
            JJRes[:, n, :, :, n, :] = jres[:, n, :, :, :]  # Fill blocks with Jacobian values

        return JJRes.reshape((bigN, bigN))         # Reshape to 2D matrix

    @property
    def j2_B(self):
        # Get second Jacobian matrix component B
        n_ms, N, n_x = self.res.shape
        fut_S = self.fut_S
        bigN = n_ms * N * n_x
        fut_S = self.fut_S
        DDX = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))
        for i_x in range(n_x):
            for i_m in range(n_ms):
                for i_M in range(n_ms):
                    S = fut_S[i_m, :, i_M, :]
                    Phi = tb.Phi(S).as_matrix()     # Compute basis functions
                    # Phi(S) c = Phi(S) (B^{-1}) x
                    # i_m -> i_M
                    DDX[i_m, :, i_x, i_m, :, i_x] = (Phi @ numpy.linalg.inv(B))[:, 1:-1]  # Fill derivative blocks
    #                     DDX[i_M,:,i_x,i_m,:,i_x] = (Phi @ numpy.linalg.inv(B))[:,1:-1]

        return DDX.reshape((bigN, bigN))           # Reshape to 2D matrix

    @property
    def jac_2(self):
        # Get full second Jacobian matrix
        B = self.get_filter()                      # Get filter matrix
        d = len(self.tb.bases)                     # Number of dimensions
        mdims = tuple([b.m for b in self.tb.bases])  # Points per dimension

        # works
        n_ms, N, n_x = self.res.shape
        #         jres = self.jres
        #         fut_S = self.fut_S
        bigN = n_ms * N * n_x
        jres = self.jres

        fut_S = self.fut_S
        try:
            B = B.todense()                        # Convert to dense if sparse
        except:
            pass
        Binv = numpy.linalg.inv(B)                # Compute inverse of filter
        mat = numpy.zeros((n_ms, N, n_x, n_ms, N, n_x))

        for i in range(n_ms):
            for j in range(n_ms):
                S = fut_S[i, :, j, :]
                Phi = self.tb.Phi(S).as_matrix()   # Compute basis functions
                X = Phi @ Binv  # [:,1:-1]
                X = numpy.array(X)
                X = X.reshape((X.shape[0],) + mdims)  # Reshape to match dimensions
                if d == 1:
                    X = X[:, 1:-1]                 # Handle boundary points
                elif d == 2:
                    X = X[:, 1:-1, 1:-1]
                elif d == 3:
                    X = X[:, 1:-1, 1:-1, 1:-1]
                X = X.reshape((X.shape[0], -1))    # Flatten inner dimensions
                XX = numpy.zeros((N, n_x, N, n_x))
                for y in range(n_x):
                    XX[:, y, :, y] = X             # Fill control blocks
                XX = XX.reshape((N * n_x, N * n_x))
                ff = jres[i, :, :, j, :]
                ff_m = numpy.zeros((N, n_x, N, n_x))
                for n in range(N):
                    ff_m[n, :, n, :] = ff[n, :, :]  # Fill Jacobian blocks
                m = ff_m.reshape((N * n_x, N * n_x)) @ XX  # Matrix multiply
                m = m.reshape((N, n_x, N, n_x))
                mat[i, :, :, j, :, :] = m          # Store result
        mat = mat.reshape((bigN, bigN))
        mat[abs(mat) < 1e-6] = 0                  # Zero small elements
        import scipy.sparse

        mat = scipy.sparse.coo_matrix(mat)         # Convert to sparse format
        return mat

    @property
    def jac(self):
        # Get full Jacobian matrix
        return self.jac_1 + self.jac_2             # Combine first and second components

    def solve(self, rr):
        # Solve linear system using full Jacobian
        import numpy.linalg

        return numpy.linalg.solve(self.jac, rr.ravel()).reshape(rr.shape)  # Direct solver

    def solve_sp(self, rr):
        # Solve linear system using sparse Jacobian
        import scipy.sparse
        from scipy.sparse.linalg import spsolve

        jj = scipy.sparse.csr_matrix(self.jac)     # Convert to CSR format
        res = spsolve(jj, rr.ravel())             # Sparse solver
        return res.reshape(rr.shape)

    def solve_smart(
        self, rr, tol=1e-10, maxit=1000, verbose=False, filt=None, scale=1.0
    ):
        # Solve linear system using iterative method
        n_m, N, n_x = self.res.shape
        fut_S = self.fut_S
        bigN = n_m * N * n_x
        dres = self.dres
        jres = self.jres
        grid = self.grid
        fut_S = self.fut_S
        sol, nn = invert_jac(
            rr * scale,                            # Scale residuals
            dres,                                  # Current state derivatives
            jres,                                  # Future state derivatives
            fut_S,                                 # Future states
            n_m,                                   # Number of exogenous states
            N,                                     # Number of grid points
            n_x,                                   # Number of controls
            grid,                                  # Grid object
            tol=tol,                              # Convergence tolerance
            maxit=maxit,                          # Maximum iterations
            verbose=verbose,                       # Print progress
            filt=filt,                            # Optional filter
        )
        sol /= scale                              # Unscale solution
        return sol

    def solve_ind(self, rr, tol=1e-12):
        # works only if there are no shocks
        jac1 = self.jac_1                         # Get first Jacobian component
        j2_A = self.j2_A                          # Get second component A
        j2_B = self.j2_B                          # Get second component B
        M = jac1                                  # System matrix
        N = -numpy.linalg.solve(jac1, j2_A @ j2_B)  # Iteration matrix
        # I need to prove sp(N)<[0,1[

        abs(numpy.linalg.eig(N)[0]).max()         # Check max eigenvalue
        abs(numpy.linalg.eig(N)[0]).min()         # Check min eigenvalue

        term = numpy.linalg.solve(M, rr.ravel())  # Initial solution
        tot = term
        for i in range(10000):
            term = N @ term                       # Apply iteration matrix
            err = abs(term).max()                 # Check convergence
            tot = tot + term                      # Update solution
            if err < tol:                         # Check tolerance
                break
        return tot.reshape(rr.shape)              # Return reshaped solution 