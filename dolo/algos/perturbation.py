from dolo.compiler.model import Model
from dolo.numeric.processes import VAR1, IIDProcess  # Shock process specifications
from dolo.numeric.distribution import Normal  # Distribution for shocks
from dolo.numeric.grids import PointGrid, EmptyGrid  # Grid types for state space
from dolo.numeric.decision_rule import CallableDecisionRule  # Base class for policy functions
from numpy import column_stack, dot, eye, row_stack, zeros  # Linear algebra tools
from numpy.linalg import solve  # Linear system solver
from dolo.numeric.extern.qz import qzordered  # QZ decomposition for generalized eigenvalue problem
from .results import AlgoResult, PerturbationResult  # Result containers


def get_derivatives(model: Model, steady_state=None):
    """
    Compute derivatives of model equations needed for perturbation.
    
    This implements the first step of perturbation analysis as described
    in perturbation.md. It computes derivatives of:
    1. Transition equations (g): How states evolve
    2. Arbitrage equations (f): Optimality conditions
    
    The derivatives are taken with respect to:
    - Current states (s)
    - Current controls (x)
    - Current shocks (m)
    - Future states (S)
    - Future controls (X)
    - Future shocks (M)
    
    Parameters
    ----------
    model : Model
        Model with properly defined transition and arbitrage equations
    steady_state : dict, optional
        Steady state values to compute derivatives around
        
    Returns
    -------
    tuple
        Derivatives needed for perturbation:
        - G_s: State transition wrt current states
        - G_x: State transition wrt controls
        - G_e: State transition wrt shocks
        - F_s: Arbitrage wrt current states
        - F_x: Arbitrage wrt current controls
        - F_S: Arbitrage wrt future states
        - F_X: Arbitrage wrt future controls
    """
    if steady_state is None:
        steady_state = model.calibration

    m = steady_state["exogenous"]
    s = steady_state["states"]
    x = steady_state["controls"]
    p = steady_state["parameters"]

    f = model.functions["arbitrage"]
    g = model.functions["transition"]

    n_m = len(m)
    n_s = len(s)
    n_x = len(x)

    f_, f_m, f_s, f_x, f_M, f_S, f_X = f(m, s, x, m, s, x, p, diff=True)
    g_, g_m, g_s, g_x, g_M = g(m, s, x, m, p, diff=True)

    process = model.exogenous

    if isinstance(process, VAR1):
        A = process.rho
        B = np.eye(A.shape[0])
        G_s = np.row_stack([
            np.column_stack([A, np.zeros((n_m, n_s))]),  # Shock persistence
            np.column_stack([g_m + g_M @ A, g_s])        # State transition
        ])
        G_x = np.row_stack([np.zeros((n_m, n_x)), g_x])
        G_e = g_m @ B
        F_s = np.column_stack([f_m, f_s])
        F_S = np.column_stack([f_M, f_S])
        F_x = f_x
        F_X = f_X
    elif isinstance(process, IIDProcess):
        G_s = g_s
        G_x = g_x
        G_e = g_m
        F_s = f_s
        F_S = f_S
        F_x = f_x
        F_X = f_X
    else:
        raise Exception(f"Not implemented: perturbation for shock {process.__class__}")

    return G_s, G_x, G_e, F_s, F_x, F_S, F_X


from dolo.numeric.grids import PointGrid, EmptyGrid
from dolo.numeric.decision_rule import CallableDecisionRule


class BivariateTaylor(CallableDecisionRule):
    """
    First-order Taylor approximation of policy function.
    
    This represents the linear approximation:
    x(m,s) ≈ x_bar + C_m(m - m_bar) + C_s(s - s_bar)
    where:
    - (m_bar, s_bar, x_bar) is the steady state
    - C_m is the response to shock deviations
    - C_s is the response to state deviations
    
    Parameters
    ----------
    m_bar : ndarray
        Steady state exogenous values
    s_bar : ndarray
        Steady state states
    x_bar : ndarray
        Steady state controls
    C_m : ndarray
        Policy derivatives wrt exogenous
    C_s : ndarray
        Policy derivatives wrt states
    """
    def __init__(self, m_bar, s_bar, x_bar, C_m, C_s):

        self.endo_grid = PointGrid(s_bar)
        if C_m is None:
            self.exo_grid = EmptyGrid()
        else:
            self.exo_grid = PointGrid(m_bar)

        self.m_bar = m_bar
        self.s_bar = s_bar
        self.x_bar = x_bar
        self.C_m = C_m
        self.C_s = C_s

    def eval_s(self, s):
        """Evaluate policy at state s, integrating out exogenous"""
        s = np.array(s)
        if s.ndim == 1:
            return self.eval_s(s[None, :])[0, :]
        return self.eval_ms(None, s)

    def eval_ms(self, m, s):
        """
        Evaluate policy at exogenous state m and endogenous state s.
        
        Implements the linear approximation:
        x(m,s) ≈ x_bar + C_m(m - m_bar) + C_s(s - s_bar)
        """
        m = np.array(m)
        s = np.array(s)
        if m.ndim == 1 and s.ndim == 1:
            if self.C_m is not None:
                return self.eval_ms(m[None, :], s[None, :])[0, :]
            else:
                return self.eval_ms(None, s[None, :])[0, :]
        elif m.ndim == 1:
            m = m[None, :]
        elif s.ndim == 1:
            s = s[None, :]

        C_m = self.C_m
        C_s = self.C_s
        if C_m is not None:
            dm = m - self.m_bar[None, :]
            ds = s - self.s_bar[None, :]
            return self.x_bar[None, :] + dm @ C_m.T + ds @ C_s.T
        else:
            ds = s - self.s_bar[None, :]
            return self.x_bar[None, :] + ds @ C_s.T


import numpy as np
from numpy import column_stack, dot, eye, row_stack, zeros
from numpy.linalg import solve

from dolo.numeric.extern.qz import qzordered


def approximate_1st_order(g_s, g_x, g_e, f_s, f_x, f_S, f_X):
    """
    Compute first-order approximation of policy function.
    
    This implements the linear approximation method described in
    perturbation.md. It works by:
    1. Setting up the linear system from model derivatives
    2. Using QZ decomposition to find the stable manifold
    3. Computing policy derivatives from the stable manifold
    
    Parameters
    ----------
    g_s : ndarray
        State transition derivatives wrt states
    g_x : ndarray
        State transition derivatives wrt controls
    g_e : ndarray
        State transition derivatives wrt shocks
    f_s : ndarray
        Arbitrage derivatives wrt current states
    f_x : ndarray
        Arbitrage derivatives wrt current controls
    f_S : ndarray
        Arbitrage derivatives wrt future states
    f_X : ndarray
        Arbitrage derivatives wrt future controls
        
    Returns
    -------
    tuple
        - C: Policy derivatives
        - eigval_s: Sorted eigenvalues for stability analysis
    """
    n_s = g_s.shape[0]  # number of controls
    n_x = g_x.shape[1]  # number of states
    n_e = g_e.shape[1]
    n_v = n_s + n_x

    A = row_stack([
        column_stack([eye(n_s), zeros((n_s, n_x))]),  # State transition
        column_stack([-f_S, -f_X])                    # Arbitrage conditions
    ])
    B = row_stack([
        column_stack([g_s, g_x]),  # State dynamics
        column_stack([f_s, f_x])   # Control response
    ])

    [S, T, Q, Z, eigval] = qzordered(A, B, 1.0 - 1e-8)

    Z = Z.real

    diag_S = np.diag(S)
    diag_T = np.diag(T)

    tol_geneigvals = 1e-10

    try:
        ok = sum((abs(diag_S) < tol_geneigvals) * (abs(diag_T) < tol_geneigvals)) == 0
        assert ok
    except Exception as e:
        raise GeneralizedEigenvaluesError(diag_S=diag_S, diag_T=diag_T)

    eigval_s = sorted(eigval, reverse=False)
    if max(eigval[:n_s]) >= 1 and min(eigval[n_s:]) < 1:
        # BK conditions are met
        pass
    else:
        ev_a = eigval_s[n_s - 1]
        ev_b = eigval_s[n_s]
        cutoff = (ev_a - ev_b) / 2
        if not ev_a > ev_b:
            raise GeneralizedEigenvaluesSelectionError(
                A=A,
                B=B,
                eigval=eigval,
                cutoff=cutoff,
                diag_S=diag_S,
                diag_T=diag_T,
                n_states=n_s,
            )
        import warnings

        if cutoff > 1:
            warnings.warn("Solution is not convergent.")
        else:
            warnings.warn(
                "There are multiple convergent solutions. The one with the smaller eigenvalues was selected."
            )
        [S, T, Q, Z, eigval] = qzordered(A, B, cutoff)

    Z11 = Z[:n_s, :n_s]
    # Z12 = Z[:n_s, n_s:]
    Z21 = Z[n_s:, :n_s]
    # Z22 = Z[n_s:, n_s:]
    # S11 = S[:n_s, :n_s]
    # T11 = T[:n_s, :n_s]

    # first order solution
    # P = (solve(S11.T, Z11.T).T @ solve(Z11.T, T11.T).T)
    C = solve(Z11.T, Z21.T).T

    A = g_s + g_x @ C
    B = g_e

    return C, eigval_s


from .results import AlgoResult, PerturbationResult
from dolo.compiler.model import Model


def perturb(
    model: Model,
    *,
    details: bool = True,  #
    verbose: bool = True,  #
    steady_state=None,
    eigmax=1.0 - 1e-6,
    solve_steady_state=False,
    order=1,
) -> PerturbationResult:
    """
    Compute perturbation approximation of optimal policy.
    
    This implements the perturbation method described in perturbation.md.
    The method works by:
    1. Computing steady state
    2. Taking derivatives of model equations
    3. Solving linear system for policy derivatives
    
    Parameters
    ----------

    model: NumericModel
        Model to be solved

    verbose : bool, default=True
        Print computation details

    steady_state : dict, optional
        Use provided steady state
    eigmax : float, default=1.0-1e-6
        Maximum eigenvalue for stability

    solve_steady_state: boolean
        Use nonlinear solver to find the steady-state

    orders: {1}
        Approximation order. (Currently, only first order is supported).

    Returns:
    --------

    PerturbationResult
        Object containing:
        - Decision rule (linear approximation)
        - Eigenvalues for stability analysis
        - Whether solution is stable/unique
    """

    if order > 1:
        raise Exception("Only first order implemented.")

    if steady_state is None:
        steady_state = model.calibration

    G_s, G_x, G_e, F_s, F_x, F_S, F_X = get_derivatives(
        model, steady_state=steady_state
    )

    C, eigvals = approximate_1st_order(G_s, G_x, G_e, F_s, F_x, F_S, F_X)

    m = steady_state["exogenous"]
    s = steady_state["states"]
    x = steady_state["controls"]

    from dolo.numeric.processes import VAR1, IIDProcess
    from dolo.numeric.distribution import MvNormal

    n_s = len(s)
    stable = max(eigvals[:n_s]) >= 1 and min(eigvals[n_s:]) < 1
    unique = True  # For first order
    determined = True  # For first order

    # Return results
    if not details:
        return dr
    else:
        return PerturbationResult(
            dr,
            eigvals,
            True,  # otherwise an Exception should have been raised already
            True,  # otherwise an Exception should have been raised already
            True,  # otherwise an Exception should have been raised already
        )
