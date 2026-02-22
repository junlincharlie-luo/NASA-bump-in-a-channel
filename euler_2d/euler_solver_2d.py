"""
2D Euler Equation Solver

Finite volume solver for the 2D compressible Euler equations on
structured body-fitted grids. Uses dimensional splitting with
rotated 1D HLLC flux.

The 2D Euler equations in conservation form:
    ∂U/∂t + ∂F/∂x + ∂G/∂y = 0

where:
    U = [ρ, ρu, ρv, ρE]^T
    F = [ρu, ρu² + p, ρuv, u(ρE + p)]^T
    G = [ρv, ρuv, ρv² + p, v(ρE + p)]^T
"""

import numpy as np
from .metrics import GridMetrics
from .flux_schemes_2d import hllc_flux_2d, rusanov_flux_2d
from .boundary_conditions import BoundaryConditionManager, SubsonicInlet, SubsonicOutlet, SlipWall


class EulerSolver2D:
    """
    2D Finite Volume Euler Solver.

    Uses cell-centered storage with ghost cell boundary treatment.
    Time integration via RK2 (Heun's method).

    Attributes
    ----------
    ni, nj : int
        Number of cells in i and j directions
    gamma : float
        Ratio of specific heats
    cfl : float
        CFL number for time stepping
    U : ndarray
        Conservative variables, shape (4, ni, nj)
    metrics : GridMetrics
        Grid transformation metrics
    """

    def __init__(self, x, y, gamma=1.4, cfl=0.5, flux_scheme='hllc'):
        """
        Initialize the 2D Euler solver.

        Parameters
        ----------
        x : ndarray
            Vertex x-coordinates, shape (ni+1, nj+1)
        y : ndarray
            Vertex y-coordinates, shape (ni+1, nj+1)
        gamma : float
            Ratio of specific heats (default 1.4 for air)
        cfl : float
            CFL number for stability (default 0.5)
        flux_scheme : str
            'hllc' or 'rusanov'
        """
        self.x = x
        self.y = y
        self.ni = x.shape[0] - 1
        self.nj = x.shape[1] - 1
        self.gamma = gamma
        self.cfl = cfl

        # Compute grid metrics
        self.metrics = GridMetrics(x, y)

        # Select flux function
        if flux_scheme.lower() == 'hllc':
            self.flux_function = hllc_flux_2d
        elif flux_scheme.lower() == 'rusanov':
            self.flux_function = rusanov_flux_2d
        else:
            raise ValueError(f"Unknown flux scheme: {flux_scheme}")

        # Initialize state vector
        self.U = np.zeros((4, self.ni, self.nj))

        # Boundary condition manager
        self.bc_manager = BoundaryConditionManager(gamma)

        # History for monitoring
        self.residual_history = []
        self.time_history = []

    def primitive_to_conservative(self, rho, u, v, p):
        """
        Convert primitive variables to conservative variables.

        Parameters
        ----------
        rho : ndarray
            Density
        u : ndarray
            x-velocity
        v : ndarray
            y-velocity
        p : ndarray
            Pressure

        Returns
        -------
        U : ndarray
            Conservative variables (4, ...)
        """
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        return np.array([rho, rho * u, rho * v, E])

    def conservative_to_primitive(self, U):
        """
        Convert conservative variables to primitive variables.

        Parameters
        ----------
        U : ndarray
            Conservative variables (4, ...)

        Returns
        -------
        rho, u, v, p : ndarray
            Primitive variables
        """
        rho = np.maximum(U[0], 1e-10)
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))
        p = np.maximum(p, 1e-10)
        return rho, u, v, p

    def set_initial_condition(self, rho_func=None, u_func=None, v_func=None, p_func=None):
        """
        Set initial condition from functions.

        Parameters
        ----------
        rho_func, u_func, v_func, p_func : callable
            Functions (x, y) -> value for each primitive variable
        """
        xc = self.metrics.xc
        yc = self.metrics.yc

        rho = rho_func(xc, yc) if rho_func else np.ones_like(xc)
        u = u_func(xc, yc) if u_func else np.zeros_like(xc)
        v = v_func(xc, yc) if v_func else np.zeros_like(xc)
        p = p_func(xc, yc) if p_func else np.ones_like(xc)

        self.U = self.primitive_to_conservative(rho, u, v, p)

    def set_uniform_flow(self, M=0.5, alpha=0.0, p=1.0, rho=1.0):
        """
        Set uniform flow initial condition.

        Parameters
        ----------
        M : float
            Mach number
        alpha : float
            Flow angle in radians
        p : float
            Static pressure
        rho : float
            Density
        """
        c = np.sqrt(self.gamma * p / rho)
        V = M * c
        u = V * np.cos(alpha)
        v = V * np.sin(alpha)

        self.U[0, :, :] = rho
        self.U[1, :, :] = rho * u
        self.U[2, :, :] = rho * v
        E = p / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
        self.U[3, :, :] = E

    def set_inlet_bc(self, p0=1.0, T0=1.0, theta=0.0, R=1.0):
        """Set subsonic inlet boundary condition."""
        self.bc_manager.set_inlet(p0, T0, theta, R)

    def set_outlet_bc(self, p_back=0.8):
        """Set subsonic outlet boundary condition."""
        self.bc_manager.set_outlet(p_back)

    def set_wall_bc(self, wall='bottom'):
        """Set slip wall boundary condition."""
        self.bc_manager.set_wall(wall)

    def compute_time_step(self):
        """
        Compute time step based on CFL condition.

        Returns
        -------
        dt : float
            Time step satisfying CFL condition
        """
        rho, u, v, p = self.conservative_to_primitive(self.U)
        c = np.sqrt(self.gamma * p / rho)

        # Get local spacing
        dx, dy = self.metrics.get_local_spacing()

        # Spectral radii
        lambda_x = (np.abs(u) + c) / dx
        lambda_y = (np.abs(v) + c) / dy

        # CFL condition
        dt = self.cfl / np.max(lambda_x + lambda_y)

        return dt

    def compute_rhs(self, U):
        """
        Compute right-hand side of the semi-discrete equations.

        dU/dt = -1/V * (sum of face fluxes)

        Parameters
        ----------
        U : ndarray
            Conservative variables (4, ni, nj)

        Returns
        -------
        rhs : ndarray
            Right-hand side, same shape as U
        """
        ni, nj = self.ni, self.nj
        rhs = np.zeros_like(U)

        # Get extended state with ghost cells
        U_ext = self.bc_manager.get_extended_state(U, self.metrics)

        # --- i-direction fluxes ---
        # Flux at face (i+1/2, j): between cell (i,j) and (i+1,j)
        # We need fluxes at faces 0, 1, ..., ni (ni+1 faces)
        # U_ext indices: ghost_left=0, cells=1..ni, ghost_right=ni+1
        # Face i is between U_ext[:,i,:] and U_ext[:,i+1,:]

        for i in range(ni + 1):
            # Left and right states
            U_L = U_ext[:, i, 1:-1]    # Shape: (4, nj)
            U_R = U_ext[:, i+1, 1:-1]

            # Face normal
            nx, ny = self.metrics.get_i_face_normal(i, slice(None))
            S_mag = self.metrics.Si_mag[i, :]

            # Compute flux
            F = self.flux_function(U_L, U_R, nx, ny, self.gamma)

            # Accumulate: flux leaves cell i-1, enters cell i
            # rhs[cell] -= outward_flux * face_area / volume
            if i > 0:
                rhs[:, i-1, :] -= F * S_mag / self.metrics.volume[i-1, :]
            if i < ni:
                rhs[:, i, :] += F * S_mag / self.metrics.volume[i, :]

        # --- j-direction fluxes ---
        # Flux at face (i, j+1/2): between cell (i,j) and (i,j+1)

        for j in range(nj + 1):
            # Left (below) and right (above) states
            U_L = U_ext[:, 1:-1, j]
            U_R = U_ext[:, 1:-1, j+1]

            # Face normal
            nx, ny = self.metrics.get_j_face_normal(slice(None), j)
            S_mag = self.metrics.Sj_mag[:, j]

            # Compute flux
            G = self.flux_function(U_L, U_R, nx, ny, self.gamma)

            # Accumulate
            if j > 0:
                rhs[:, :, j-1] -= G * S_mag / self.metrics.volume[:, j-1]
            if j < nj:
                rhs[:, :, j] += G * S_mag / self.metrics.volume[:, j]

        return rhs

    def rk2_step(self, U, dt):
        """
        Perform one RK2 (Heun's method) time step.

        Stage 1: U* = U^n + dt * L(U^n)
        Stage 2: U^{n+1} = 0.5 * (U^n + U* + dt * L(U*))

        Parameters
        ----------
        U : ndarray
            Current state
        dt : float
            Time step

        Returns
        -------
        U_new : ndarray
            Updated state
        """
        # Stage 1
        rhs1 = self.compute_rhs(U)
        U_star = U + dt * rhs1

        # Ensure positivity
        U_star = self._enforce_positivity(U_star)

        # Stage 2
        rhs2 = self.compute_rhs(U_star)
        U_new = 0.5 * (U + U_star + dt * rhs2)

        # Ensure positivity
        U_new = self._enforce_positivity(U_new)

        return U_new

    def _enforce_positivity(self, U):
        """Enforce physical realizability (positive density and pressure)."""
        # Minimum density
        U[0] = np.maximum(U[0], 1e-10)

        # Check pressure
        rho = U[0]
        u = U[1] / rho
        v = U[2] / rho
        E = U[3]
        p = (self.gamma - 1) * (E - 0.5 * rho * (u**2 + v**2))

        # If pressure is negative, limit internal energy
        neg_p = p < 1e-10
        if np.any(neg_p):
            p_min = 1e-10
            E_min = p_min / (self.gamma - 1) + 0.5 * rho * (u**2 + v**2)
            U[3] = np.where(neg_p, E_min, E)

        return U

    def compute_residual(self, rhs):
        """
        Compute residual norm for convergence monitoring.

        Parameters
        ----------
        rhs : ndarray
            Right-hand side of equations

        Returns
        -------
        residual : float
            L2 norm of density residual
        """
        return np.sqrt(np.mean(rhs[0]**2))

    def solve(self, t_end, save_interval=0.1, verbose=True):
        """
        Solve to a specified time.

        Parameters
        ----------
        t_end : float
            Final time
        save_interval : float
            Interval for saving snapshots
        verbose : bool
            Print progress

        Returns
        -------
        rho, u, v, p : ndarray
            Final primitive variables
        """
        t = 0.0
        step = 0
        next_save = save_interval

        self.time_history = [0.0]
        self.solution_history = [self.U.copy()]

        while t < t_end:
            dt = self.compute_time_step()

            # Don't overshoot
            if t + dt > t_end:
                dt = t_end - t

            self.U = self.rk2_step(self.U, dt)
            t += dt
            step += 1

            # Save snapshot
            if t >= next_save:
                self.time_history.append(t)
                self.solution_history.append(self.U.copy())
                next_save += save_interval

            if verbose and step % 100 == 0:
                rhs = self.compute_rhs(self.U)
                res = self.compute_residual(rhs)
                print(f"Step {step}, t = {t:.4f}, residual = {res:.2e}")

        return self.conservative_to_primitive(self.U)

    def solve_steady(self, max_iter=10000, tol=1e-6, verbose=True, print_interval=100):
        """
        Solve to steady state using local time stepping.

        Parameters
        ----------
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance on density residual
        verbose : bool
            Print progress
        print_interval : int
            Iterations between progress prints

        Returns
        -------
        converged : bool
            Whether solution converged
        rho, u, v, p : ndarray
            Final primitive variables
        """
        self.residual_history = []

        for iteration in range(max_iter):
            # Compute RHS and residual
            rhs = self.compute_rhs(self.U)
            residual = self.compute_residual(rhs)
            self.residual_history.append(residual)

            # Check convergence
            if residual < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}, residual = {residual:.2e}")
                return True, *self.conservative_to_primitive(self.U)

            # Local time stepping
            dt = self.compute_time_step()

            # RK2 update
            self.U = self.rk2_step(self.U, dt)

            # Progress
            if verbose and iteration % print_interval == 0:
                print(f"Iteration {iteration}, residual = {residual:.2e}")

        if verbose:
            print(f"Did not converge after {max_iter} iterations, residual = {residual:.2e}")

        return False, *self.conservative_to_primitive(self.U)

    def get_mach_number(self):
        """Compute Mach number field."""
        rho, u, v, p = self.conservative_to_primitive(self.U)
        c = np.sqrt(self.gamma * p / rho)
        V = np.sqrt(u**2 + v**2)
        return V / c

    def get_pressure_coefficient(self, p_inf=1.0, rho_inf=1.0, V_inf=None, M_inf=None):
        """
        Compute pressure coefficient.

        Cp = (p - p_inf) / (0.5 * rho_inf * V_inf^2)

        Parameters
        ----------
        p_inf : float
            Freestream pressure
        rho_inf : float
            Freestream density
        V_inf : float, optional
            Freestream velocity (computed from M_inf if not given)
        M_inf : float, optional
            Freestream Mach number

        Returns
        -------
        Cp : ndarray
            Pressure coefficient field
        """
        _, _, _, p = self.conservative_to_primitive(self.U)

        if V_inf is None:
            if M_inf is None:
                raise ValueError("Must specify either V_inf or M_inf")
            c_inf = np.sqrt(self.gamma * p_inf / rho_inf)
            V_inf = M_inf * c_inf

        q_inf = 0.5 * rho_inf * V_inf**2
        Cp = (p - p_inf) / q_inf

        return Cp

    def compute_mass_flow_rate(self, boundary='left'):
        """
        Compute mass flow rate through a boundary.

        Parameters
        ----------
        boundary : str
            'left' or 'right'

        Returns
        -------
        mdot : float
            Mass flow rate (positive = flow in +x direction)
        """
        rho, u, v, p = self.conservative_to_primitive(self.U)

        if boundary == 'left':
            # Sum over j-faces at i=0
            rho_face = rho[0, :]
            u_face = u[0, :]
            v_face = v[0, :]
            S_mag = self.metrics.Si_mag[0, :]
            nx, ny = self.metrics.get_i_face_normal(0, slice(None))
        elif boundary == 'right':
            rho_face = rho[-1, :]
            u_face = u[-1, :]
            v_face = v[-1, :]
            S_mag = self.metrics.Si_mag[-1, :]
            nx, ny = self.metrics.get_i_face_normal(-1, slice(None))
        else:
            raise ValueError(f"Unknown boundary: {boundary}")

        # Normal velocity
        V_n = u_face * nx + v_face * ny

        # Mass flow rate
        mdot = np.sum(rho_face * V_n * S_mag)

        return mdot

    def check_physical_validity(self):
        """Check that solution is physically valid."""
        rho, u, v, p = self.conservative_to_primitive(self.U)

        valid = True
        if np.any(rho <= 0):
            print(f"Warning: Negative density detected, min = {np.min(rho):.2e}")
            valid = False
        if np.any(p <= 0):
            print(f"Warning: Negative pressure detected, min = {np.min(p):.2e}")
            valid = False

        return valid
