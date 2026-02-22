"""
1D Euler Equation Solver using Finite Volume Method.

Solves the 1D compressible Euler equations:
    ∂U/∂t + ∂F(U)/∂x = 0

where:
    U = [ρ, ρu, ρE]ᵀ (conservative variables)
    F = [ρu, ρu² + p, u(ρE + p)]ᵀ (flux vector)
"""

import numpy as np
from .flux_schemes import rusanov_flux, hllc_flux


class EulerSolver:
    """
    Finite volume solver for 1D Euler equations.

    Features:
    - HLLC or Rusanov flux schemes
    - RK2 (Heun's method) time integration
    - CFL-based adaptive time stepping
    - Transmissive boundary conditions

    Parameters
    ----------
    nx : int
        Number of grid cells
    x_range : tuple
        Domain extent (x_min, x_max)
    gamma : float
        Ratio of specific heats (default 1.4 for air)
    cfl : float
        CFL number for stability (default 0.5)
    flux_scheme : str
        'hllc' or 'rusanov'
    """

    def __init__(self, nx=400, x_range=(0.0, 1.0), gamma=1.4, cfl=0.5, flux_scheme='hllc'):
        self.nx = nx
        self.x_min, self.x_max = x_range
        self.gamma = gamma
        self.cfl = cfl
        self.flux_scheme = flux_scheme.lower()

        # Grid setup
        self.dx = (self.x_max - self.x_min) / nx
        self.x = np.linspace(self.x_min + 0.5*self.dx,
                             self.x_max - 0.5*self.dx, nx)  # Cell centers

        # Conservative variables: U = [rho, rho*u, rho*E]
        self.U = np.zeros((3, nx))

        # Solution history for animation
        self.history = []
        self.time_history = []

        # Select flux function
        if self.flux_scheme == 'hllc':
            self.flux_function = hllc_flux
        elif self.flux_scheme == 'rusanov':
            self.flux_function = rusanov_flux
        else:
            raise ValueError(f"Unknown flux scheme: {flux_scheme}")

    def primitive_to_conservative(self, rho, u, p):
        """
        Convert primitive variables to conservative variables.

        Parameters
        ----------
        rho : ndarray
            Density
        u : ndarray
            Velocity
        p : ndarray
            Pressure

        Returns
        -------
        U : ndarray of shape (3, n)
            Conservative variables [rho, rho*u, rho*E]
        """
        U = np.zeros((3, len(rho)))
        U[0] = rho
        U[1] = rho * u
        # Total energy: E = p/(gamma-1) + 0.5*rho*u^2
        U[2] = p / (self.gamma - 1) + 0.5 * rho * u**2
        return U

    def conservative_to_primitive(self, U):
        """
        Convert conservative variables to primitive variables.

        Parameters
        ----------
        U : ndarray of shape (3, n)
            Conservative variables [rho, rho*u, rho*E]

        Returns
        -------
        rho, u, p : tuple of ndarrays
            Primitive variables (density, velocity, pressure)
        """
        rho = U[0]
        u = U[1] / rho
        p = (self.gamma - 1) * (U[2] - 0.5 * rho * u**2)
        return rho, u, p

    def set_initial_condition(self, rho_func=None, u_func=None, p_func=None):
        """
        Set initial conditions using provided functions or default Sod problem.

        If no functions provided, uses standard Sod shock tube:
            Left (x < 0.5):  ρ=1.0, u=0.0, p=1.0
            Right (x >= 0.5): ρ=0.125, u=0.0, p=0.1
        """
        if rho_func is None and u_func is None and p_func is None:
            # Default: Standard Sod shock tube problem
            self.set_sod_initial_condition()
        else:
            rho = rho_func(self.x) if rho_func else np.ones(self.nx)
            u = u_func(self.x) if u_func else np.zeros(self.nx)
            p = p_func(self.x) if p_func else np.ones(self.nx)
            self.U = self.primitive_to_conservative(rho, u, p)

    def set_sod_initial_condition(self, x_discontinuity=0.5):
        """
        Set standard Sod shock tube initial conditions.

        Left state (x < x_d):  ρ=1.0, u=0.0, p=1.0
        Right state (x >= x_d): ρ=0.125, u=0.0, p=0.1
        """
        rho = np.where(self.x < x_discontinuity, 1.0, 0.125)
        u = np.zeros(self.nx)
        p = np.where(self.x < x_discontinuity, 1.0, 0.1)

        self.U = self.primitive_to_conservative(rho, u, p)

    def apply_boundary_conditions(self, U):
        """
        Apply transmissive (outflow) boundary conditions.

        Extends the domain with ghost cells that copy the boundary values,
        allowing waves to exit the domain without reflection.
        """
        # Add one ghost cell on each side
        U_ext = np.zeros((3, self.nx + 2))
        U_ext[:, 1:-1] = U

        # Transmissive BCs: copy boundary values to ghost cells
        U_ext[:, 0] = U[:, 0]      # Left ghost cell
        U_ext[:, -1] = U[:, -1]    # Right ghost cell

        return U_ext

    def compute_time_step(self, U):
        """
        Compute time step based on CFL condition.

        dt = CFL * dx / max(|u| + c)
        """
        rho, u, p = self.conservative_to_primitive(U)
        c = np.sqrt(self.gamma * p / rho)  # Sound speed

        max_speed = np.max(np.abs(u) + c)
        dt = self.cfl * self.dx / max_speed

        return dt

    def compute_rhs(self, U):
        """
        Compute the right-hand side: -∂F/∂x using finite volume method.

        Returns
        -------
        dUdt : ndarray of shape (3, nx)
            Time derivative of conservative variables
        """
        # Apply boundary conditions
        U_ext = self.apply_boundary_conditions(U)

        # Get left and right states at each face (nx + 1 faces)
        U_L = U_ext[:, :-1]  # Left states
        U_R = U_ext[:, 1:]   # Right states

        # Compute numerical flux at each face
        F = self.flux_function(U_L, U_R, self.gamma)

        # Finite volume update: dU/dt = -(F_{i+1/2} - F_{i-1/2}) / dx
        dUdt = -(F[:, 1:] - F[:, :-1]) / self.dx

        return dUdt

    def rk2_step(self, U, dt):
        """
        Perform one RK2 (Heun's method) time integration step.

        Stage 1: U* = U^n + dt * L(U^n)
        Stage 2: U^{n+1} = 0.5 * (U^n + U* + dt * L(U*))
        """
        # Stage 1
        k1 = self.compute_rhs(U)
        U_star = U + dt * k1

        # Stage 2
        k2 = self.compute_rhs(U_star)
        U_new = 0.5 * (U + U_star + dt * k2)

        return U_new

    def check_physical_validity(self, U):
        """
        Check that solution remains physically valid (positive density and pressure).
        """
        rho, u, p = self.conservative_to_primitive(U)

        if np.any(rho <= 0):
            raise RuntimeError("Negative density detected!")
        if np.any(p <= 0):
            raise RuntimeError("Negative pressure detected!")

        return True

    def solve(self, t_end, save_interval=None, verbose=True):
        """
        Solve the Euler equations from t=0 to t=t_end.

        Parameters
        ----------
        t_end : float
            Final simulation time
        save_interval : float, optional
            Time interval for saving snapshots (for animation)
        verbose : bool
            Print progress information

        Returns
        -------
        rho, u, p : tuple of ndarrays
            Final solution in primitive variables
        """
        t = 0.0
        n_steps = 0
        next_save_time = 0.0

        # Save initial condition
        self.history = [self.U.copy()]
        self.time_history = [t]

        if save_interval is None:
            save_interval = t_end / 50  # Default: 50 frames

        if verbose:
            print(f"Starting simulation: t_end={t_end}, nx={self.nx}, "
                  f"flux={self.flux_scheme}, CFL={self.cfl}")

        while t < t_end:
            # Compute adaptive time step
            dt = self.compute_time_step(self.U)

            # Don't overshoot t_end
            if t + dt > t_end:
                dt = t_end - t

            # RK2 time integration
            self.U = self.rk2_step(self.U, dt)

            # Check physical validity
            self.check_physical_validity(self.U)

            t += dt
            n_steps += 1

            # Save snapshot for animation
            if t >= next_save_time:
                self.history.append(self.U.copy())
                self.time_history.append(t)
                next_save_time += save_interval

            if verbose and n_steps % 100 == 0:
                print(f"  Step {n_steps}: t = {t:.6f}, dt = {dt:.6e}")

        # Save final state
        if self.time_history[-1] < t:
            self.history.append(self.U.copy())
            self.time_history.append(t)

        if verbose:
            print(f"Simulation complete: {n_steps} steps")

        return self.conservative_to_primitive(self.U)

    def get_solution(self):
        """Return current solution in primitive variables."""
        return self.conservative_to_primitive(self.U)

    def compute_errors(self, exact_rho, exact_u, exact_p):
        """
        Compute L1, L2, and L∞ errors against exact solution.

        Parameters
        ----------
        exact_rho, exact_u, exact_p : ndarrays
            Exact solution values at cell centers

        Returns
        -------
        errors : dict
            Dictionary with L1, L2, Linf errors for each variable
        """
        rho, u, p = self.get_solution()

        errors = {}
        for name, num, exact in [('rho', rho, exact_rho),
                                  ('u', u, exact_u),
                                  ('p', p, exact_p)]:
            diff = np.abs(num - exact)
            errors[f'{name}_L1'] = np.mean(diff)
            errors[f'{name}_L2'] = np.sqrt(np.mean(diff**2))
            errors[f'{name}_Linf'] = np.max(diff)

        return errors
