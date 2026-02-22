"""
Boundary Conditions for 2D Euler Equations

Implements various boundary condition types:
- Slip wall (inviscid wall)
- Subsonic inlet (total conditions specified)
- Subsonic outlet (back pressure specified)
- Supersonic inlet/outlet (extrapolation)
- Characteristic boundary conditions
"""

import numpy as np


class BoundaryCondition:
    """Base class for boundary conditions."""

    def apply(self, U, metrics, boundary, gamma):
        """
        Apply boundary condition to ghost cells.

        Parameters
        ----------
        U : ndarray
            Conservative variables (4, ni, nj)
        metrics : GridMetrics
            Grid metric information
        boundary : str
            Which boundary: 'left', 'right', 'bottom', 'top'
        gamma : float
            Ratio of specific heats

        Returns
        -------
        U_ghost : ndarray
            Ghost cell values to be used for flux computation
        """
        raise NotImplementedError


class SlipWall(BoundaryCondition):
    """
    Inviscid slip wall boundary condition.

    Normal velocity is reflected (zero normal flux),
    tangential velocity is preserved.
    """

    def apply(self, U, metrics, boundary, gamma):
        """Apply slip wall BC by reflecting normal velocity."""
        ni, nj = U.shape[1], U.shape[2]
        U_ghost = np.zeros((4,) + self._get_ghost_shape(boundary, ni, nj))

        if boundary == 'bottom':
            # Get interior cells adjacent to boundary
            U_int = U[:, :, 0]

            # Get wall normals (j=0 faces point in -j direction for ghost)
            nx, ny = metrics.get_j_face_normal(slice(None), 0)

            # Copy density and energy
            U_ghost[0] = U_int[0]
            U_ghost[3] = U_int[3]

            # Reflect velocity: v_ghost = v_int - 2*(v_int . n)*n
            rho = U_int[0]
            u = U_int[1] / rho
            v = U_int[2] / rho

            v_dot_n = u * nx + v * ny
            u_new = u - 2 * v_dot_n * nx
            v_new = v - 2 * v_dot_n * ny

            U_ghost[1] = rho * u_new
            U_ghost[2] = rho * v_new

        elif boundary == 'top':
            U_int = U[:, :, -1]
            nx, ny = metrics.get_j_face_normal(slice(None), -1)

            U_ghost[0] = U_int[0]
            U_ghost[3] = U_int[3]

            rho = U_int[0]
            u = U_int[1] / rho
            v = U_int[2] / rho

            v_dot_n = u * nx + v * ny
            u_new = u - 2 * v_dot_n * nx
            v_new = v - 2 * v_dot_n * ny

            U_ghost[1] = rho * u_new
            U_ghost[2] = rho * v_new

        elif boundary == 'left':
            U_int = U[:, 0, :]
            nx, ny = metrics.get_i_face_normal(0, slice(None))

            U_ghost[0] = U_int[0]
            U_ghost[3] = U_int[3]

            rho = U_int[0]
            u = U_int[1] / rho
            v = U_int[2] / rho

            v_dot_n = u * nx + v * ny
            u_new = u - 2 * v_dot_n * nx
            v_new = v - 2 * v_dot_n * ny

            U_ghost[1] = rho * u_new
            U_ghost[2] = rho * v_new

        elif boundary == 'right':
            U_int = U[:, -1, :]
            nx, ny = metrics.get_i_face_normal(-1, slice(None))

            U_ghost[0] = U_int[0]
            U_ghost[3] = U_int[3]

            rho = U_int[0]
            u = U_int[1] / rho
            v = U_int[2] / rho

            v_dot_n = u * nx + v * ny
            u_new = u - 2 * v_dot_n * nx
            v_new = v - 2 * v_dot_n * ny

            U_ghost[1] = rho * u_new
            U_ghost[2] = rho * v_new

        return U_ghost

    def _get_ghost_shape(self, boundary, ni, nj):
        if boundary in ['bottom', 'top']:
            return (ni,)
        else:
            return (nj,)


class SubsonicInlet(BoundaryCondition):
    """
    Subsonic inlet boundary condition.

    Specifies total pressure, total temperature, and flow angle.
    Extrapolates one characteristic from interior.
    """

    def __init__(self, p0=1.0, T0=1.0, theta=0.0, R=1.0):
        """
        Initialize subsonic inlet.

        Parameters
        ----------
        p0 : float
            Total (stagnation) pressure
        T0 : float
            Total (stagnation) temperature
        theta : float
            Flow angle in radians (0 = aligned with x-axis)
        R : float
            Gas constant (for p = rho * R * T)
        """
        self.p0 = p0
        self.T0 = T0
        self.theta = theta
        self.R = R

    def apply(self, U, metrics, boundary, gamma):
        """Apply subsonic inlet BC."""
        if boundary != 'left':
            raise ValueError("SubsonicInlet only implemented for left boundary")

        # Interior values at inlet plane
        U_int = U[:, 0, :]
        rho_int = np.maximum(U_int[0], 1e-10)
        u_int = U_int[1] / rho_int
        v_int = U_int[2] / rho_int
        E_int = U_int[3]
        p_int = (gamma - 1) * (E_int - 0.5 * rho_int * (u_int**2 + v_int**2))
        p_int = np.maximum(p_int, 1e-10)

        # Speed of sound at interior
        c_int = np.sqrt(gamma * p_int / rho_int)

        # Extrapolate one Riemann invariant from interior
        # J- = u - 2c/(gamma-1) = const along C- characteristic
        J_minus = u_int - 2 * c_int / (gamma - 1)

        # From isentropic relations with total conditions:
        # p/p0 = (T/T0)^(gamma/(gamma-1))
        # T/T0 = 1 - (gamma-1)/2 * M^2
        # Need to solve for Mach number

        # Use Newton iteration or approximate
        # c = sqrt(gamma * R * T), u = M * c
        # J- = M*c - 2c/(gamma-1) = c * (M - 2/(gamma-1))
        # c = sqrt(gamma * R * T0 * (1 - (gamma-1)/2 * M^2))

        # Simplified: iterate to find M consistent with J-
        nj = U.shape[2]
        U_ghost = np.zeros((4, nj))

        for j in range(nj):
            M = self._solve_mach(J_minus[j], gamma)
            M = max(min(M, 0.99), 0.01)  # Keep subsonic

            # Isentropic relations
            T_ratio = 1.0 - (gamma - 1) / 2 * M**2
            T = self.T0 * T_ratio
            p = self.p0 * T_ratio**(gamma / (gamma - 1))
            rho = p / (self.R * T)

            c = np.sqrt(gamma * p / rho)
            V = M * c

            u = V * np.cos(self.theta)
            v = V * np.sin(self.theta)

            E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

            U_ghost[0, j] = rho
            U_ghost[1, j] = rho * u
            U_ghost[2, j] = rho * v
            U_ghost[3, j] = E

        return U_ghost

    def _solve_mach(self, J_minus, gamma, tol=1e-6, max_iter=50):
        """Solve for Mach number given J- characteristic."""
        # Initial guess
        M = 0.5

        for _ in range(max_iter):
            T_ratio = 1 - (gamma - 1) / 2 * M**2
            if T_ratio <= 0:
                M *= 0.5
                continue

            c = np.sqrt(gamma * self.R * self.T0 * T_ratio)
            J_calc = M * c - 2 * c / (gamma - 1)

            residual = J_calc - J_minus

            # Derivative: dJ/dM
            dT_dM = -(gamma - 1) * M
            dc_dM = 0.5 * gamma * self.R * self.T0 * dT_dM / (c + 1e-20)
            dJ_dM = c + M * dc_dM - 2 * dc_dM / (gamma - 1)

            if abs(dJ_dM) < 1e-20:
                break

            M_new = M - residual / dJ_dM
            M_new = max(min(M_new, 0.99), 0.01)

            if abs(M_new - M) < tol:
                return M_new

            M = M_new

        return M


class SubsonicOutlet(BoundaryCondition):
    """
    Subsonic outlet boundary condition.

    Specifies static back pressure.
    Extrapolates density, velocity from interior.
    """

    def __init__(self, p_back=0.8):
        """
        Initialize subsonic outlet.

        Parameters
        ----------
        p_back : float
            Static back pressure at outlet
        """
        self.p_back = p_back

    def apply(self, U, metrics, boundary, gamma):
        """Apply subsonic outlet BC."""
        if boundary != 'right':
            raise ValueError("SubsonicOutlet only implemented for right boundary")

        # Interior values at outlet plane
        U_int = U[:, -1, :]
        rho_int = np.maximum(U_int[0], 1e-10)
        u_int = U_int[1] / rho_int
        v_int = U_int[2] / rho_int

        # Extrapolate density and velocity, use specified pressure
        p = self.p_back
        E = p / (gamma - 1) + 0.5 * rho_int * (u_int**2 + v_int**2)

        nj = U.shape[2]
        U_ghost = np.zeros((4, nj))
        U_ghost[0] = rho_int
        U_ghost[1] = rho_int * u_int
        U_ghost[2] = rho_int * v_int
        U_ghost[3] = E

        return U_ghost


class SupersonicInlet(BoundaryCondition):
    """
    Supersonic inlet - all characteristics come from outside.

    All flow quantities are specified.
    """

    def __init__(self, rho=1.0, u=2.0, v=0.0, p=1.0):
        self.rho = rho
        self.u = u
        self.v = v
        self.p = p

    def apply(self, U, metrics, boundary, gamma):
        nj = U.shape[2]
        U_ghost = np.zeros((4, nj))

        E = self.p / (gamma - 1) + 0.5 * self.rho * (self.u**2 + self.v**2)

        U_ghost[0, :] = self.rho
        U_ghost[1, :] = self.rho * self.u
        U_ghost[2, :] = self.rho * self.v
        U_ghost[3, :] = E

        return U_ghost


class SupersonicOutlet(BoundaryCondition):
    """
    Supersonic outlet - all characteristics leave domain.

    Extrapolate all quantities from interior.
    """

    def apply(self, U, metrics, boundary, gamma):
        if boundary == 'right':
            return U[:, -1, :].copy()
        elif boundary == 'left':
            return U[:, 0, :].copy()
        elif boundary == 'top':
            return U[:, :, -1].copy()
        elif boundary == 'bottom':
            return U[:, :, 0].copy()


class Extrapolation(BoundaryCondition):
    """
    Simple zero-gradient (extrapolation) boundary condition.

    Copies interior values to ghost cells.
    """

    def apply(self, U, metrics, boundary, gamma):
        if boundary == 'right':
            return U[:, -1, :].copy()
        elif boundary == 'left':
            return U[:, 0, :].copy()
        elif boundary == 'top':
            return U[:, :, -1].copy()
        elif boundary == 'bottom':
            return U[:, :, 0].copy()


class Periodic(BoundaryCondition):
    """
    Periodic boundary condition.

    Copies values from opposite boundary.
    """

    def apply(self, U, metrics, boundary, gamma):
        if boundary == 'right':
            return U[:, 0, :].copy()
        elif boundary == 'left':
            return U[:, -1, :].copy()
        elif boundary == 'top':
            return U[:, :, 0].copy()
        elif boundary == 'bottom':
            return U[:, :, -1].copy()


class BoundaryConditionManager:
    """
    Manages boundary conditions for all boundaries.

    Provides a unified interface for applying BCs.
    """

    def __init__(self, gamma=1.4):
        self.gamma = gamma
        self.bcs = {
            'left': Extrapolation(),
            'right': Extrapolation(),
            'bottom': SlipWall(),
            'top': SlipWall()
        }

    def set_bc(self, boundary, bc):
        """
        Set boundary condition for a boundary.

        Parameters
        ----------
        boundary : str
            'left', 'right', 'bottom', or 'top'
        bc : BoundaryCondition
            Boundary condition object
        """
        self.bcs[boundary] = bc

    def set_inlet(self, p0=1.0, T0=1.0, theta=0.0, R=1.0):
        """Set subsonic inlet on left boundary."""
        self.bcs['left'] = SubsonicInlet(p0, T0, theta, R)

    def set_outlet(self, p_back=0.8):
        """Set subsonic outlet on right boundary."""
        self.bcs['right'] = SubsonicOutlet(p_back)

    def set_wall(self, boundary):
        """Set slip wall on specified boundary."""
        self.bcs[boundary] = SlipWall()

    def apply_all(self, U, metrics):
        """
        Apply all boundary conditions.

        Parameters
        ----------
        U : ndarray
            Conservative variables (4, ni, nj)
        metrics : GridMetrics
            Grid metrics object

        Returns
        -------
        ghost_cells : dict
            Dictionary of ghost cell values for each boundary
        """
        ghost = {}
        for boundary, bc in self.bcs.items():
            ghost[boundary] = bc.apply(U, metrics, boundary, self.gamma)
        return ghost

    def get_extended_state(self, U, metrics):
        """
        Create extended state array with ghost cells.

        Parameters
        ----------
        U : ndarray
            Conservative variables (4, ni, nj)
        metrics : GridMetrics
            Grid metrics object

        Returns
        -------
        U_ext : ndarray
            Extended array (4, ni+2, nj+2) with ghost cells
        """
        ni, nj = U.shape[1], U.shape[2]
        U_ext = np.zeros((4, ni + 2, nj + 2))

        # Copy interior
        U_ext[:, 1:-1, 1:-1] = U

        # Apply BCs
        ghost = self.apply_all(U, metrics)

        # Fill ghost cells
        U_ext[:, 0, 1:-1] = ghost['left']
        U_ext[:, -1, 1:-1] = ghost['right']
        U_ext[:, 1:-1, 0] = ghost['bottom']
        U_ext[:, 1:-1, -1] = ghost['top']

        # Corners (average of adjacent ghost cells)
        U_ext[:, 0, 0] = 0.5 * (ghost['left'][:, 0] + ghost['bottom'][:, 0])
        U_ext[:, -1, 0] = 0.5 * (ghost['right'][:, 0] + ghost['bottom'][:, -1])
        U_ext[:, 0, -1] = 0.5 * (ghost['left'][:, -1] + ghost['top'][:, 0])
        U_ext[:, -1, -1] = 0.5 * (ghost['right'][:, -1] + ghost['top'][:, -1])

        return U_ext
