"""
Exact solution for the Sod shock tube problem.

The Sod problem generates three waves:
1. Left-moving rarefaction wave
2. Contact discontinuity
3. Right-moving shock wave

The exact solution is computed by:
1. Newton iteration to find the pressure in the star region
2. Reconstruction of the solution based on wave positions
"""

import numpy as np


class SodExactSolution:
    """
    Exact Riemann solver for the Sod shock tube problem.

    Parameters
    ----------
    gamma : float
        Ratio of specific heats (default 1.4)
    rho_L, u_L, p_L : float
        Left state (default: Sod left state)
    rho_R, u_R, p_R : float
        Right state (default: Sod right state)
    """

    def __init__(self, gamma=1.4,
                 rho_L=1.0, u_L=0.0, p_L=1.0,
                 rho_R=0.125, u_R=0.0, p_R=0.1):
        self.gamma = gamma
        self.rho_L = rho_L
        self.u_L = u_L
        self.p_L = p_L
        self.rho_R = rho_R
        self.u_R = u_R
        self.p_R = p_R

        # Derived quantities
        self.c_L = np.sqrt(gamma * p_L / rho_L)  # Left sound speed
        self.c_R = np.sqrt(gamma * p_R / rho_R)  # Right sound speed

        # Useful constants
        self.g1 = (gamma - 1) / (2 * gamma)
        self.g2 = (gamma + 1) / (2 * gamma)
        self.g3 = 2 * gamma / (gamma - 1)
        self.g4 = 2 / (gamma - 1)
        self.g5 = 2 / (gamma + 1)
        self.g6 = (gamma - 1) / (gamma + 1)
        self.g7 = (gamma - 1) / 2
        self.g8 = gamma - 1

        # Solve for star region pressure and velocity
        self.p_star, self.u_star = self._solve_star_region()

        # Compute star region densities
        self.rho_star_L = self._compute_rho_star_L()
        self.rho_star_R = self._compute_rho_star_R()

        # Compute wave speeds
        self._compute_wave_speeds()

    def _f_L(self, p):
        """Pressure function for left wave."""
        if p > self.p_L:
            # Shock wave
            A = self.g5 / self.rho_L
            B = self.g6 * self.p_L
            return (p - self.p_L) * np.sqrt(A / (p + B))
        else:
            # Rarefaction wave
            return self.g4 * self.c_L * ((p / self.p_L)**self.g1 - 1)

    def _f_R(self, p):
        """Pressure function for right wave."""
        if p > self.p_R:
            # Shock wave
            A = self.g5 / self.rho_R
            B = self.g6 * self.p_R
            return (p - self.p_R) * np.sqrt(A / (p + B))
        else:
            # Rarefaction wave
            return self.g4 * self.c_R * ((p / self.p_R)**self.g1 - 1)

    def _f(self, p):
        """Total pressure function f(p) = f_L(p) + f_R(p) + du."""
        return self._f_L(p) + self._f_R(p) + (self.u_R - self.u_L)

    def _df_L(self, p):
        """Derivative of left pressure function."""
        if p > self.p_L:
            A = self.g5 / self.rho_L
            B = self.g6 * self.p_L
            return np.sqrt(A / (p + B)) * (1 - (p - self.p_L) / (2 * (p + B)))
        else:
            return 1 / (self.rho_L * self.c_L) * (p / self.p_L)**(-self.g2)

    def _df_R(self, p):
        """Derivative of right pressure function."""
        if p > self.p_R:
            A = self.g5 / self.rho_R
            B = self.g6 * self.p_R
            return np.sqrt(A / (p + B)) * (1 - (p - self.p_R) / (2 * (p + B)))
        else:
            return 1 / (self.rho_R * self.c_R) * (p / self.p_R)**(-self.g2)

    def _df(self, p):
        """Derivative of total pressure function."""
        return self._df_L(p) + self._df_R(p)

    def _solve_star_region(self, tol=1e-10, max_iter=100):
        """
        Solve for pressure and velocity in the star region using Newton iteration.
        """
        # Initial guess using PVRS (Primitive Variable Riemann Solver)
        p_old = 0.5 * (self.p_L + self.p_R) - 0.125 * (self.u_R - self.u_L) * \
                (self.rho_L + self.rho_R) * (self.c_L + self.c_R)
        p_old = max(p_old, 1e-10)

        # Newton iteration
        for i in range(max_iter):
            f_val = self._f(p_old)
            df_val = self._df(p_old)

            p_new = p_old - f_val / df_val
            p_new = max(p_new, 1e-10)  # Ensure positive pressure

            if abs(p_new - p_old) / (0.5 * (p_new + p_old)) < tol:
                break

            p_old = p_new

        p_star = p_new
        u_star = 0.5 * (self.u_L + self.u_R) + 0.5 * (self._f_R(p_star) - self._f_L(p_star))

        return p_star, u_star

    def _compute_rho_star_L(self):
        """Compute density behind left wave."""
        if self.p_star > self.p_L:
            # Left shock
            return self.rho_L * ((self.p_star / self.p_L + self.g6) /
                                 (self.g6 * self.p_star / self.p_L + 1))
        else:
            # Left rarefaction
            return self.rho_L * (self.p_star / self.p_L)**(1 / self.gamma)

    def _compute_rho_star_R(self):
        """Compute density behind right wave."""
        if self.p_star > self.p_R:
            # Right shock
            return self.rho_R * ((self.p_star / self.p_R + self.g6) /
                                 (self.g6 * self.p_star / self.p_R + 1))
        else:
            # Right rarefaction
            return self.rho_R * (self.p_star / self.p_R)**(1 / self.gamma)

    def _compute_wave_speeds(self):
        """Compute the speeds of all waves."""
        # Left wave
        if self.p_star > self.p_L:
            # Left shock: single speed
            self.S_L = self.u_L - self.c_L * np.sqrt(self.g2 * self.p_star / self.p_L + self.g1)
            self.S_HL = self.S_L  # Head = tail for shock
            self.S_TL = self.S_L
        else:
            # Left rarefaction: head and tail speeds
            self.c_star_L = self.c_L * (self.p_star / self.p_L)**self.g1
            self.S_HL = self.u_L - self.c_L  # Head
            self.S_TL = self.u_star - self.c_star_L  # Tail

        # Contact discontinuity
        self.S_contact = self.u_star

        # Right wave
        if self.p_star > self.p_R:
            # Right shock
            self.S_R = self.u_R + self.c_R * np.sqrt(self.g2 * self.p_star / self.p_R + self.g1)
            self.S_HR = self.S_R
            self.S_TR = self.S_R
        else:
            # Right rarefaction
            self.c_star_R = self.c_R * (self.p_star / self.p_R)**self.g1
            self.S_HR = self.u_R + self.c_R  # Head
            self.S_TR = self.u_star + self.c_star_R  # Tail

    def sample(self, x, t, x_0=0.5):
        """
        Sample the exact solution at position x and time t.

        Parameters
        ----------
        x : float or ndarray
            Position(s) to evaluate
        t : float
            Time (must be > 0)
        x_0 : float
            Initial discontinuity position

        Returns
        -------
        rho, u, p : floats or ndarrays
            Exact solution values
        """
        if t <= 0:
            # Return initial condition
            x = np.atleast_1d(x)
            rho = np.where(x < x_0, self.rho_L, self.rho_R)
            u = np.where(x < x_0, self.u_L, self.u_R)
            p = np.where(x < x_0, self.p_L, self.p_R)
            return rho, u, p

        # Similarity variable
        x = np.atleast_1d(x)
        S = (x - x_0) / t

        rho = np.zeros_like(S)
        u = np.zeros_like(S)
        p = np.zeros_like(S)

        for i, s in enumerate(S):
            rho[i], u[i], p[i] = self._sample_point(s)

        return rho, u, p

    def _sample_point(self, S):
        """Sample solution at a single point given similarity variable S = (x-x0)/t."""
        # Left of left wave
        if self.p_star > self.p_L:
            # Left shock
            if S < self.S_L:
                return self.rho_L, self.u_L, self.p_L
        else:
            # Left rarefaction
            if S < self.S_HL:
                return self.rho_L, self.u_L, self.p_L
            elif S < self.S_TL:
                # Inside rarefaction fan
                u = self.g5 * (self.c_L + self.g7 * self.u_L + S)
                c = self.g5 * (self.c_L + self.g7 * (self.u_L - S))
                rho = self.rho_L * (c / self.c_L)**self.g4
                p = self.p_L * (c / self.c_L)**self.g3
                return rho, u, p

        # Between left wave and contact
        if S < self.S_contact:
            return self.rho_star_L, self.u_star, self.p_star

        # Between contact and right wave
        if self.p_star > self.p_R:
            # Right shock
            if S < self.S_R:
                return self.rho_star_R, self.u_star, self.p_star
        else:
            # Right rarefaction
            if S < self.S_TR:
                return self.rho_star_R, self.u_star, self.p_star
            elif S < self.S_HR:
                # Inside rarefaction fan
                u = self.g5 * (-self.c_R + self.g7 * self.u_R + S)
                c = self.g5 * (self.c_R - self.g7 * (self.u_R - S))
                rho = self.rho_R * (c / self.c_R)**self.g4
                p = self.p_R * (c / self.c_R)**self.g3
                return rho, u, p

        # Right of right wave
        return self.rho_R, self.u_R, self.p_R

    def get_wave_positions(self, t, x_0=0.5):
        """
        Get the positions of all waves at time t.

        Returns
        -------
        dict with keys:
            'rarefaction_head', 'rarefaction_tail', 'contact', 'shock'
        """
        positions = {}

        if self.p_star <= self.p_L:
            positions['rarefaction_head'] = x_0 + self.S_HL * t
            positions['rarefaction_tail'] = x_0 + self.S_TL * t
        else:
            positions['left_shock'] = x_0 + self.S_L * t

        positions['contact'] = x_0 + self.S_contact * t

        if self.p_star > self.p_R:
            positions['shock'] = x_0 + self.S_R * t
        else:
            positions['right_rarefaction_tail'] = x_0 + self.S_TR * t
            positions['right_rarefaction_head'] = x_0 + self.S_HR * t

        return positions

    def print_solution_info(self):
        """Print information about the exact solution."""
        print("=" * 50)
        print("Sod Shock Tube - Exact Solution")
        print("=" * 50)
        print(f"\nInitial conditions:")
        print(f"  Left:  ρ={self.rho_L}, u={self.u_L}, p={self.p_L}")
        print(f"  Right: ρ={self.rho_R}, u={self.u_R}, p={self.p_R}")
        print(f"\nStar region:")
        print(f"  p* = {self.p_star:.6f}")
        print(f"  u* = {self.u_star:.6f}")
        print(f"  ρ*_L = {self.rho_star_L:.6f}")
        print(f"  ρ*_R = {self.rho_star_R:.6f}")
        print(f"\nWave structure:")
        if self.p_star > self.p_L:
            print(f"  Left shock: S = {self.S_L:.6f}")
        else:
            print(f"  Left rarefaction: head = {self.S_HL:.6f}, tail = {self.S_TL:.6f}")
        print(f"  Contact: S = {self.S_contact:.6f}")
        if self.p_star > self.p_R:
            print(f"  Right shock: S = {self.S_R:.6f}")
        else:
            print(f"  Right rarefaction: tail = {self.S_TR:.6f}, head = {self.S_HR:.6f}")

        # Wave positions at t=0.25
        t = 0.25
        pos = self.get_wave_positions(t)
        print(f"\nWave positions at t={t}:")
        for name, x in pos.items():
            print(f"  {name}: x = {x:.4f}")
        print("=" * 50)
