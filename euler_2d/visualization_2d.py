"""
Visualization Tools for 2D Euler Solver

Provides contour plots, streamlines, and validation plots for the
NASA bump in channel problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm


class Visualizer2D:
    """
    Visualization tools for 2D flow solutions.

    Provides various plotting methods for flow fields on structured grids.
    """

    def __init__(self, figsize=(12, 6)):
        """
        Initialize visualizer.

        Parameters
        ----------
        figsize : tuple
            Default figure size
        """
        self.figsize = figsize
        self.default_cmap = 'viridis'

    def plot_mach_contours(self, solver, levels=20, show_grid=False,
                           save_path=None, title='Mach Number'):
        """
        Plot Mach number contours.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        levels : int or array
            Number of contour levels or explicit levels
        show_grid : bool
            Overlay grid lines
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        M = solver.get_mach_number()
        xc, yc = solver.metrics.xc, solver.metrics.yc

        cf = ax.contourf(xc, yc, M, levels=levels, cmap='jet')
        ax.contour(xc, yc, M, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

        if show_grid:
            self._add_grid(ax, solver.x, solver.y)

        plt.colorbar(cf, ax=ax, label='Mach Number')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        # Plot bump surface
        ax.fill_between(solver.x[:, 0], 0, solver.y[:, 0], color='gray', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_pressure_contours(self, solver, levels=20, save_path=None,
                               title='Static Pressure'):
        """
        Plot pressure contours.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        levels : int
            Number of contour levels
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        _, _, _, p = solver.conservative_to_primitive(solver.U)
        xc, yc = solver.metrics.xc, solver.metrics.yc

        cf = ax.contourf(xc, yc, p, levels=levels, cmap='coolwarm')
        ax.contour(xc, yc, p, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

        plt.colorbar(cf, ax=ax, label='Pressure')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        ax.fill_between(solver.x[:, 0], 0, solver.y[:, 0], color='gray', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_density_contours(self, solver, levels=20, save_path=None,
                              title='Density'):
        """
        Plot density contours.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        levels : int
            Number of contour levels
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        rho, _, _, _ = solver.conservative_to_primitive(solver.U)
        xc, yc = solver.metrics.xc, solver.metrics.yc

        cf = ax.contourf(xc, yc, rho, levels=levels, cmap='viridis')
        ax.contour(xc, yc, rho, levels=levels, colors='k', linewidths=0.3, alpha=0.5)

        plt.colorbar(cf, ax=ax, label='Density')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        ax.fill_between(solver.x[:, 0], 0, solver.y[:, 0], color='gray', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_pressure_coefficient(self, solver, M_inf=0.5, p_inf=None,
                                  save_path=None, title='Pressure Coefficient on Bump'):
        """
        Plot pressure coefficient distribution on the bump surface.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        M_inf : float
            Freestream Mach number
        p_inf : float, optional
            Freestream pressure (default: compute from inlet conditions)
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # Get pressure at bottom boundary (bump surface)
        _, _, _, p = solver.conservative_to_primitive(solver.U)
        p_wall = p[:, 0]  # Bottom row
        x_wall = solver.metrics.xc[:, 0]

        # Compute Cp
        if p_inf is None:
            p_inf = 1.0  # Normalized

        rho_inf = 1.0
        c_inf = np.sqrt(solver.gamma * p_inf / rho_inf)
        V_inf = M_inf * c_inf
        q_inf = 0.5 * rho_inf * V_inf**2

        Cp = (p_wall - p_inf) / q_inf

        ax.plot(x_wall, -Cp, 'b-', linewidth=2, label='Computed')
        ax.set_xlabel('x')
        ax.set_ylabel('$-C_p$')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Mark bump location
        ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, label='Bump center')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_convergence(self, solver, save_path=None, title='Convergence History'):
        """
        Plot residual convergence history.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with residual history
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        if not solver.residual_history:
            print("No residual history available")
            return None, None

        iterations = range(len(solver.residual_history))
        ax.semilogy(iterations, solver.residual_history, 'b-', linewidth=1.5)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Density Residual (L2)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_grid(self, x, y, save_path=None, title='Computational Grid'):
        """
        Plot the computational grid.

        Parameters
        ----------
        x, y : ndarray
            Vertex coordinates
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Plot grid lines
        ni, nj = x.shape[0], x.shape[1]

        # i-lines (constant i)
        for i in range(ni):
            ax.plot(x[i, :], y[i, :], 'b-', linewidth=0.5)

        # j-lines (constant j)
        for j in range(nj):
            ax.plot(x[:, j], y[:, j], 'b-', linewidth=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_velocity_vectors(self, solver, skip=5, scale=20, save_path=None,
                              title='Velocity Field'):
        """
        Plot velocity vectors.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        skip : int
            Plot every skip-th vector
        scale : float
            Vector scaling factor
        save_path : str, optional
            Path to save figure
        title : str
            Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        _, u, v, _ = solver.conservative_to_primitive(solver.U)
        xc, yc = solver.metrics.xc, solver.metrics.yc

        # Subsample
        xs = xc[::skip, ::skip]
        ys = yc[::skip, ::skip]
        us = u[::skip, ::skip]
        vs = v[::skip, ::skip]

        # Velocity magnitude for coloring
        V = np.sqrt(u**2 + v**2)
        Vs = V[::skip, ::skip]

        ax.quiver(xs, ys, us, vs, Vs, cmap='viridis', scale=scale)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')

        ax.fill_between(solver.x[:, 0], 0, solver.y[:, 0], color='gray', alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, ax

    def plot_all(self, solver, M_inf=0.5, save_prefix=None):
        """
        Generate all standard plots.

        Parameters
        ----------
        solver : EulerSolver2D
            Solver object with solution
        M_inf : float
            Freestream Mach number for Cp calculation
        save_prefix : str, optional
            Prefix for saved files
        """
        plots = []

        # Mach contours
        save_path = f"{save_prefix}_mach.png" if save_prefix else None
        fig, ax = self.plot_mach_contours(solver, save_path=save_path)
        plots.append(('mach', fig, ax))

        # Pressure contours
        save_path = f"{save_prefix}_pressure.png" if save_prefix else None
        fig, ax = self.plot_pressure_contours(solver, save_path=save_path)
        plots.append(('pressure', fig, ax))

        # Pressure coefficient
        save_path = f"{save_prefix}_cp.png" if save_prefix else None
        fig, ax = self.plot_pressure_coefficient(solver, M_inf=M_inf, save_path=save_path)
        plots.append(('cp', fig, ax))

        # Convergence
        if solver.residual_history:
            save_path = f"{save_prefix}_convergence.png" if save_prefix else None
            fig, ax = self.plot_convergence(solver, save_path=save_path)
            plots.append(('convergence', fig, ax))

        return plots

    def _add_grid(self, ax, x, y, color='k', alpha=0.2, linewidth=0.3):
        """Add grid lines to plot."""
        ni, nj = x.shape

        for i in range(ni):
            ax.plot(x[i, :], y[i, :], color=color, alpha=alpha, linewidth=linewidth)

        for j in range(nj):
            ax.plot(x[:, j], y[:, j], color=color, alpha=alpha, linewidth=linewidth)


def quick_plot(solver, quantity='mach', **kwargs):
    """
    Quick plotting function.

    Parameters
    ----------
    solver : EulerSolver2D
        Solver with solution
    quantity : str
        'mach', 'pressure', 'density', 'cp', or 'velocity'
    **kwargs
        Additional arguments passed to plot function
    """
    viz = Visualizer2D()

    if quantity == 'mach':
        return viz.plot_mach_contours(solver, **kwargs)
    elif quantity == 'pressure':
        return viz.plot_pressure_contours(solver, **kwargs)
    elif quantity == 'density':
        return viz.plot_density_contours(solver, **kwargs)
    elif quantity == 'cp':
        return viz.plot_pressure_coefficient(solver, **kwargs)
    elif quantity == 'velocity':
        return viz.plot_velocity_vectors(solver, **kwargs)
    else:
        raise ValueError(f"Unknown quantity: {quantity}")
