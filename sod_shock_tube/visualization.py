"""
Visualization module for Sod shock tube results.

Provides:
- Static comparison plots (numerical vs exact)
- Animated GIF of solution evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os


class Visualizer:
    """
    Visualization tools for Euler equation solutions.

    Parameters
    ----------
    output_dir : str
        Directory for saving output files
    """

    def __init__(self, output_dir='output/results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = {
            'numerical': '#2196F3',  # Blue
            'exact': '#F44336',      # Red
        }

    def plot_comparison(self, x, rho_num, u_num, p_num,
                        rho_exact=None, u_exact=None, p_exact=None,
                        t=None, title=None, filename='sod_final_solution.png'):
        """
        Plot numerical solution with optional exact solution comparison.

        Parameters
        ----------
        x : ndarray
            Grid points
        rho_num, u_num, p_num : ndarrays
            Numerical solution
        rho_exact, u_exact, p_exact : ndarrays, optional
            Exact solution for comparison
        t : float, optional
            Time for title
        title : str, optional
            Custom title
        filename : str
            Output filename
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        variables = [
            ('Density (ρ)', rho_num, rho_exact),
            ('Velocity (u)', u_num, u_exact),
            ('Pressure (p)', p_num, p_exact),
        ]

        for ax, (name, num, exact) in zip(axes, variables):
            # Plot numerical solution
            ax.plot(x, num, 'o-', color=self.colors['numerical'],
                    markersize=1.5, linewidth=0.8, label='Numerical')

            # Plot exact solution if provided
            if exact is not None:
                ax.plot(x, exact, '-', color=self.colors['exact'],
                        linewidth=2, label='Exact')

            ax.set_xlabel('x')
            ax.set_ylabel(name)
            ax.legend(loc='best')
            ax.set_xlim(x[0], x[-1])

        if title:
            fig.suptitle(title, fontsize=14)
        elif t is not None:
            fig.suptitle(f'Sod Shock Tube Solution at t = {t:.4f}', fontsize=14)

        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

        return filepath

    def plot_errors(self, x, rho_num, u_num, p_num,
                    rho_exact, u_exact, p_exact,
                    filename='sod_errors.png'):
        """Plot error distributions."""
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        variables = [
            ('Density Error', rho_num - rho_exact),
            ('Velocity Error', u_num - u_exact),
            ('Pressure Error', p_num - p_exact),
        ]

        for ax, (name, error) in zip(axes, variables):
            ax.plot(x, error, '-', color='#9C27B0', linewidth=1)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            ax.set_xlabel('x')
            ax.set_ylabel(name)
            ax.set_xlim(x[0], x[-1])

        fig.suptitle('Numerical Error Distribution', fontsize=14)
        plt.tight_layout()

        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filepath}")

        return filepath

    def create_animation(self, x, history, time_history, gamma,
                         exact_solver=None, filename='sod_animation.gif',
                         fps=10, x_0=0.5):
        """
        Create animated GIF of solution evolution.

        Parameters
        ----------
        x : ndarray
            Grid points
        history : list of ndarrays
            Solution history [U_0, U_1, ...]
        time_history : list of floats
            Time stamps
        gamma : float
            Ratio of specific heats
        exact_solver : SodExactSolution, optional
            Exact solution for comparison
        filename : str
            Output filename
        fps : int
            Frames per second
        x_0 : float
            Initial discontinuity position
        """
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Initialize plots
        lines_num = []
        lines_exact = []

        for ax in axes:
            line_num, = ax.plot([], [], 'o-', color=self.colors['numerical'],
                               markersize=1.5, linewidth=0.8, label='Numerical')
            lines_num.append(line_num)

            if exact_solver:
                line_exact, = ax.plot([], [], '-', color=self.colors['exact'],
                                     linewidth=2, label='Exact')
                lines_exact.append(line_exact)

        # Set labels and limits
        labels = ['Density (ρ)', 'Velocity (u)', 'Pressure (p)']
        y_limits = [(0, 1.1), (-0.1, 1.0), (0, 1.1)]

        for ax, label, ylim in zip(axes, labels, y_limits):
            ax.set_xlabel('x')
            ax.set_ylabel(label)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(ylim)
            ax.legend(loc='best')

        title = fig.suptitle('', fontsize=14)
        plt.tight_layout()

        def conservative_to_primitive(U):
            rho = U[0]
            u = U[1] / rho
            p = (gamma - 1) * (U[2] - 0.5 * rho * u**2)
            return rho, u, p

        def init():
            for line in lines_num + lines_exact:
                line.set_data([], [])
            return lines_num + lines_exact

        def animate(frame):
            U = history[frame]
            t = time_history[frame]

            rho, u, p = conservative_to_primitive(U)

            lines_num[0].set_data(x, rho)
            lines_num[1].set_data(x, u)
            lines_num[2].set_data(x, p)

            if exact_solver and t > 0:
                rho_ex, u_ex, p_ex = exact_solver.sample(x, t, x_0)
                lines_exact[0].set_data(x, rho_ex)
                lines_exact[1].set_data(x, u_ex)
                lines_exact[2].set_data(x, p_ex)

            title.set_text(f'Sod Shock Tube: t = {t:.4f}')

            return lines_num + lines_exact

        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=len(history), interval=1000/fps, blit=True)

        filepath = os.path.join(self.output_dir, filename)
        anim.save(filepath, writer='pillow', fps=fps)
        plt.close()
        print(f"Saved: {filepath}")

        return filepath

    def save_data(self, x, rho, u, p, t, filename='solution_data.npz'):
        """Save solution data to NumPy archive."""
        filepath = os.path.join(self.output_dir, filename)
        np.savez(filepath, x=x, rho=rho, u=u, p=p, t=t)
        print(f"Saved: {filepath}")
        return filepath

    def print_error_summary(self, errors):
        """Print formatted error summary."""
        print("\n" + "=" * 50)
        print("Error Summary (Numerical vs Exact)")
        print("=" * 50)
        print(f"{'Variable':<10} {'L1 Error':<15} {'L2 Error':<15} {'L∞ Error':<15}")
        print("-" * 50)
        for var in ['rho', 'u', 'p']:
            l1 = errors[f'{var}_L1']
            l2 = errors[f'{var}_L2']
            linf = errors[f'{var}_Linf']
            print(f"{var:<10} {l1:<15.6e} {l2:<15.6e} {linf:<15.6e}")
        print("=" * 50)
