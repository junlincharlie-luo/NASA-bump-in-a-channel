#!/usr/bin/env python3
"""
NASA Bump in Channel - 2D Euler Solver

This script runs the NASA bump in channel benchmark problem using
the 2D Euler solver. The bump geometry is a Gaussian profile on the
lower wall of a channel.

Benchmark problem:
- 2D channel with Gaussian bump on lower wall
- Subsonic flow (M ~ 0.5) with inlet/outlet conditions
- Tests grid generation, boundary conditions, and solver accuracy

Reference:
- NASA Verification and Validation website
- Classic CFD validation case for compressible flow solvers
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from euler_2d import (
    EulerSolver2D,
    generate_bump_grid,
    Visualizer2D,
    check_grid_quality
)


def run_nasa_bump(ni=200, nj=80, M_inf=0.5, cfl=0.5, max_iter=10000, tol=1e-6,
                  flux_scheme='hllc', verbose=True, save_plots=True,
                  output_dir='results'):
    """
    Run the NASA bump in channel simulation.

    Parameters
    ----------
    ni : int
        Number of cells in streamwise direction
    nj : int
        Number of cells in wall-normal direction
    M_inf : float
        Freestream Mach number
    cfl : float
        CFL number for time stepping
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    flux_scheme : str
        'hllc' or 'rusanov'
    verbose : bool
        Print progress
    save_plots : bool
        Save plots to files
    output_dir : str
        Directory for output files

    Returns
    -------
    solver : EulerSolver2D
        Solver object with converged solution
    """
    print("=" * 60)
    print("NASA Bump in Channel - 2D Euler Solver")
    print("=" * 60)

    # Grid parameters (NASA bump geometry)
    L = 3.0       # Channel length
    H = 0.8       # Channel height
    h = 0.0625    # Bump height (approximately 10% of channel at bump center)
    x0 = 1.5      # Bump center
    w = 0.2       # Bump width parameter

    print(f"\nGrid parameters:")
    print(f"  Domain: {L} x {H}")
    print(f"  Cells: {ni} x {nj}")
    print(f"  Bump: height={h}, center={x0}, width={w}")

    # Generate grid
    print("\nGenerating body-fitted grid...")
    x, y = generate_bump_grid(ni=ni, nj=nj, L=L, H=H, h=h, x0=x0, w=w)

    # Check grid quality
    quality = check_grid_quality(x, y, verbose=verbose)
    if not quality['is_valid']:
        print("ERROR: Invalid grid (negative volumes)")
        return None

    # Create solver
    print(f"\nCreating solver (flux scheme: {flux_scheme})...")
    solver = EulerSolver2D(x, y, gamma=1.4, cfl=cfl, flux_scheme=flux_scheme)

    # Set boundary conditions
    # Subsonic inlet: specify total conditions
    # For isentropic flow: p0/p = (1 + (gamma-1)/2 * M^2)^(gamma/(gamma-1))
    gamma = 1.4
    p_static = 1.0
    T_static = 1.0
    T0 = T_static * (1 + (gamma - 1) / 2 * M_inf**2)
    p0 = p_static * (1 + (gamma - 1) / 2 * M_inf**2)**(gamma / (gamma - 1))

    print(f"\nBoundary conditions:")
    print(f"  Inlet: p0={p0:.4f}, T0={T0:.4f}, M_inf={M_inf}")
    print(f"  Outlet: p_back={p_static}")
    print(f"  Walls: slip (inviscid)")

    solver.set_inlet_bc(p0=p0, T0=T0, theta=0.0, R=1.0)
    solver.set_outlet_bc(p_back=p_static)
    solver.set_wall_bc(wall='bottom')
    solver.set_wall_bc(wall='top')

    # Initialize with uniform flow
    print(f"\nInitializing with uniform flow at M={M_inf}...")
    solver.set_uniform_flow(M=M_inf, alpha=0.0, p=p_static, rho=1.0)

    # Solve to steady state
    print(f"\nSolving to steady state (max_iter={max_iter}, tol={tol})...")
    converged, rho, u, v, p = solver.solve_steady(
        max_iter=max_iter,
        tol=tol,
        verbose=verbose,
        print_interval=500
    )

    # Check solution
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)

    M = solver.get_mach_number()
    print(f"  Converged: {converged}")
    print(f"  Mach number: min={np.min(M):.4f}, max={np.max(M):.4f}")
    print(f"  Pressure: min={np.min(p):.4f}, max={np.max(p):.4f}")
    print(f"  Density: min={np.min(rho):.4f}, max={np.max(rho):.4f}")

    # Mass conservation check
    mdot_in = solver.compute_mass_flow_rate('left')
    mdot_out = solver.compute_mass_flow_rate('right')
    mass_error = abs(mdot_in - mdot_out) / (abs(mdot_in) + 1e-10) * 100

    print(f"\nMass flow rates:")
    print(f"  Inlet:  {mdot_in:.6f}")
    print(f"  Outlet: {mdot_out:.6f}")
    print(f"  Error:  {mass_error:.4f}%")

    # Visualization
    if save_plots:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"\nSaving plots to {output_dir}/...")

        viz = Visualizer2D()

        # Grid
        viz.plot_grid(x, y, save_path=f"{output_dir}/grid.png",
                      title=f'NASA Bump Grid ({ni}x{nj})')

        # Mach contours
        viz.plot_mach_contours(solver, levels=30,
                               save_path=f"{output_dir}/mach_contours.png",
                               title=f'Mach Number (M_inf={M_inf})')

        # Pressure contours
        viz.plot_pressure_contours(solver, levels=30,
                                   save_path=f"{output_dir}/pressure_contours.png",
                                   title='Static Pressure')

        # Pressure coefficient on bump
        viz.plot_pressure_coefficient(solver, M_inf=M_inf,
                                      save_path=f"{output_dir}/cp_distribution.png",
                                      title='Pressure Coefficient on Bump Surface')

        # Convergence history
        if solver.residual_history:
            viz.plot_convergence(solver, save_path=f"{output_dir}/convergence.png",
                                title='Convergence History')

        # Density contours
        viz.plot_density_contours(solver, levels=30,
                                  save_path=f"{output_dir}/density_contours.png",
                                  title='Density')

        print("  Done!")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)

    return solver


def run_validation_tests():
    """Run validation tests for the 2D solver."""
    print("\n" + "=" * 60)
    print("Running Validation Tests")
    print("=" * 60)

    # Test 1: Uniform flow preservation
    print("\n1. Uniform Flow Preservation Test")
    print("-" * 40)

    x, y = generate_bump_grid(ni=50, nj=20, h=0.0)  # Flat channel
    solver = EulerSolver2D(x, y, gamma=1.4, cfl=0.5)

    solver.set_uniform_flow(M=0.5, alpha=0.0, p=1.0, rho=1.0)
    solver.set_inlet_bc(p0=1.186, T0=1.05, theta=0.0)
    solver.set_outlet_bc(p_back=1.0)
    solver.set_wall_bc('bottom')
    solver.set_wall_bc('top')

    # Run for a few steps
    for _ in range(100):
        dt = solver.compute_time_step()
        solver.U = solver.rk2_step(solver.U, dt)

    M = solver.get_mach_number()
    M_variation = np.max(M) - np.min(M)
    print(f"  Initial Mach: 0.5")
    print(f"  Final Mach range: [{np.min(M):.6f}, {np.max(M):.6f}]")
    print(f"  Mach variation: {M_variation:.6e}")
    print(f"  PASS: {M_variation < 0.01}")

    # Test 2: Mass conservation
    print("\n2. Mass Conservation Test")
    print("-" * 40)

    x, y = generate_bump_grid(ni=100, nj=40)
    solver = EulerSolver2D(x, y, gamma=1.4, cfl=0.5)

    solver.set_uniform_flow(M=0.5, alpha=0.0)
    solver.set_inlet_bc(p0=1.186, T0=1.05, theta=0.0)
    solver.set_outlet_bc(p_back=1.0)
    solver.set_wall_bc('bottom')
    solver.set_wall_bc('top')

    # Run to partial convergence
    solver.solve_steady(max_iter=1000, tol=1e-4, verbose=False)

    mdot_in = solver.compute_mass_flow_rate('left')
    mdot_out = solver.compute_mass_flow_rate('right')
    mass_error = abs(mdot_in - mdot_out) / abs(mdot_in) * 100

    print(f"  Inlet mass flow:  {mdot_in:.6f}")
    print(f"  Outlet mass flow: {mdot_out:.6f}")
    print(f"  Mass error: {mass_error:.4f}%")
    print(f"  PASS: {mass_error < 1.0}")

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)


def main():
    """Main entry point with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='NASA Bump in Channel - 2D Euler Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--ni', type=int, default=200,
                        help='Number of cells in x-direction')
    parser.add_argument('--nj', type=int, default=80,
                        help='Number of cells in y-direction')
    parser.add_argument('--mach', type=float, default=0.5,
                        help='Freestream Mach number')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number')
    parser.add_argument('--max-iter', type=int, default=10000,
                        help='Maximum iterations')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='Convergence tolerance')
    parser.add_argument('--flux', choices=['hllc', 'rusanov'], default='hllc',
                        help='Flux scheme')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save plots')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    parser.add_argument('--validate', action='store_true',
                        help='Run validation tests instead of main simulation')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')

    args = parser.parse_args()

    if args.validate:
        run_validation_tests()
    else:
        solver = run_nasa_bump(
            ni=args.ni,
            nj=args.nj,
            M_inf=args.mach,
            cfl=args.cfl,
            max_iter=args.max_iter,
            tol=args.tol,
            flux_scheme=args.flux,
            verbose=not args.quiet,
            save_plots=not args.no_save,
            output_dir=args.output
        )

        if args.show:
            plt.show()


if __name__ == '__main__':
    main()
