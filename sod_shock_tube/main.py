#!/usr/bin/env python3
"""
Sod Shock Tube Solver - Main Program

Solves the 1D Sod shock tube problem using finite volume methods.
Compares numerical solution with exact solution and generates visualizations.

Usage:
    python -m sod_shock_tube.main [options]

Options:
    --nx          Number of grid cells (default: 400)
    --cfl         CFL number (default: 0.5)
    --t_end       Final time (default: 0.25)
    --flux        Flux scheme: 'hllc' or 'rusanov' (default: 'hllc')
    --output_dir  Output directory (default: 'output/results')
    --no_animation Skip animation generation
    --verbose     Print detailed progress
"""

import argparse
import os
import sys
import time

import numpy as np

from .euler_solver import EulerSolver
from .exact_solution import SodExactSolution
from .visualization import Visualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Sod Shock Tube Solver',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--nx', type=int, default=400,
                        help='Number of grid cells')
    parser.add_argument('--cfl', type=float, default=0.5,
                        help='CFL number for stability')
    parser.add_argument('--t_end', type=float, default=0.25,
                        help='Final simulation time')
    parser.add_argument('--gamma', type=float, default=1.4,
                        help='Ratio of specific heats')
    parser.add_argument('--flux', type=str, default='hllc',
                        choices=['hllc', 'rusanov'],
                        help='Numerical flux scheme')
    parser.add_argument('--output_dir', type=str, default='output/results',
                        help='Output directory')
    parser.add_argument('--no_animation', action='store_true',
                        help='Skip animation generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print detailed progress')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Sod Shock Tube Solver")
    print("1D Compressible Euler Equations")
    print("=" * 60)

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Grid cells:    {args.nx}")
    print(f"  CFL number:    {args.cfl}")
    print(f"  Final time:    {args.t_end}")
    print(f"  Gamma:         {args.gamma}")
    print(f"  Flux scheme:   {args.flux.upper()}")
    print(f"  Output dir:    {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize solver
    print("\n" + "-" * 60)
    print("Initializing solver...")
    solver = EulerSolver(
        nx=args.nx,
        x_range=(0.0, 1.0),
        gamma=args.gamma,
        cfl=args.cfl,
        flux_scheme=args.flux
    )

    # Set initial condition (standard Sod problem)
    solver.set_sod_initial_condition(x_discontinuity=0.5)

    # Initialize exact solution
    exact = SodExactSolution(gamma=args.gamma)
    if args.verbose:
        exact.print_solution_info()

    # Run simulation
    print("\n" + "-" * 60)
    print("Running simulation...")
    start_time = time.time()

    rho, u, p = solver.solve(
        t_end=args.t_end,
        save_interval=args.t_end / 100,  # 100 frames for animation
        verbose=args.verbose
    )

    elapsed = time.time() - start_time
    print(f"Simulation completed in {elapsed:.2f} seconds")

    # Compute exact solution at final time
    x = solver.x
    rho_exact, u_exact, p_exact = exact.sample(x, args.t_end, x_0=0.5)

    # Compute errors
    errors = solver.compute_errors(rho_exact, u_exact, p_exact)

    # Initialize visualizer
    viz = Visualizer(output_dir=args.output_dir)

    # Print error summary
    viz.print_error_summary(errors)

    # Generate final comparison plot
    print("\n" + "-" * 60)
    print("Generating visualizations...")

    viz.plot_comparison(
        x, rho, u, p,
        rho_exact, u_exact, p_exact,
        t=args.t_end,
        title=f'Sod Shock Tube: {args.flux.upper()} flux, nx={args.nx}, t={args.t_end}'
    )

    # Generate error plot
    viz.plot_errors(x, rho, u, p, rho_exact, u_exact, p_exact)

    # Save numerical data
    viz.save_data(x, rho, u, p, args.t_end)

    # Generate animation
    if not args.no_animation:
        print("Generating animation (this may take a moment)...")
        viz.create_animation(
            x, solver.history, solver.time_history,
            gamma=args.gamma,
            exact_solver=exact,
            fps=15
        )

    # Print wave positions
    print("\n" + "-" * 60)
    print(f"Wave positions at t = {args.t_end}:")
    positions = exact.get_wave_positions(args.t_end)
    for name, pos in positions.items():
        print(f"  {name}: x = {pos:.4f}")

    print("\n" + "=" * 60)
    print("All outputs saved to:", args.output_dir)
    print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
