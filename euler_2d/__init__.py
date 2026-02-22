"""
2D Euler Equation Solver Package

A finite volume solver for the 2D compressible Euler equations on
structured body-fitted grids. Designed for the NASA bump in channel
benchmark problem.

Main Components
---------------
EulerSolver2D : Main solver class
    Cell-centered finite volume solver with HLLC/Rusanov flux schemes
    and RK2 time integration.

Grid Generation : generate_bump_grid, generate_uniform_grid
    Create structured meshes including body-fitted grids for the
    NASA bump geometry.

Boundary Conditions : BoundaryConditionManager
    Subsonic inlet/outlet, slip walls, and other boundary types.

Visualization : Visualizer2D
    Contour plots, pressure coefficient distributions, and
    convergence monitoring.

Example Usage
-------------
>>> from euler_2d import EulerSolver2D, generate_bump_grid, Visualizer2D
>>>
>>> # Generate grid for NASA bump
>>> x, y = generate_bump_grid(ni=200, nj=80)
>>>
>>> # Create solver
>>> solver = EulerSolver2D(x, y, gamma=1.4, cfl=0.5)
>>>
>>> # Set boundary conditions
>>> solver.set_inlet_bc(p0=1.0, T0=1.0, theta=0.0)
>>> solver.set_outlet_bc(p_back=0.8)
>>> solver.set_wall_bc(wall='bottom')
>>> solver.set_wall_bc(wall='top')
>>>
>>> # Initialize and solve
>>> solver.set_uniform_flow(M=0.5, alpha=0.0)
>>> converged, rho, u, v, p = solver.solve_steady(max_iter=5000)
>>>
>>> # Visualize
>>> viz = Visualizer2D()
>>> viz.plot_mach_contours(solver)
"""

from .euler_solver_2d import EulerSolver2D
from .grid_generation import (
    generate_bump_grid,
    generate_uniform_grid,
    generate_stretched_grid,
    generate_channel_grid,
    compute_cell_centers,
    check_grid_quality
)
from .metrics import GridMetrics, compute_simple_metrics
from .boundary_conditions import (
    BoundaryConditionManager,
    SlipWall,
    SubsonicInlet,
    SubsonicOutlet,
    SupersonicInlet,
    SupersonicOutlet,
    Extrapolation,
    Periodic
)
from .flux_schemes_2d import (
    hllc_flux_2d,
    rusanov_flux_2d,
    hllc_flux_1d,
    rusanov_flux_1d
)
from .visualization_2d import Visualizer2D, quick_plot

__all__ = [
    # Main solver
    'EulerSolver2D',

    # Grid generation
    'generate_bump_grid',
    'generate_uniform_grid',
    'generate_stretched_grid',
    'generate_channel_grid',
    'compute_cell_centers',
    'check_grid_quality',

    # Metrics
    'GridMetrics',
    'compute_simple_metrics',

    # Boundary conditions
    'BoundaryConditionManager',
    'SlipWall',
    'SubsonicInlet',
    'SubsonicOutlet',
    'SupersonicInlet',
    'SupersonicOutlet',
    'Extrapolation',
    'Periodic',

    # Flux schemes
    'hllc_flux_2d',
    'rusanov_flux_2d',
    'hllc_flux_1d',
    'rusanov_flux_1d',

    # Visualization
    'Visualizer2D',
    'quick_plot'
]

__version__ = '0.1.0'
