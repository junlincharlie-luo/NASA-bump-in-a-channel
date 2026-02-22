"""
Sod Shock Tube - 1D Compressible Euler Equation Solver

A finite volume solver for the 1D Euler equations using HLLC/Rusanov flux schemes
with RK2 time integration.
"""

from .euler_solver import EulerSolver
from .flux_schemes import rusanov_flux, hllc_flux
from .exact_solution import SodExactSolution
from .visualization import Visualizer

__version__ = "1.0.0"
__all__ = ["EulerSolver", "rusanov_flux", "hllc_flux", "SodExactSolution", "Visualizer"]
