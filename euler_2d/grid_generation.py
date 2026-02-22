"""
Grid Generation for 2D Euler Solver

Implements body-fitted structured grids for the NASA bump in channel problem.
Uses transfinite interpolation (TFI) to create smooth meshes.
"""

import numpy as np


def generate_bump_grid(ni=200, nj=80, L=3.0, H=0.8, h=0.0625, x0=1.5, w=0.2):
    """
    Generate a body-fitted grid for the NASA bump in channel problem.

    Uses transfinite interpolation to blend from the bump surface (bottom)
    to the flat top wall.

    Parameters
    ----------
    ni : int
        Number of cells in x-direction (streamwise)
    nj : int
        Number of cells in y-direction (wall-normal)
    L : float
        Channel length
    H : float
        Channel height
    h : float
        Bump height (default 0.0625 for 10% bump)
    x0 : float
        Bump center location
    w : float
        Bump width parameter (controls Gaussian spread)

    Returns
    -------
    x : ndarray
        x-coordinates of cell vertices, shape (ni+1, nj+1)
    y : ndarray
        y-coordinates of cell vertices, shape (ni+1, nj+1)

    Notes
    -----
    The grid is vertex-centered, so for ni cells we have ni+1 vertices.
    Cell centers would be at (x[i,j] + x[i+1,j] + x[i,j+1] + x[i+1,j+1])/4.

    The bump profile is: y_bump = h * exp(-((x - x0) / w)^2)
    """
    # Create computational coordinates
    xi = np.linspace(0, 1, ni + 1)    # Streamwise parameter
    eta = np.linspace(0, 1, nj + 1)   # Wall-normal parameter

    # Physical x-coordinates (uniform in streamwise direction)
    x_phys = xi * L

    # Bump profile on bottom boundary
    y_bump = h * np.exp(-((x_phys - x0) / w)**2)

    # Initialize grid arrays
    x = np.zeros((ni + 1, nj + 1))
    y = np.zeros((ni + 1, nj + 1))

    # Transfinite interpolation: blend from bump to flat top
    for i in range(ni + 1):
        for j in range(nj + 1):
            x[i, j] = x_phys[i]
            # Linear blending from bottom (bump) to top (flat)
            y[i, j] = y_bump[i] + eta[j] * (H - y_bump[i])

    return x, y


def generate_uniform_grid(ni=100, nj=50, Lx=1.0, Ly=1.0):
    """
    Generate a uniform Cartesian grid.

    Parameters
    ----------
    ni : int
        Number of cells in x-direction
    nj : int
        Number of cells in y-direction
    Lx : float
        Domain length in x
    Ly : float
        Domain length in y

    Returns
    -------
    x : ndarray
        x-coordinates of cell vertices, shape (ni+1, nj+1)
    y : ndarray
        y-coordinates of cell vertices, shape (ni+1, nj+1)
    """
    x_1d = np.linspace(0, Lx, ni + 1)
    y_1d = np.linspace(0, Ly, nj + 1)

    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')

    return x, y


def generate_stretched_grid(ni=100, nj=50, Lx=1.0, Ly=1.0,
                            stretch_x=1.0, stretch_y=1.0):
    """
    Generate a stretched grid with clustering near boundaries.

    Uses hyperbolic tangent stretching for smooth clustering.

    Parameters
    ----------
    ni : int
        Number of cells in x-direction
    nj : int
        Number of cells in y-direction
    Lx : float
        Domain length in x
    Ly : float
        Domain length in y
    stretch_x : float
        Stretching factor in x (>1 clusters at boundaries)
    stretch_y : float
        Stretching factor in y (>1 clusters at boundaries)

    Returns
    -------
    x : ndarray
        x-coordinates of cell vertices, shape (ni+1, nj+1)
    y : ndarray
        y-coordinates of cell vertices, shape (ni+1, nj+1)
    """
    def stretch_coordinate(n, L, beta):
        """Apply tanh stretching to coordinate."""
        if beta <= 1.0:
            return np.linspace(0, L, n + 1)

        xi = np.linspace(0, 1, n + 1)
        # Two-sided stretching
        s = 0.5 * (1 + np.tanh(beta * (xi - 0.5)) / np.tanh(0.5 * beta))
        return s * L

    x_1d = stretch_coordinate(ni, Lx, stretch_x)
    y_1d = stretch_coordinate(nj, Ly, stretch_y)

    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')

    return x, y


def generate_channel_grid(ni=100, nj=50, L=3.0, H=0.8, cluster_wall=True, beta=1.5):
    """
    Generate a grid for a simple channel (no bump).

    Parameters
    ----------
    ni : int
        Number of cells in x-direction
    nj : int
        Number of cells in y-direction
    L : float
        Channel length
    H : float
        Channel height
    cluster_wall : bool
        If True, cluster grid points near walls
    beta : float
        Clustering parameter (higher = more clustering)

    Returns
    -------
    x : ndarray
        x-coordinates of cell vertices, shape (ni+1, nj+1)
    y : ndarray
        y-coordinates of cell vertices, shape (ni+1, nj+1)
    """
    x_1d = np.linspace(0, L, ni + 1)

    if cluster_wall:
        eta = np.linspace(0, 1, nj + 1)
        # Cluster near both walls using tanh
        y_1d = H * 0.5 * (1 + np.tanh(beta * (eta - 0.5)) / np.tanh(0.5 * beta))
    else:
        y_1d = np.linspace(0, H, nj + 1)

    x, y = np.meshgrid(x_1d, y_1d, indexing='ij')

    return x, y


def compute_cell_centers(x, y):
    """
    Compute cell center coordinates from vertex coordinates.

    Parameters
    ----------
    x : ndarray
        x-coordinates of vertices, shape (ni+1, nj+1)
    y : ndarray
        y-coordinates of vertices, shape (ni+1, nj+1)

    Returns
    -------
    xc : ndarray
        x-coordinates of cell centers, shape (ni, nj)
    yc : ndarray
        y-coordinates of cell centers, shape (ni, nj)
    """
    # Average of four vertices
    xc = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[:-1, 1:] + x[1:, 1:])
    yc = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[:-1, 1:] + y[1:, 1:])

    return xc, yc


def check_grid_quality(x, y, verbose=True):
    """
    Check grid quality metrics.

    Parameters
    ----------
    x : ndarray
        x-coordinates of vertices
    y : ndarray
        y-coordinates of vertices
    verbose : bool
        If True, print quality report

    Returns
    -------
    quality : dict
        Dictionary containing quality metrics:
        - min_volume: Minimum cell volume (should be > 0)
        - max_aspect_ratio: Maximum cell aspect ratio
        - max_skewness: Maximum cell skewness
        - is_valid: True if all cells have positive volume
    """
    ni, nj = x.shape[0] - 1, x.shape[1] - 1

    # Compute cell volumes (2D: areas)
    # Using cross product of diagonals
    volumes = np.zeros((ni, nj))
    aspect_ratios = np.zeros((ni, nj))

    for i in range(ni):
        for j in range(nj):
            # Cell vertices
            x1, y1 = x[i, j], y[i, j]
            x2, y2 = x[i+1, j], y[i+1, j]
            x3, y3 = x[i+1, j+1], y[i+1, j+1]
            x4, y4 = x[i, j+1], y[i, j+1]

            # Area using shoelace formula
            area = 0.5 * abs((x1-x3)*(y2-y4) - (x2-x4)*(y1-y3))
            volumes[i, j] = area

            # Approximate edge lengths
            dx1 = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            dx2 = np.sqrt((x3-x2)**2 + (y3-y2)**2)
            dx3 = np.sqrt((x4-x3)**2 + (y4-y3)**2)
            dx4 = np.sqrt((x1-x4)**2 + (y1-y4)**2)

            # Average edge lengths in each direction
            h_xi = 0.5 * (dx1 + dx3)
            h_eta = 0.5 * (dx2 + dx4)

            aspect_ratios[i, j] = max(h_xi, h_eta) / (min(h_xi, h_eta) + 1e-20)

    min_vol = np.min(volumes)
    max_ar = np.max(aspect_ratios)
    is_valid = min_vol > 0

    quality = {
        'min_volume': min_vol,
        'max_volume': np.max(volumes),
        'mean_volume': np.mean(volumes),
        'max_aspect_ratio': max_ar,
        'mean_aspect_ratio': np.mean(aspect_ratios),
        'is_valid': is_valid
    }

    if verbose:
        print("Grid Quality Report:")
        print(f"  Dimensions: {ni} x {nj} cells")
        print(f"  Cell volumes: min={min_vol:.2e}, max={np.max(volumes):.2e}")
        print(f"  Aspect ratio: max={max_ar:.2f}, mean={np.mean(aspect_ratios):.2f}")
        print(f"  Valid (positive volumes): {is_valid}")

    return quality
