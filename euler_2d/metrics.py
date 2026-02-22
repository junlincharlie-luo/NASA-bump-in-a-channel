"""
Grid Metrics for 2D Curvilinear Grids

Computes transformation metrics needed for finite volume discretization
on body-fitted grids:
- Cell volumes (areas in 2D)
- Face areas (lengths in 2D)
- Face normal vectors
- Jacobian of coordinate transformation

For a structured grid with computational coordinates (xi, eta) and
physical coordinates (x, y), the metrics relate derivatives in the
two systems.
"""

import numpy as np


class GridMetrics:
    """
    Computes and stores grid metrics for a structured 2D grid.

    The grid is assumed to have cell-centered data storage with
    vertices at the corners of each cell.

    Attributes
    ----------
    ni : int
        Number of cells in i-direction
    nj : int
        Number of cells in j-direction
    x : ndarray
        Vertex x-coordinates, shape (ni+1, nj+1)
    y : ndarray
        Vertex y-coordinates, shape (ni+1, nj+1)
    volume : ndarray
        Cell volumes (areas), shape (ni, nj)
    xc, yc : ndarray
        Cell center coordinates, shape (ni, nj)

    Face metrics (for i-faces, between cells i and i+1):
    Si_x, Si_y : ndarray
        i-face area vectors (S_x, S_y), shape (ni+1, nj)
        Normal vector = (Si_x, Si_y) / |S|

    Face metrics (for j-faces, between cells j and j+1):
    Sj_x, Sj_y : ndarray
        j-face area vectors, shape (ni, nj+1)
    """

    def __init__(self, x, y):
        """
        Initialize grid metrics from vertex coordinates.

        Parameters
        ----------
        x : ndarray
            Vertex x-coordinates, shape (ni+1, nj+1)
        y : ndarray
            Vertex y-coordinates, shape (ni+1, nj+1)
        """
        self.x = x
        self.y = y
        self.ni = x.shape[0] - 1
        self.nj = x.shape[1] - 1

        self._compute_cell_centers()
        self._compute_cell_volumes()
        self._compute_face_metrics()

    def _compute_cell_centers(self):
        """Compute cell center coordinates."""
        self.xc = 0.25 * (self.x[:-1, :-1] + self.x[1:, :-1] +
                         self.x[:-1, 1:] + self.x[1:, 1:])
        self.yc = 0.25 * (self.y[:-1, :-1] + self.y[1:, :-1] +
                         self.y[:-1, 1:] + self.y[1:, 1:])

    def _compute_cell_volumes(self):
        """
        Compute cell volumes (areas in 2D) using cross product of diagonals.

        For a quadrilateral cell with vertices (x1,y1), (x2,y2), (x3,y3), (x4,y4)
        ordered counterclockwise, the area is:
        A = 0.5 * |diag1 x diag2| = 0.5 * |(x3-x1)(y4-y2) - (x4-x2)(y3-y1)|
        """
        # Diagonal vectors: (1,1)-(3,3) and (2,2)-(4,4) in local numbering
        # Using vertices: [i,j], [i+1,j], [i+1,j+1], [i,j+1]
        dx13 = self.x[1:, 1:] - self.x[:-1, :-1]  # i+1,j+1 to i,j
        dy13 = self.y[1:, 1:] - self.y[:-1, :-1]
        dx24 = self.x[1:, :-1] - self.x[:-1, 1:]  # i+1,j to i,j+1
        dy24 = self.y[1:, :-1] - self.y[:-1, 1:]

        self.volume = 0.5 * np.abs(dx13 * dy24 - dx24 * dy13)

    def _compute_face_metrics(self):
        """
        Compute face area vectors for both i-faces and j-faces.

        For finite volume method, we need the face area vector S = n * |A|
        where n is the outward unit normal and |A| is the face area (length in 2D).

        For i-faces (constant i, between cells i-1 and i):
        The face connects vertices (i,j) to (i,j+1).
        Face vector points in +i direction (from i-1 cell to i cell).
        S_i = (dy, -dx) where dx = x[i,j+1] - x[i,j], dy = y[i,j+1] - y[i,j]

        For j-faces (constant j, between cells j-1 and j):
        The face connects vertices (i,j) to (i+1,j).
        Face vector points in +j direction.
        S_j = (-dy, dx) where dx = x[i+1,j] - x[i,j], dy = y[i+1,j] - y[i,j]
        """
        # i-faces: ni+1 faces in i-direction, nj faces (one per cell row)
        # Face from (i,j) to (i,j+1)
        dx_i = self.x[:, 1:] - self.x[:, :-1]   # shape: (ni+1, nj)
        dy_i = self.y[:, 1:] - self.y[:, :-1]

        # Normal pointing in +i direction: rotate tangent vector by -90 degrees
        # tangent = (dx, dy) -> normal = (dy, -dx)
        self.Si_x = dy_i    # x-component of face area vector
        self.Si_y = -dx_i   # y-component of face area vector
        self.Si_mag = np.sqrt(self.Si_x**2 + self.Si_y**2)

        # j-faces: ni faces, nj+1 faces in j-direction
        # Face from (i,j) to (i+1,j)
        dx_j = self.x[1:, :] - self.x[:-1, :]   # shape: (ni, nj+1)
        dy_j = self.y[1:, :] - self.y[:-1, :]

        # Normal pointing in +j direction: rotate tangent by +90 degrees
        # tangent = (dx, dy) -> normal = (-dy, dx)
        self.Sj_x = -dy_j
        self.Sj_y = dx_j
        self.Sj_mag = np.sqrt(self.Sj_x**2 + self.Sj_y**2)

    def get_i_face_normal(self, i, j):
        """
        Get unit normal vector for i-face at position (i, j).

        Parameters
        ----------
        i : int or slice
            i-index of face
        j : int or slice
            j-index of face

        Returns
        -------
        nx, ny : float or ndarray
            Components of unit normal vector
        """
        mag = self.Si_mag[i, j]
        mag = np.where(mag < 1e-20, 1e-20, mag)
        return self.Si_x[i, j] / mag, self.Si_y[i, j] / mag

    def get_j_face_normal(self, i, j):
        """
        Get unit normal vector for j-face at position (i, j).

        Parameters
        ----------
        i : int or slice
            i-index of face
        j : int or slice
            j-index of face

        Returns
        -------
        nx, ny : float or ndarray
            Components of unit normal vector
        """
        mag = self.Sj_mag[i, j]
        mag = np.where(mag < 1e-20, 1e-20, mag)
        return self.Sj_x[i, j] / mag, self.Sj_y[i, j] / mag

    def get_minimum_spacing(self):
        """
        Get minimum grid spacing for CFL calculation.

        Returns
        -------
        dx_min : float
            Minimum spacing in x-direction
        dy_min : float
            Minimum spacing in y-direction
        """
        # Use face magnitudes as characteristic lengths
        # Effective spacing is volume / face_length
        dx_eff = self.volume / (0.5 * (self.Si_mag[:-1, :] + self.Si_mag[1:, :]) + 1e-20)
        dy_eff = self.volume / (0.5 * (self.Sj_mag[:, :-1] + self.Sj_mag[:, 1:]) + 1e-20)

        return np.min(dx_eff), np.min(dy_eff)

    def get_local_spacing(self):
        """
        Get local grid spacing for each cell.

        Returns
        -------
        dx : ndarray
            Effective x-spacing, shape (ni, nj)
        dy : ndarray
            Effective y-spacing, shape (ni, nj)
        """
        dx = self.volume / (0.5 * (self.Si_mag[:-1, :] + self.Si_mag[1:, :]) + 1e-20)
        dy = self.volume / (0.5 * (self.Sj_mag[:, :-1] + self.Sj_mag[:, 1:]) + 1e-20)
        return dx, dy


def compute_simple_metrics(x, y):
    """
    Simplified metric computation for Cartesian-aligned grids.

    For grids where faces are nearly aligned with coordinate axes,
    this provides a faster but less general metric computation.

    Parameters
    ----------
    x : ndarray
        Vertex x-coordinates
    y : ndarray
        Vertex y-coordinates

    Returns
    -------
    dx : ndarray
        Cell widths in x-direction
    dy : ndarray
        Cell heights in y-direction
    """
    # Average face lengths
    dx = 0.5 * (np.abs(x[1:, :-1] - x[:-1, :-1]) +
                np.abs(x[1:, 1:] - x[:-1, 1:]))
    dy = 0.5 * (np.abs(y[:-1, 1:] - y[:-1, :-1]) +
                np.abs(y[1:, 1:] - y[1:, :-1]))

    return dx, dy
