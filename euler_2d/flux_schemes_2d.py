"""
2D Flux Schemes for Euler Equations

Implements HLLC and Rusanov fluxes with rotation for arbitrary face orientations.
Uses dimensional splitting approach - rotates to face-normal frame, applies 1D flux,
then rotates back.

The 2D Euler equations use conservative variables U = [rho, rho*u, rho*v, rho*E]^T
"""

import numpy as np


def compute_flux_x(rho, u, v, p, E, gamma):
    """
    Compute the x-direction physical flux F(U).

    Parameters
    ----------
    rho : ndarray
        Density
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    p : ndarray
        Pressure
    E : ndarray
        Total energy per unit volume
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray
        Flux array of shape (4, ...) containing [rho*u, rho*u^2+p, rho*u*v, u*(E+p)]
    """
    rho_u = rho * u
    F = np.array([
        rho_u,
        rho_u * u + p,
        rho_u * v,
        u * (E + p)
    ])
    return F


def compute_flux_y(rho, u, v, p, E, gamma):
    """
    Compute the y-direction physical flux G(U).

    Parameters
    ----------
    rho : ndarray
        Density
    u : ndarray
        x-velocity component
    v : ndarray
        y-velocity component
    p : ndarray
        Pressure
    E : ndarray
        Total energy per unit volume
    gamma : float
        Ratio of specific heats

    Returns
    -------
    G : ndarray
        Flux array of shape (4, ...) containing [rho*v, rho*u*v, rho*v^2+p, v*(E+p)]
    """
    rho_v = rho * v
    G = np.array([
        rho_v,
        rho_v * u,
        rho_v * v + p,
        v * (E + p)
    ])
    return G


def compute_sound_speed(p, rho, gamma):
    """
    Compute the speed of sound.

    Parameters
    ----------
    p : ndarray
        Pressure
    rho : ndarray
        Density
    gamma : float
        Ratio of specific heats

    Returns
    -------
    c : ndarray
        Speed of sound
    """
    return np.sqrt(gamma * np.maximum(p, 1e-10) / np.maximum(rho, 1e-10))


def rotate_to_normal(U, nx, ny):
    """
    Rotate velocity components to face-normal frame.

    Parameters
    ----------
    U : ndarray
        Conservative variables (4, ...)
    nx, ny : float or ndarray
        Face normal components (should be unit vector)

    Returns
    -------
    U_rot : ndarray
        Rotated conservative variables with u_n = u*nx + v*ny, u_t = -u*ny + v*nx
    """
    U_rot = U.copy()
    rho = U[0]
    rho_u = U[1]
    rho_v = U[2]

    # Rotate momentum to face-normal frame
    # u_n = u*nx + v*ny (normal velocity)
    # u_t = -u*ny + v*nx (tangential velocity)
    U_rot[1] = rho_u * nx + rho_v * ny   # rho * u_n
    U_rot[2] = -rho_u * ny + rho_v * nx  # rho * u_t

    return U_rot


def rotate_from_normal(F, nx, ny):
    """
    Rotate flux components from face-normal frame back to physical frame.

    Parameters
    ----------
    F : ndarray
        Flux in face-normal frame (4, ...)
    nx, ny : float or ndarray
        Face normal components

    Returns
    -------
    F_phys : ndarray
        Flux in physical frame
    """
    F_phys = F.copy()
    F_n = F[1]
    F_t = F[2]

    # Rotate back: inverse rotation
    # F_x = F_n*nx - F_t*ny
    # F_y = F_n*ny + F_t*nx
    F_phys[1] = F_n * nx - F_t * ny
    F_phys[2] = F_n * ny + F_t * nx

    return F_phys


def hllc_flux_1d(U_L, U_R, gamma):
    """
    1D HLLC flux for Euler equations.

    This is the core 1D solver used after rotation to face-normal frame.

    Parameters
    ----------
    U_L : ndarray
        Left state conservative variables (4, ...)
    U_R : ndarray
        Right state conservative variables (4, ...)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray
        HLLC numerical flux (4, ...)
    """
    # Extract primitives from left state
    rho_L = np.maximum(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    v_L = U_L[2] / rho_L
    E_L = U_L[3]
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))
    p_L = np.maximum(p_L, 1e-10)
    c_L = compute_sound_speed(p_L, rho_L, gamma)

    # Extract primitives from right state
    rho_R = np.maximum(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    v_R = U_R[2] / rho_R
    E_R = U_R[3]
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))
    p_R = np.maximum(p_R, 1e-10)
    c_R = compute_sound_speed(p_R, rho_R, gamma)

    # Roe averages for wave speed estimates
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    v_roe = (sqrt_rho_L * v_L + sqrt_rho_R * v_R) / denom
    H_L = (E_L + p_L) / rho_L
    H_R = (E_R + p_R) / rho_R
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_roe = np.sqrt(np.maximum((gamma - 1) * (H_roe - 0.5 * (u_roe**2 + v_roe**2)), 1e-10))

    # Wave speed estimates (Davis estimates)
    S_L = np.minimum(u_L - c_L, u_roe - c_roe)
    S_R = np.maximum(u_R + c_R, u_roe + c_roe)

    # Contact wave speed
    denom_star = rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    denom_star = np.where(np.abs(denom_star) < 1e-10, 1e-10, denom_star)
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / denom_star

    # Compute physical fluxes
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        rho_L * u_L * v_L,
        u_L * (E_L + p_L)
    ])

    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        rho_R * u_R * v_R,
        u_R * (E_R + p_R)
    ])

    # Compute star states
    # Left star state
    coeff_L = rho_L * (S_L - u_L) / np.where(np.abs(S_L - S_star) < 1e-10, 1e-10, S_L - S_star)
    p_star_L = p_L + rho_L * (S_L - u_L) * (S_star - u_L)

    U_star_L = np.array([
        coeff_L,
        coeff_L * S_star,
        coeff_L * v_L,
        coeff_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))
    ])

    # Handle divide by zero in energy term
    factor_L = np.where(np.abs(S_L - u_L) < 1e-10, 0.0, p_L / (rho_L * (S_L - u_L)))
    U_star_L[3] = coeff_L * (E_L / rho_L + (S_star - u_L) * (S_star + factor_L))

    # Right star state
    coeff_R = rho_R * (S_R - u_R) / np.where(np.abs(S_R - S_star) < 1e-10, 1e-10, S_R - S_star)

    U_star_R = np.array([
        coeff_R,
        coeff_R * S_star,
        coeff_R * v_R,
        coeff_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))
    ])

    factor_R = np.where(np.abs(S_R - u_R) < 1e-10, 0.0, p_R / (rho_R * (S_R - u_R)))
    U_star_R[3] = coeff_R * (E_R / rho_R + (S_star - u_R) * (S_star + factor_R))

    # Select appropriate flux based on wave configuration
    # Case 1: S_L >= 0 -> F = F_L
    # Case 2: S_L < 0 <= S_star -> F = F_L + S_L*(U_star_L - U_L)
    # Case 3: S_star < 0 < S_R -> F = F_R + S_R*(U_star_R - U_R)
    # Case 4: S_R <= 0 -> F = F_R

    F = np.zeros_like(F_L)

    # Vectorized flux selection
    cond1 = S_L >= 0
    cond2 = (S_L < 0) & (S_star >= 0)
    cond3 = (S_star < 0) & (S_R > 0)
    cond4 = S_R <= 0

    for i in range(4):
        F[i] = np.where(cond1, F_L[i],
               np.where(cond2, F_L[i] + S_L * (U_star_L[i] - U_L[i]),
               np.where(cond3, F_R[i] + S_R * (U_star_R[i] - U_R[i]),
                        F_R[i])))

    return F


def rusanov_flux_1d(U_L, U_R, gamma):
    """
    1D Rusanov (Local Lax-Friedrichs) flux.

    Parameters
    ----------
    U_L : ndarray
        Left state conservative variables (4, ...)
    U_R : ndarray
        Right state conservative variables (4, ...)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray
        Rusanov numerical flux (4, ...)
    """
    # Extract primitives from left state
    rho_L = np.maximum(U_L[0], 1e-10)
    u_L = U_L[1] / rho_L
    v_L = U_L[2] / rho_L
    E_L = U_L[3]
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * (u_L**2 + v_L**2))
    p_L = np.maximum(p_L, 1e-10)
    c_L = compute_sound_speed(p_L, rho_L, gamma)

    # Extract primitives from right state
    rho_R = np.maximum(U_R[0], 1e-10)
    u_R = U_R[1] / rho_R
    v_R = U_R[2] / rho_R
    E_R = U_R[3]
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * (u_R**2 + v_R**2))
    p_R = np.maximum(p_R, 1e-10)
    c_R = compute_sound_speed(p_R, rho_R, gamma)

    # Physical fluxes
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 + p_L,
        rho_L * u_L * v_L,
        u_L * (E_L + p_L)
    ])

    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 + p_R,
        rho_R * u_R * v_R,
        u_R * (E_R + p_R)
    ])

    # Maximum wave speed
    alpha = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)

    # Rusanov flux
    F = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)

    return F


def hllc_flux_2d(U_L, U_R, nx, ny, gamma):
    """
    2D HLLC flux for arbitrary face orientation.

    Uses rotation to face-normal frame, applies 1D HLLC, rotates back.

    Parameters
    ----------
    U_L : ndarray
        Left state conservative variables (4, ...)
    U_R : ndarray
        Right state conservative variables (4, ...)
    nx, ny : float or ndarray
        Face normal components (unit vector pointing from L to R)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray
        Numerical flux in physical frame (4, ...)
    """
    # Rotate to face-normal frame
    U_L_rot = rotate_to_normal(U_L, nx, ny)
    U_R_rot = rotate_to_normal(U_R, nx, ny)

    # Apply 1D HLLC in rotated frame
    F_rot = hllc_flux_1d(U_L_rot, U_R_rot, gamma)

    # Rotate flux back to physical frame
    F = rotate_from_normal(F_rot, nx, ny)

    return F


def rusanov_flux_2d(U_L, U_R, nx, ny, gamma):
    """
    2D Rusanov flux for arbitrary face orientation.

    Parameters
    ----------
    U_L : ndarray
        Left state conservative variables (4, ...)
    U_R : ndarray
        Right state conservative variables (4, ...)
    nx, ny : float or ndarray
        Face normal components (unit vector pointing from L to R)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray
        Numerical flux in physical frame (4, ...)
    """
    # Rotate to face-normal frame
    U_L_rot = rotate_to_normal(U_L, nx, ny)
    U_R_rot = rotate_to_normal(U_R, nx, ny)

    # Apply 1D Rusanov in rotated frame
    F_rot = rusanov_flux_1d(U_L_rot, U_R_rot, gamma)

    # Rotate flux back to physical frame
    F = rotate_from_normal(F_rot, nx, ny)

    return F
