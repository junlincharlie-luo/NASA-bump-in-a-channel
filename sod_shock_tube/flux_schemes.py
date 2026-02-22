"""
Numerical flux schemes for the 1D Euler equations.

Implements:
- Rusanov (Local Lax-Friedrichs) flux
- HLLC flux
"""

import numpy as np


def compute_flux(rho, u, p, E, gamma):
    """
    Compute the physical flux F(U) for the Euler equations.

    Parameters
    ----------
    rho : ndarray
        Density
    u : ndarray
        Velocity
    p : ndarray
        Pressure
    E : ndarray
        Total energy per unit volume (rho * e_total)
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray of shape (3, n)
        Flux vector [F_rho, F_rhou, F_rhoE]
    """
    F = np.zeros((3, len(rho)))
    F[0] = rho * u                    # Mass flux
    F[1] = rho * u**2 + p             # Momentum flux
    F[2] = u * (E + p)                # Energy flux
    return F


def compute_sound_speed(p, rho, gamma):
    """Compute sound speed c = sqrt(gamma * p / rho)."""
    return np.sqrt(gamma * p / rho)


def rusanov_flux(U_L, U_R, gamma):
    """
    Rusanov (Local Lax-Friedrichs) flux.

    F = 0.5 * [F(U_L) + F(U_R)] - 0.5 * alpha * (U_R - U_L)
    alpha = max(|u| + c)

    Simple and robust, but more diffusive than HLLC.

    Parameters
    ----------
    U_L : ndarray of shape (3, n)
        Left states [rho, rho*u, rho*E]
    U_R : ndarray of shape (3, n)
        Right states [rho, rho*u, rho*E]
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray of shape (3, n)
        Numerical flux at each interface
    """
    # Extract primitive variables - Left
    rho_L = U_L[0]
    u_L = U_L[1] / rho_L
    E_L = U_L[2]
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    c_L = compute_sound_speed(p_L, rho_L, gamma)

    # Extract primitive variables - Right
    rho_R = U_R[0]
    u_R = U_R[1] / rho_R
    E_R = U_R[2]
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
    c_R = compute_sound_speed(p_R, rho_R, gamma)

    # Compute physical fluxes
    F_L = compute_flux(rho_L, u_L, p_L, E_L, gamma)
    F_R = compute_flux(rho_R, u_R, p_R, E_R, gamma)

    # Maximum wave speed
    alpha = np.maximum(np.abs(u_L) + c_L, np.abs(u_R) + c_R)

    # Rusanov flux
    F = 0.5 * (F_L + F_R) - 0.5 * alpha * (U_R - U_L)

    return F


def hllc_flux(U_L, U_R, gamma):
    """
    HLLC (Harten-Lax-van Leer-Contact) flux.

    Three-wave approximate Riemann solver that accurately captures
    contact discontinuities.

    Parameters
    ----------
    U_L : ndarray of shape (3, n)
        Left states [rho, rho*u, rho*E]
    U_R : ndarray of shape (3, n)
        Right states [rho, rho*u, rho*E]
    gamma : float
        Ratio of specific heats

    Returns
    -------
    F : ndarray of shape (3, n)
        Numerical flux at each interface
    """
    n_faces = U_L.shape[1]

    # Extract primitive variables - Left
    rho_L = U_L[0]
    u_L = U_L[1] / rho_L
    E_L = U_L[2]
    p_L = (gamma - 1) * (E_L - 0.5 * rho_L * u_L**2)
    p_L = np.maximum(p_L, 1e-10)  # Ensure positive pressure
    c_L = compute_sound_speed(p_L, rho_L, gamma)
    H_L = (E_L + p_L) / rho_L  # Total enthalpy

    # Extract primitive variables - Right
    rho_R = U_R[0]
    u_R = U_R[1] / rho_R
    E_R = U_R[2]
    p_R = (gamma - 1) * (E_R - 0.5 * rho_R * u_R**2)
    p_R = np.maximum(p_R, 1e-10)
    c_R = compute_sound_speed(p_R, rho_R, gamma)
    H_R = (E_R + p_R) / rho_R

    # Roe averages for wave speed estimates
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    denom = sqrt_rho_L + sqrt_rho_R

    u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / denom
    H_roe = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / denom
    c_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe**2))

    # Wave speed estimates (Davis estimates)
    S_L = np.minimum(u_L - c_L, u_roe - c_roe)
    S_R = np.maximum(u_R + c_R, u_roe + c_roe)

    # Contact wave speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R) + 1e-20)

    # Physical fluxes
    F_L = compute_flux(rho_L, u_L, p_L, E_L, gamma)
    F_R = compute_flux(rho_R, u_R, p_R, E_R, gamma)

    # HLLC intermediate states
    def compute_U_star(rho, u, E, p, S, S_star):
        """Compute the HLLC intermediate state U*."""
        coeff = rho * (S - u) / (S - S_star + 1e-20)
        U_star = np.zeros((3, len(rho)))
        U_star[0] = coeff
        U_star[1] = coeff * S_star
        U_star[2] = coeff * (E / rho + (S_star - u) * (S_star + p / (rho * (S - u) + 1e-20)))
        return U_star

    U_star_L = compute_U_star(rho_L, u_L, E_L, p_L, S_L, S_star)
    U_star_R = compute_U_star(rho_R, u_R, E_R, p_R, S_R, S_star)

    # Select flux based on wave configuration
    F = np.zeros((3, n_faces))

    for i in range(n_faces):
        if S_L[i] >= 0:
            # Supersonic flow to the right
            F[:, i] = F_L[:, i]
        elif S_L[i] < 0 <= S_star[i]:
            # Left star region
            F[:, i] = F_L[:, i] + S_L[i] * (U_star_L[:, i] - U_L[:, i])
        elif S_star[i] < 0 <= S_R[i]:
            # Right star region
            F[:, i] = F_R[:, i] + S_R[i] * (U_star_R[:, i] - U_R[:, i])
        else:
            # Supersonic flow to the left
            F[:, i] = F_R[:, i]

    return F
