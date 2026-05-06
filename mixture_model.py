"""Mass-loss prescriptions for the RLOF mixture model.

All dimensional inputs are expected in cgs units:
    masses      [g]
    lengths     [cm]
    temperature [K]
    density     [g cm^-3]

The model returns mass-loss rates per unit surface density by default.
Multiply by ``rho_surface`` to obtain an absolute mass-loss rate in g/s.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import root_scalar


G_CGS = 6.67430e-8
K_B_CGS = 1.380649e-16
M_P_CGS = 1.67262192369e-24

DEFAULT_BETA = np.array([-7.00187011, 6.12688998, -4.02644083])
K_SPH = 2.47


def collinear_eq(x, Mstar, Mp, a, G=G_CGS):
    """Force balance along the star-planet axis in the rotating frame."""
    x_cm = Mp * a / (Mstar + Mp)
    Omega2 = G * (Mstar + Mp) / a**3

    term_star = -G * Mstar * x / np.abs(x) ** 3
    term_planet = -G * Mp * (x - a) / np.abs(x - a) ** 3
    term_rot = Omega2 * (x - x_cm)

    return term_star + term_planet + term_rot


def find_L1_L2(Mstar, Mp, a, G=G_CGS):
    """Find L1 and L2 positions for star at x=0 and planet at x=a."""
    eps = 1e-8 * a

    def safe_brentq(bracket):
        x1, x2 = bracket
        try:
            f1 = collinear_eq(x1, Mstar, Mp, a, G)
            f2 = collinear_eq(x2, Mstar, Mp, a, G)
        except Exception:
            return np.nan

        if not np.isfinite(f1) or not np.isfinite(f2):
            return np.nan
        if f1 == 0:
            return x1
        if f2 == 0:
            return x2
        if f1 * f2 > 0:
            return np.nan

        try:
            sol = root_scalar(
                collinear_eq,
                args=(Mstar, Mp, a, G),
                bracket=[x1, x2],
                method="brentq",
            )
            return sol.root if sol.converged else np.nan
        except Exception:
            return np.nan

    xL1 = safe_brentq([eps, a - eps])
    xL2 = safe_brentq([a + eps, 3.0 * a])
    return xL1, xL2


def sound_speed_squared(T, mu, gamma):
    """Sound speed squared [cm^2/s^2].

    Use gamma=1 for an isothermal sound speed. Use gamma=5/3 for an
    adiabatic monatomic gas.
    """
    return gamma * K_B_CGS * T / (mu * M_P_CGS)


def hill_radius(Mp, Mstar, a):
    """Planet Hill radius [cm]."""
    return (Mp / (3 * Mstar)) ** (1.0 / 3.0) * a


def lambda_p(T, mu, Mp, Rp, gamma, G=G_CGS):
    """Planet escape parameter, G Mp / (Rp cs^2)."""
    cs2 = sound_speed_squared(T, mu, gamma)
    return G * Mp / (Rp * cs2)


def mass_loss_sph_per_rho(Mp, Rp, T, mu, gamma, G=G_CGS):
    """Parker-wind and modified spherical rates per rho_surface [cm^3/s]."""
    lamp = lambda_p(T, mu, Mp, Rp, gamma, G)
    gfac = np.exp(-K_SPH / lamp**2)
    mdot_pw = np.pi * (G * Mp * Rp**3 * lamp**3) ** 0.5 * np.exp(1.5 - lamp)
    return mdot_pw, gfac * mdot_pw


def phieff(x, a, Mstar, Mp, G=G_CGS):
    """Effective potential along the star-planet axis [erg/g]."""
    x_cm = a * Mp / (Mstar + Mp)
    omega2 = G * (Mstar + Mp) / a**3

    term_star = -G * Mstar / np.abs(x)
    term_planet = -G * Mp / np.abs(x - a)
    rot_term = -0.5 * omega2 * (x - x_cm) ** 2

    return term_star + term_planet + rot_term


def mdot_anisotropic_per_rho(Mstar, Mp, Rp, a, T, mu, gamma, G=G_CGS):
    """Nozzle and modified two-tail rates per rho_surface [cm^3/s]."""
    xL1, xL2 = find_L1_L2(Mstar, Mp, a, G)
    if not np.isfinite(xL1) or not np.isfinite(xL2):
        return np.nan, np.nan

    phi_yyL1 = (
        G * Mstar / np.abs(xL1) ** 3
        + G * Mp / np.abs(xL1 - a) ** 3
        - G * (Mstar + Mp) / a**3
    )
    phi_zzL1 = G * Mstar / np.abs(xL1) ** 3 + G * Mp / np.abs(xL1 - a) ** 3

    phi_yyL2 = (
        G * Mstar / np.abs(xL2) ** 3
        + G * Mp / np.abs(xL2 - a) ** 3
        - G * (Mstar + Mp) / a**3
    )
    phi_zzL2 = G * Mstar / np.abs(xL2) ** 3 + G * Mp / np.abs(xL2 - a) ** 3

    cs2 = sound_speed_squared(T, mu, gamma)
    phiL1 = phieff(xL1, a, Mstar, Mp, G)
    phiL2 = phieff(xL2, a, Mstar, Mp, G)
    phi_s1 = phieff(a - Rp, a, Mstar, Mp, G)
    phi_s2 = phieff(a + Rp, a, Mstar, Mp, G)

    mdot_L1 = (
        (2 * np.pi * cs2 ** 1.5) / np.sqrt(phi_yyL1 * phi_zzL1)
        * np.exp(((phi_s1 - phiL1) / cs2) - 0.5)
    )
    mdot_L2 = (
        (2 * np.pi * cs2 ** 1.5) / np.sqrt(phi_yyL2 * phi_zzL2)
        * np.exp(((phi_s2 - phiL2) / cs2) - 0.5)
    )

    mdot_nozzle = mdot_L1 + mdot_L2
    chi2 = (G * (Mstar + Mp) / a**3) * (xL2 - a) ** 2 / cs2
    mdot_tail = mdot_L1 + np.exp(-0.75 * chi2) * mdot_L2

    return mdot_nozzle, mdot_tail


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def mdot_mixture_per_rho(Mstar, Mp, Rp, a, T, mu, gamma, beta=DEFAULT_BETA, G=G_CGS):
    """Mixture-model mass-loss rate per rho_surface [cm^3/s]."""
    xL1, _ = find_L1_L2(Mstar, Mp, a, G)
    if not np.isfinite(xL1):
        return np.nan

    phi_s1 = phieff(a - Rp, a, Mstar, Mp, G)
    phiL1 = phieff(xL1, a, Mstar, Mp, G)
    cs2 = sound_speed_squared(T, mu, gamma)
    lamp = lambda_p(T, mu, Mp, Rp, gamma, G)

    eta = -(phi_s1 - phiL1) / cs2
    z = beta[0] + beta[1] * np.log(lamp) + beta[2] * np.log1p(eta)
    w = sigmoid(z)

    _, mdot_sph = mass_loss_sph_per_rho(Mp, Rp, T, mu, gamma, G)
    _, mdot_tail = mdot_anisotropic_per_rho(Mstar, Mp, Rp, a, T, mu, gamma, G)

    return mdot_sph ** (1 - w) * mdot_tail**w


def mass_loss_rates(Mstar, Mp, Rp, a, T, mu, gamma, rho_surface=None, G=G_CGS):
    """Compute all mass-loss prescriptions.

    If rho_surface is None, values are returned per unit surface density
    with units cm^3/s. If rho_surface is supplied in g/cm^3, values are
    returned in g/s.
    """
    mdot_pw, mdot_sph = mass_loss_sph_per_rho(Mp, Rp, T, mu, gamma, G)
    mdot_nozzle, mdot_tail = mdot_anisotropic_per_rho(Mstar, Mp, Rp, a, T, mu, gamma, G)
    mdot_mix = mdot_mixture_per_rho(Mstar, Mp, Rp, a, T, mu, gamma, DEFAULT_BETA, G)

    rates = {
        "parker_wind": mdot_pw,
        "modified_spherical": mdot_sph,
        "nozzle": mdot_nozzle,
        "modified_two_tail": mdot_tail,
        "mixture": mdot_mix,
    }

    if rho_surface is not None:
        rates = {name: rho_surface * value for name, value in rates.items()}

    return rates


def rho_from_pressure(P_surface, T, mu, gamma):
    """Estimate surface density from pressure.
    P_surface should be in dyn/cm^2. The returned density is in g/cm^3.
    For isothermal sound speed, gamma should be 1.
    """
    cs2 = sound_speed_squared(T, mu, gamma)
    return gamma * P_surface / cs2
