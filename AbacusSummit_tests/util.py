import numpy as np
from scipy.integrate import quad

c = 299792.458 # km/s

def integrate_one_over_Ez(z_max, Omega_m, Omega_Lambda):

    def integrand(z):
        return 1./Ez(z, Omega_m, Omega_Lambda)

    integ, _ = quad(integrand, 0., z_max)
    return integ

def Ez(z, Omega_m, Omega_Lambda):
    """
    For low redshifts only
    """
    return np.sqrt(Omega_m*(1+z)**3 + Omega_Lambda)

def comoving_distance(z_max, Omega_m, Omega_Lambda):
    d = integrate_one_over_Ez(z_max, Omega_m, Omega_Lambda)

    Omega_k = 1. - Omega_m - Omega_Lambda
    if Omega_k != 0.:
        if Omega_k > 0.0:
            sqrt_Ok = np.sqrt(Omega_k)
            d = np.sinh(sqrt_Ok * d) / sqrt_Ok
        else:
            sqrt_Ok = np.sqrt(-Omega_k)
            d = np.sin(sqrt_Ok * d) / sqrt_Ok

    d *= 1.e-2 # because H_0 = 100*h
    d *= c # Mpc/h
    return d

def luminosity_distance(z_max, Omega_m, Omega_Lambda):
    # strictly increasing
    d = comoving_distance(z_max, Omega_m, Omega_Lambda)
    d *= (1+z_max)
    return d
