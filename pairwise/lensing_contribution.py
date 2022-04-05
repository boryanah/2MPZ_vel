"""
Script for computing lensing contribution to kSZ^2 galaxy signal
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import dblquad, quad

from pixell import powspec, utils
import sacc # 0.4.5

# load sacc file
s = sacc.Sacc.load_fits("camb_data/cls_cov_all.fits")

# tracer name
tracer_name = 'WISE' # 'WISE', 'DELS__0' (3D: 0, 0.8), 'DELS__1' (all), '2MPZ'

# load data
ell, cl_psi_gal = s.get_ell_cl('cl_00', tracer_name, 'CMBk') # galaxies x convergence
ell, cl_psi_gal = ell[::2], cl_psi_gal[::2]
ell, cl_psi_gal = ell[ell < 5000], cl_psi_gal[ell < 5000]
print("ell, kappa-gal", ell, cl_psi_gal)

# load power spectrum
camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
cl_th = camb_theory[0, 0, :10000]
ell_th = np.arange(cl_th.size)

# compute beam
theta_FWHM = 5. # arcmin
theta_FWHM *= utils.arcmin
bl_th = np.exp(-0.5*theta_FWHM**2*ell_th**2/(8.*np.log(2.)))
print("beam = ", bl_th)

# load filter
Fl_th = np.load("camb_data/Planck_filter_kSZ.npy")
print("F = ", Fl_th)

def L_prime_integrand(L_prime):
    """ also goes to zero """
    return cl_TT(L_prime)*f(L_prime)*L_prime**2

def phi_integrand(phi, L_prime, ell):
    argument = np.sqrt(L_prime**2 + ell**2 + 2*L_prime*ell*np.cos(phi))
    return f(argument)*np.cos(phi)

def total_integrand(phi, L_prime, ell):
    #def total_integrand(L_prime, phi, ell):
    print(L_prime, phi, ell)
    return L_prime_integrand(L_prime) * phi_integrand(phi, L_prime, ell)

# lensed primary CMB power spectrum
cl_TT = interpolate.interp1d(ell_th, cl_th, bounds_error=False, fill_value=0.)
b = interpolate.interp1d(ell_th, bl_th, bounds_error=False, fill_value=0.)
fl_th = Fl_th*bl_th
fl_th[bl_th ==  0.] = 0.
fl_th /= np.max(fl_th)
f = interpolate.interp1d(ell_th, fl_th, bounds_error=False, fill_value=0.)

if not os.path.exists("camb_data/lensing_contribution_integral.npy"):
    # compute integral
    #L_primes = np.linspace(0., 3000., 100)
    L_primes = np.arange(3000)
    integral = np.zeros(len(ell))
    for i in range(len(ell)):
        vals = np.zeros(len(L_primes))
        for j in range(len(L_primes)):
            res, err = quad(phi_integrand, 0., 2.*np.pi, args=(L_primes[j], ell[i]))
            vals[j] = res
        phi_integral = interpolate.interp1d(L_primes, vals, bounds_error=False, fill_value=0.)
        def final_integrand(L_prime):
            return L_prime_integrand(L_prime) * phi_integral(L_prime)
        res, err = quad(final_integrand, 0., np.inf)
        print(res, err, i, len(ell))
        integral[i] = res
    """
    integral = np.zeros(len(ell))
    for i in range(len(ell)):
        res, err = dblquad(total_integrand, 0., 2.*np.pi, 0., np.inf, args=(ell[i],))
        print(res, err)
        integral[i] = res
    print(integral)
    """
    np.save("camb_data/lensing_contribution_integral.npy", integral)
else:
    integral = np.load("camb_data/lensing_contribution_integral.npy")

# compute lensing contamination
lens_contam = -2. * ell * cl_psi_gal/(2.*np.pi)**2 * integral
# assuming David has shared convergence rather than psi (kappa=-1/2 nabla^2phi)
lens_contam /= (0.5*ell*(ell+1))
np.savez(f"camb_data/lensing_contribution_{tracer_name}", lensing_contribution=lens_contam, ell=ell)

plt.figure(1)
plt.plot(ell, np.zeros_like(ell), color='black', ls='--')
plt.plot(ell, lens_contam*ell**2)
plt.xlim([0, 3000])

plt.figure(2)
plt.plot(ell, cl_psi_gal)
plt.yscale('log')
plt.show()


#s.tracers
#{'WISE': <sacc.tracers.MapTracer object at 0x7f713ddcec90>, 'CMBk': <sacc.tracers.MapTracer object at 0x7f713ddba050>, 'DELS__0': <sacc.tracers.NZTracer object at 0x7f713e566050>, 'DELS__1': <sacc.tracers.NZTracer object at 0x7f713ddd3090>, '2MPZ': <sacc.tracers.NZTracer object at 0x7f713ddd3310>}
