
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.integrate import dblquad

from pixell import powspec, utils
import sacc # 0.4.5

# load sacc file
s = sacc.Sacc.load_fits("camb_data/cls_cov_all.fits")

# load data
ell, cl_psi_gal = s.get_ell_cl('cl_00', 'WISE', 'CMBk')
print("ell, psi-gal", ell, cl_psi_gal)

# load power spectrum
camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
cl_th = camb_theory[0, 0, :10000]
ell_th = np.arange(cl_th.size)

# compute beam
theta_FWHM = 5. # arcmin
theta_FWHM *= utils.arcmin
bl_th = np.exp(-0.5*theta_FWHM**2*ell_th**2/(8.*np.log(2.)))
print("beam = ", bl_th)
bl_th[0] = 0.

# load filter
fl_th = np.load("camb_data/Planck_filter_kSZ.npy")
print("F = ", fl_th)

def f(ell):
    """ have checked this goes to zero for large ell """
    return F(ell)/b(ell)

def L_prime_integrand(L_prime):
    """ also goes to zero """
    return cl_TT(L_prime)*f(L_prime)*L_prime**2

def phi_integrand(phi, L_prime, ell):
    argument = np.sqrt(L_prime**2 + ell**2 + 2*L_prime*ell*np.cos(phi))
    return f(argument)*np.cos(phi)

def total_integrand(phi, L_prime, ell):
    return L_prime_integrand(L_prime) * phi_integrand(phi, L_prime, ell)

# lensed primary CMB power spectrum
cl_TT = interpolate.interp1d(ell_th, cl_th, bounds_error=False, fill_value=0.)
b = interpolate.interp1d(ell_th, bl_th, bounds_error=False, fill_value=0.)
F = interpolate.interp1d(ell_th, fl_th, bounds_error=False, fill_value=0.)

# compute integral
integral = np.zeros(len(ell))
for i in range(len(ell)):
    res, err = dblquad(total_integrand, 0., 2.*np.pi, 0., np.inf, args=(ell[i],))
    print(res, err)
    integral[i] = res
print(integral)
    
# compute lensing contamination
lens_contam = -2. * ell * cl_psi_gal/(2.*np.pi)**2 * integral

plt.figure(1)
plt.plot(ell, lens_contam*ell**2)
plt.xlim([0, 3000])
plt.show()

#s.tracers
#{'WISE': <sacc.tracers.MapTracer object at 0x7f713ddcec90>, 'CMBk': <sacc.tracers.MapTracer object at 0x7f713ddba050>, 'DELS__0': <sacc.tracers.NZTracer object at 0x7f713e566050>, 'DELS__1': <sacc.tracers.NZTracer object at 0x7f713ddd3090>, '2MPZ': <sacc.tracers.NZTracer object at 0x7f713ddd3310>}
