"""
Script for computing lensing contribution to kSZ^2 galaxy signal
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d

# cmb power
l, dltt, dlee, dlte = np.loadtxt("../pairwise/camb_data/camb_19834798_scalcls.dat", unpack=True) # unlensed CMB
cltt = interp1d(l, dltt*2*np.pi/(l*(l+1)), bounds_error=False, fill_value=0)

# beam
ell_th = np.arange(10000)
theta_FWHM = np.radians(5./60)
theta_sigma = theta_FWHM/2.355
bl_th = np.exp(-0.5*theta_sigma**2*ell_th**2)
bl = interp1d(ell_th, bl_th, bounds_error=False, fill_value=0)

# filter
Fl_th = np.load("kszsq_gal_data/Planck_filter_kSZ.npy")
ell = np.load("kszsq_gal_data/Planck_ell_kSZ.npy")
Fl_th = Fl_th / np.amax(Fl_th)
Fl = interp1d(ell, Fl_th, bounds_error=False, fill_value=0)

# tracer name
tracer_name = 'WISE' #'DELS__1' # 'WISE', 'DELS__0' (3D: 0, 0.8), 'DELS__1' (all), '2MPZ'

def fl(l):
    return Fl(l)*bl(l)

# plot filter
#plt.plot(ell, fl(ell))
#plt.show()

# inner integral
def integ_inner(l, L, nphi=64):
    phi = np.linspace(0, 2*np.pi, nphi)
    cphi = np.cos(phi)
    l2 = l**2+L**2+2*l*L*cphi
    l2[l2 <= 0] = 0
    ll = np.sqrt(l2)
    fll = fl(ll)
    return simps(fll*cphi, x=phi)

# outer integral
def get_integ(nll, nphi=64, verbose=False):
    ls = np.linspace(0, 10000, nll)

    # Inner integral
    int1 = np.zeros([nll, nll])
    for i in range(nll):
        if (i % 10 == 0) and verbose:
            print(i)
        for j in range(i, nll):
            ii = integ_inner(ls[i], ls[j], nphi=nphi)
            int1[i, j] = ii
            int1[j, i] = ii
    
    # Outer integral
    integ = simps((ls**2*fl(ls)*cltt(ls))[None, :]*int1, x=ls, axis=-1)
    return ls, integ

if not os.path.exists("data_lensing/lensing_contribution_integral.npy"):
    # compute outer integral (different resolution choices)
    ls1, I1 = get_integ(256)
    #ls2, I2 = get_integ(512)
    #ls3, I3 = get_integ(1024, nphi=32)
    np.save("data_lensing/lensing_contribution_integral.npy", I1)
else:
    I1 = np.load("data_lensing/lensing_contribution_integral.npy")

# TESTING!!!!!!!!!!!!!!!!! (save ls1 elsewhere)
ls1 = np.linspace(0, 10000, 256)
#plt.plot(ls1, I1, 'k-')
#plt.plot(ls2, I2, 'r--')
#plt.plot(ls3, I3, 'y:')
#plt.show()

# read cross-correlation power spectrum and fit a polynomial to it (in log-log)
import sacc

# read file
s = sacc.Sacc.load_fits('data_lensing/cls_cov_all.fits')
l_s, cl_s, cov_s = s.get_ell_cl('cl_00', tracer_name, 'CMBk', return_cov=True)

# TESTING!!!!!!!!!!!!!!!!!!!!!!
#data = np.load(f"kszsq_gal_data/cl_kappa_gal_{tracer_name}.npz")
#cl_s = data['cl_kappa_gal_binned']
#l_s = data['ell_binned']

err_s = np.sqrt(np.diag(cov_s))
err_s[0] = err_s[1]
coeffs = np.polyfit(np.log(l_s[:-2]), np.log(cl_s[:-2]), w=1/err_s[:-2], deg=6) # og
#coeffs = np.polyfit(np.log(l_s[0:-5]), np.log(cl_s[0:-5]), w=1/err_s[0:-5], deg=3) # TESTING

def cl_s_fit(ls):
    poly = np.poly1d(coeffs)
    goodl = ls > 0
    cl_out = np.zeros(len(ls))
    cl_out[goodl] = np.exp(poly(np.log(ls[goodl])))
    return cl_out

plt.errorbar(l_s, cl_s, yerr=np.sqrt(np.diag(cov_s)))
plt.plot(l_s, cl_s_fit(l_s), 'k-')
plt.loglog()
#plt.xlim([0, 2000])
#plt.ylim([0, 2E-7])
plt.show()

# construct lensing template
Dcl1 = -2*cl_s_fit(ls1)*I1/((2*np.pi)**2*0.5*(ls1+1))
#Dcl2 = -2*cl_s_fit(ls2)*I2/((2*np.pi)*0.5*(ls2+1))
#Dcl3 = -2*cl_s_fit(ls3)*I3/((2*np.pi)*0.5*(ls3+1))

np.savez(f"data_lensing/lensing_contribution_{tracer_name}", lensing_contribution=Dcl1, ell=ls1)

# plot
plt.plot(ls1, ls1*(ls1+1)*Dcl1/(2*np.pi), 'k-')
#plt.plot(ls2, ls2*(ls2+1)*Dcl2/(2*np.pi), 'r--')
#plt.plot(ls3, ls3*(ls3+1)*Dcl3/(2*np.pi), 'y:')
plt.show()

plt.plot(ls1, ls1*(ls1+1)*Dcl1/(2*np.pi), 'k-')
#plt.plot(ls2, ls2*(ls2+1)*Dcl2/(2*np.pi), 'r--')
#plt.plot(ls3, ls3*(ls3+1)*Dcl3/(2*np.pi), 'y:')
plt.xlim([400, 2700])
plt.show()
