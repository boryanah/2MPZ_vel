import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pixell import powspec, utils
from utils_power import bin_mat


# load cmb power from theory
camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
cl_th = camb_theory[0, 0, :3000]
ell_th = np.arange(cl_th.size)
ell_ksz, cl_ksz = np.loadtxt("camb_data/cl_ksz_bat.dat", unpack=True)
cl_ksz /= (ell_ksz*(ell_ksz+1)/(2.*np.pi))

#plt.plot(ell_ksz, cl_ksz)
#plt.xscale('log')
#plt.show()

# load power
cl_data = np.load("camb_data/Planck_power.npy")
ell_data = np.load("camb_data/Planck_ell.npy")
#cl_data = np.load("camb_data/ACT_DR5_f090_65uK_binned_power.npy")
#ell_data = np.load("camb_data/ACT_DR5_f090_65uK_centers.npy")
#cl_data = np.load("camb_data/ACT_D56_binned_power.npy")
#ell_data = np.load("camb_data/ACT_D56_centers.npy")

# bin the power spectrum
bins = np.linspace(100, 4000, 300)
ell_data_binned, cl_data_binned = bin_mat(ell_data, cl_data, bins)

# predict the noise power spectrum
Delta_T = 47. # uK arcmin
Delta_T *= utils.arcmin
theta_FWHM = 5. # arcmin
theta_FWHM *= utils.arcmin
# noise power is Delta_T^2 b(ell)^-2
nl_th = Delta_T**2.*np.exp(theta_FWHM**2*ell_th**2/(8.*np.log(2.)))
bl_th = np.exp(-0.5*theta_FWHM**2*ell_th**2/(8.*np.log(2.)))
bl_data_binned = np.exp(-0.5*theta_FWHM**2*ell_data_binned**2/(8.*np.log(2.)))

# get the filtering function
fl_data_binned = np.interp(ell_data_binned, ell_ksz, cl_ksz)/cl_data_binned
fl_data_binned[(ell_data_binned < 100) | (ell_data_binned > 3000)] = 0.
fl_data = np.interp(ell_data, ell_ksz, cl_ksz)/cl_data
fl_data[(ell_data < 100) | (ell_data > 3000)] = 0.
fl_th = np.interp(ell_th, ell_ksz, cl_ksz)/((cl_th+nl_th)*bl_th**2)
fl_th[(ell_th < 100) | (ell_th > 3000)] = 0.
fl_data_th = np.interp(ell_th, ell_data_binned, fl_data_binned)
#np.save("camb_data/Planck_filter_kSZ.npy", fl_th)
np.save("camb_data/Planck_filter_kSZ.npy", fl_data_th)
#np.save("camb_data/Planck_ell_kSZ.npy", ell_th)
np.save("camb_data/Planck_ell_kSZ.npy", ell_th)

# for plotting
power = 2.

plt.figure(1)
plt.title("CMB power")
plt.plot(ell_data, cl_data*ell_data**power)
plt.plot(ell_data_binned, cl_data_binned*ell_data_binned**power, label="binned")
plt.plot(ell_th, cl_th*ell_th**power, lw=1, color='k')
plt.plot(ell_th, (cl_th+nl_th)*bl_th**2*ell_th**power, lw=2, color='k')
plt.plot(ell_th, nl_th*ell_th**power, lw=1, color='k')
plt.legend()
plt.xlim([0, 3000])
#plt.show()

plt.figure(2)
plt.title("filter = kSZ/Cl_data")
plt.plot(ell_data_binned, fl_data_binned*bl_data_binned/np.max(fl_data_binned*bl_data_binned), label=r'$f(\ell) = C_\ell^{kSZ} b_\ell/(C_\ell^{TT,data})$')
#plt.plot(ell_data, fl_data/np.max(fl_data))
plt.plot(ell_th, fl_th*bl_th/np.max(fl_th*bl_th), label=r'$f(\ell) = C_\ell^{kSZ} b_\ell/(C_\ell^{TT,theo}+N_\ell^{TT})$')
plt.legend()
plt.xlim([0, 3000])

plt.figure(3)
plt.title("kSZ file David sent (normed)")
plt.plot(ell_ksz, cl_ksz/np.max(cl_ksz))
plt.xlim([0, 10000])
plt.show()
