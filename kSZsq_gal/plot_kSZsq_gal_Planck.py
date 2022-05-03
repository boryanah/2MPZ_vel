import sys
sys.path.append('../pairwise')

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from utils_power import bin_mat

galaxy_sample = sys.argv[1] #"DECALS" "2MPZ" "WISExSCOS"
try:
    extra_fn = f"_{sys.argv[2]}"
except:
    extra_fn = ""
data = np.load(f"kszsq_gal_data/cl_all_{galaxy_sample}_Planck.npz")

cl_gal = data['cl_gal']
cl_kszsq = data['cl_kszsq']
cl_ksz = data['cl_ksz']
cl_ksz_gal = data['cl_ksz_gal']
cl_kszsq_gal = data['cl_kszsq_gal']
ell = data['ell']
fsky = data['fsky_mix'] # already corrected for

ell_binned = np.linspace(300, 2900, 14)
_, cl_kszsq_gal_binned = bin_mat(ell, cl_kszsq_gal, ell_binned)
_, cl_kszsq_binned = bin_mat(ell, cl_kszsq, ell_binned)
_, cl_ksz_binned = bin_mat(ell, cl_ksz, ell_binned)
_, cl_gal_binned = bin_mat(ell, cl_gal, ell_binned)
ell_binned, cl_ksz_gal_binned = bin_mat(ell, cl_ksz_gal, ell_binned)

# load lensing contribution
if galaxy_sample == 'DECALS':
    galaxy_ccl_sample = "DELS__1"
else:
    galaxy_ccl_sample = galaxy_sample
data = np.load(f"data_lensing/lensing_contribution_{galaxy_ccl_sample}.npz")
cl_contam = data['lensing_contribution']
l_contam = data['ell']
contam_binned = interpolate.interp1d(l_contam, cl_contam, bounds_error=False, fill_value=0)(ell_binned)

diff = np.diff(ell_binned)
diff = np.append(diff, diff[-1])
print(diff.shape, ell_binned.shape)

cl_kszsq_gal_err = np.sqrt(1./((2.*ell_binned+1)*fsky*diff)*(cl_kszsq_gal_binned**2+cl_gal_binned*cl_kszsq_binned))
cl_ksz_gal_err = np.sqrt(1./((2.*ell_binned+1)*fsky*diff)*(cl_ksz_gal_binned**2+cl_gal_binned*cl_ksz_binned))
factor = ell_binned*(ell_binned+1)/(2.*np.pi)

chi2 = np.sum((cl_kszsq_gal_binned/cl_kszsq_gal_err)**2.)
print("chi2, dof (kszsq_gal) = ", chi2, len(cl_kszsq_gal_binned))

fs = 20
plt.figure(1, figsize=(9, 7))
plt.axhline(0., ls='--', c='black')
plt.errorbar(ell_binned, cl_kszsq_gal_binned*factor, yerr=cl_kszsq_gal_err*factor, ls='', marker='o', capsize=4., color='dodgerblue')
#plt.errorbar(ell_binned, (cl_kszsq_gal_binned-contam_binned)*factor, yerr=cl_kszsq_gal_err*factor, ls='', marker='o', capsize=4., color='black')
plt.plot(ell_binned, (contam_binned)*factor, ls='-', color='black')
plt.xlabel(r'$\ell$', fontsize=fs)
plt.ylabel(r'$\ell (\ell + 1) C_\ell^{T^2_{\rm clean}\times \delta_g}/2 \pi \ [\mu K^2]$', fontsize=fs)
plt.savefig(f'kszsq_gal_figs/cl_kszsq_gal_{galaxy_sample}{extra_fn}.png')
#plt.show()
plt.close()

chi2 = np.sum((cl_ksz_gal_binned/cl_ksz_gal_err)[1:]**2.)
print("chi2, dof (ksz_gal) = ", chi2, len(cl_ksz_gal_binned[1:]))

plt.figure(2, figsize=(9, 7))
plt.axhline(0., ls='--', c='black')
plt.errorbar(ell_binned, cl_ksz_gal_binned*1.e5*ell_binned/np.pi, yerr=cl_ksz_gal_err*1.e5*ell_binned/np.pi, ls='', marker='o', capsize=4., color='dodgerblue')
plt.xlabel(r'$\ell$', fontsize=fs)
plt.ylabel(r'$\ell C_\ell^{T_{\rm clean}\times \delta_g}/\pi \times 10^5 \ [\mu K]$', fontsize=fs)
plt.savefig(f'kszsq_gal_figs/cl_ksz_gal_{galaxy_sample}{extra_fn}.png')
#plt.show()
plt.close()

plt.figure(3, figsize=(9, 7))
#plt.axhline(N_gg[0], ls='--', c='black')
plt.plot(ell, cl_gal)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\ell$', fontsize=fs)
plt.ylabel(r'$C_\ell^{gg}$', fontsize=fs)
plt.savefig(f'kszsq_gal_figs/cl_gal{extra_fn}.png')
#plt.show()
plt.close()




quit()
plt.figure(1, figsize=(9, 7))
plt.errorbar(ell_binned, cl_kszsq_gal_binned*factor, yerr=cl_kszsq_gal_err*factor, ls='', marker='o', capsize=4., color='dodgerblue')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell (\ell + 1) C_\ell^{T^2_{\rm clean}\times \delta_g}/2 \pi \ [\mu K^2]$')
plt.savefig(f'kszsq_gal_figs/cl_kszsq_gal_{galaxy_sample}.png')
plt.show()
