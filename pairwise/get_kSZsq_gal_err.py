import sys

import numpy as np
import matplotlib.pyplot as plt

from utils_power import bin_mat

gal_sample = sys.argv[1] #"DECALS" "2MPZ" "WISExSCOS"


data = np.load(f"kszsq_gal_data/cl_all_{gal_sample}_Planck.npz")

cl_gal = data['cl_gal']
cl_kszsq = data['cl_kszsq']
cl_ksz_gal = data['cl_ksz_gal']
cl_kszsq_gal = data['cl_kszsq_gal']
ell = data['ell']

ell_binned = np.linspace(100, 3000, 15)
_, cl_kszsq_gal_binned = bin_mat(ell, cl_kszsq_gal, ell_binned)
_, cl_kszsq_binned = bin_mat(ell, cl_kszsq, ell_binned)
_, cl_gal_binned = bin_mat(ell, cl_gal, ell_binned)
ell_binned, cl_ksz_gal_binned = bin_mat(ell, cl_ksz_gal, ell_binned)

diff = np.diff(ell_binned)
diff = np.append(diff, diff[-1])
print(diff.shape, ell_binned.shape)
fsky = 1. # already corrected for

cl_kszsq_gal_err = np.sqrt(1./((2.*ell_binned+1)*fsky*diff)*(cl_kszsq_gal_binned**2+cl_gal_binned*cl_kszsq_binned))

factor = ell_binned*(ell_binned+1)/(2.*np.pi)

plt.figure(1, figsize=(9, 7))
plt.errorbar(ell_binned, cl_kszsq_gal_binned*factor, yerr=cl_kszsq_gal_err*factor, ls='', marker='o', capsize=4., color='dodgerblue')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell (\ell + 1) C_\ell^{T^2_{\rm clean}\times \delta_g}/2 \pi \ [\mu K^2]$')
plt.savefig(f'kszsq_gal_figs/cl_kszsq_gal_{gal_sample}.png')
plt.show()
