"""
Conversion factor taken from: https://irsa.ipac.caltech.edu/docs/knowledgebase/ercsc-validation-conversions.pdf. There is a typo for Table V, 545 GHz (Tthermo to MJy/sr): 57 -> 571.
"""
import os
import sys
sys.path.append('../pairwise')

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from utils_power import bin_mat

# names of stuff
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
galaxy_sample = "WISE"; tracer_name = "WISE"
galaxy_sample = "DECALS"; tracer_name = "DELS__1" # (all)

# load mask
msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"
star_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all.fits"
msk = hp.read_map(msk_fn, verbose=True)
npix = len(msk)
nside = hp.npix2nside(npix)
star_msk = hp.read_map(star_msk_fn, verbose=True)
if galaxy_sample == 'WISE':
    gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
elif galaxy_sample == 'DECALS':
    gal_msk = hp.read_map(data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits")
gal_msk = hp.ud_grade(gal_msk, nside)

# load galaxy delta map
gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits")

# load CMB kappa
file_klm = data_dir+"/cmb_data/lensing/dat_klm.fits"
file_mask = data_dir+"/cmb_data/lensing/mask.fits"
klm, lmax = hp.read_alm(file_klm, return_mmax=True)
kappa = np.array([hp.alm2map(klm, nside)])
kappa_msk = hp.read_map(file_mask, dtype=float)
kappa_msk = hp.ud_grade(kappa_msk, nside_out=nside)

# combine masks
msk *= (kappa_msk*gal_msk*star_msk)
fsky = np.sum(msk)*1./len(msk)

# mask fields
gal_masked = gal_delta*msk
kappa_masked = kappa*msk

# power parameters
LMIN = 0
LMAX = 3*nside+1
ell_data = np.arange(LMIN, LMAX, 1)

# compute power spectrum
cl_kappa_gal = hp.anafast(kappa_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky

# bin in ell
ell_binned = np.linspace(0, 3000, 30)
ell_binned, cl_kappa_gal_binned = bin_mat(ell_data, cl_kappa_gal, ell_binned)
np.save("cl_kappa_gal.npy", cl_kappa_gal)

# read cross-correlation power spectrum and fit a polynomial to it (in log-log)
import sacc

# read file
s = sacc.Sacc.load_fits('data_lensing/cls_cov_all.fits')
l_s, cl_s, cov_s = s.get_ell_cl('cl_00', tracer_name, 'CMBk', return_cov=True)
print("David bins = ", l_s, len(l_s))

# plot
plt.errorbar(l_s, cl_s, yerr=np.sqrt(np.diag(cov_s)))
plt.plot(ell_data, cl_kappa_gal)
plt.plot(ell_binned, cl_kappa_gal_binned)
plt.loglog()
plt.savefig("kappa.png")
plt.show()
