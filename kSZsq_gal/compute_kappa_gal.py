"""
Conversion factor taken from: https://irsa.ipac.caltech.edu/docs/knowledgebase/ercsc-validation-conversions.pdf. There is a typo for Table V, 545 GHz (Tthermo to MJy/sr): 57 -> 571.
"""
import os
import sys
sys.path.append('../pairwise')

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sacc
from pixell import utils

from utils_power import bin_mat

# names of stuff
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
galaxy_sample = "WISE"; tracer_name = "WISE"
#galaxy_sample = "DECALS"; tracer_name = "DELS__1" # (all)
want_joint = False
apod_mw = 10.; apod_star = 10.; apod_gal = 10. # arcmin
apod_mw *= utils.arcmin # (10 arcmin in paper)
apod_star *= utils.arcmin # (2 arcmin for point source masks)
apod_gal = apod_gal*utils.arcmin if want_joint else 0. # if treated separately, then galaxies don't need a mask

# read David's file (only used to compare)
s = sacc.Sacc.load_fits('data_lensing/cls_cov_all.fits')
l_s, cl_s, cov_s = s.get_ell_cl('cl_00', tracer_name, 'CMBk', return_cov=True)
print("David bins = ", l_s, len(l_s))

# load CMB masks
mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"
mw_msk = hp.read_map(mw_msk_fn, verbose=True)
star_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all.fits"
star_msk = hp.read_map(star_msk_fn, verbose=True)
npix = len(mw_msk)
nside = hp.npix2nside(npix)

# load galaxy delta map
gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits") # og
#gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}_all_DELS__1.fits") # og
#gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}_ZHOU_z0.0_z0.8.fits") # TESTING

# load galaxy mask
if galaxy_sample == 'WISE':
    gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
elif galaxy_sample == 'DECALS':
    gal_msk = hp.read_map(data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits")
gal_msk = hp.ud_grade(gal_msk, nside)

# load CMB kappa map
fn_klm = data_dir+"/cmb_data/lensing/dat_klm.fits"
klm, lmax = hp.read_alm(fn_klm, return_mmax=True)
kappa = np.array([hp.alm2map(klm, nside)])

# load CMB kappa mask
fn_mask = data_dir+"/cmb_data/lensing/mask.fits"
kappa_msk = hp.read_map(fn_mask, dtype=float)
kappa_msk = hp.ud_grade(kappa_msk, nside_out=nside)
#hp.mollview(kappa_msk)
#plt.show()

# apodize masks
if apod_mw > 0.:
    mw_msk = hp.smoothing(mw_msk, fwhm=apod_mw, iter=5, verbose=False)
    mw_msk[mw_msk > 1.] = 1.; mw_msk[mw_msk < 0.] = 0.
if apod_gal > 0.:
    gal_msk = hp.smoothing(gal_msk, fwhm=apod_gal, iter=5, verbose=False)
    gal_msk[gal_msk > 1.] = 1.; gal_msk[gal_msk < 0.] = 0.;
if apod_star > 0.:
    star_msk = hp.smoothing(star_msk, fwhm=apod_star, iter=5, verbose=False)
    star_msk[star_msk > 1.] = 1.; star_msk[star_msk < 0.] = 0.;

# combine masks
if want_joint:
    cmb_msk = kappa_msk*mw_msk*star_msk*gal_msk
    gal_msk = cmb_msk
else:
    cmb_msk = mw_msk*star_msk
fsky = np.sum(cmb_msk*gal_msk)/len(cmb_msk)

# mask fields
gal_masked = gal_delta*gal_msk
kappa_masked = kappa*cmb_msk

# power parameters
LMIN = 0
LMAX = 3*nside+1
ell_data = np.arange(LMIN, LMAX, 1)

# compute power spectrum
cl_kappa_gal = hp.anafast(kappa_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky

# bin in ell (close to David)
#ell_bins = np.linspace(1, 3000, 30)
diff = np.diff(l_s)
diff = np.hstack((diff, diff[-1]))
ell_bins = np.append(0., np.cumsum(diff))
ell_binned, cl_kappa_gal_binned = bin_mat(ell_data, cl_kappa_gal, ell_bins)
np.savez(f"kszsq_gal_data/cl_kappa_gal_{galaxy_sample}.npz", cl_kappa_gal_binned=cl_kappa_gal_binned, ell_binned=ell_binned, ell_david=l_s, cl_david=cl_s)# TESTING!!!!!!!!!!!!!
#np.savez(f"kszsq_gal_data/cl_kappa_gal_{galaxy_sample}_ZHOU_z0.0_z0.8.npz", cl_kappa_gal_binned=cl_kappa_gal_binned, ell_binned=ell_binned)# TESTING!!!!!!!!!!!!!
#np.savez(f"kszsq_gal_data/cl_kappa_gal_{galaxy_sample}_all_DELS__1.npz", cl_kappa_gal_binned=cl_kappa_gal_binned, ell_binned=ell_binned)# TESTING!!!!!!!!!!!!!
print(ell_binned)
print(l_s)

# plot
plt.errorbar(l_s, cl_s, yerr=np.sqrt(np.diag(cov_s)))
plt.plot(ell_data, cl_kappa_gal)
plt.plot(ell_binned, cl_kappa_gal_binned)
plt.loglog()
plt.savefig(f"kszsq_gal_figs/kappa_{galaxy_sample}.png") 
plt.show()
