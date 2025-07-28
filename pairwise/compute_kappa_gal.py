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
import pyccl as ccl
import pymangle
from pixell import utils
from astropy.io import fits

from utils_power import bin_mat

# file names
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
#galaxy_sample = "MGS"
#galaxy_sample = "WISE"
galaxy_sample = "2MPZ"
gal_fn = data_dir+"/sdss_data/post_catalog.dr72bright0.fits"
bias_fn = data_dir+"/sdss_data/bias_dr72bright0.npy"

if galaxy_sample == "WISE":
    # load galaxy delta map
    gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits") 
    nside = hp.npix2nside(len(gal_delta))
    
    # load galaxy mask
    gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
    gal_msk = hp.ud_grade(gal_msk, nside)
elif galaxy_sample == "MGS":
    # load catalog
    hdul = fits.open(gal_fn)
    RA = hdul[1].data['RA'].flatten() # 0, 360
    DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 70
    Z = hdul[1].data['Z'].flatten()
    ABSM = hdul[1].data['ABSM'][:, 2].flatten() # ugrizJKY (r)
    print("maximum redshift = ", np.max(Z))
    print("number of objects = ", len(RA))

    # load galaxy map and low-resolution mask
    delta_fn = data_dir+"/sdss_data/delta_dr72bright0.fits"
    mask_fn = data_dir+"/sdss_data/mask_lores_dr72bright0.fits"
    gal_delta = hp.read_map(delta_fn)
    gal_msk = hp.read_map(mask_fn)
    nside = hp.npix2nside(len(gal_delta))
    print("mean delta = ", np.sum(gal_delta*gal_msk)/np.sum(gal_msk))

# load CMB kappa map
fn_klm = data_dir+"/cmb_data/lensing/dat_klm.fits"
klm, lmax = hp.read_alm(fn_klm, return_mmax=True)
kappa = np.array([hp.alm2map(klm, nside)])

# load CMB kappa mask
fn_mask = data_dir+"/cmb_data/lensing/mask.fits"
cmb_msk = hp.read_map(fn_mask, dtype=float)
cmb_msk = hp.ud_grade(cmb_msk, nside_out=nside)

# load CMB masks
mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"
mw_msk = hp.read_map(mw_msk_fn, verbose=True)
star_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all.fits"
star_msk = hp.read_map(star_msk_fn, verbose=True)

# apodize masks
print("apodizing")
apod_mw = 10.; apod_star = 10.; apod_gal = 10. # arcmin
if apod_mw > 0.:
    mw_msk = hp.smoothing(mw_msk, fwhm=apod_mw*utils.arcmin, iter=5, verbose=False)
    mw_msk[mw_msk > 1.] = 1.; mw_msk[mw_msk < 0.] = 0.
if apod_gal > 0.:
    gal_msk = hp.smoothing(gal_msk, fwhm=apod_gal*utils.arcmin, iter=5, verbose=False)
    gal_msk[gal_msk > 1.] = 1.; gal_msk[gal_msk < 0.] = 0.;
if apod_star > 0.:
    star_msk = hp.smoothing(star_msk, fwhm=apod_star*utils.arcmin, iter=5, verbose=False)
    star_msk[star_msk > 1.] = 1.; star_msk[star_msk < 0.] = 0.;
#cmb_msk *= mw_msk*star_msk
cmb_msk = cmb_msk*mw_msk*star_msk*gal_msk

# combine masks
fsky_cross = np.sum(cmb_msk*gal_msk)/len(cmb_msk)
fsky_cmb = np.sum(cmb_msk**2)/len(cmb_msk)
fsky_gal = np.sum(gal_msk**2)/len(cmb_msk)

# mask fields
gal_masked = gal_delta*gal_msk
kappa_masked = kappa*cmb_msk

# power parameters
LMIN = 0
LMAX = 3001 #3*nside+1
ell_data = np.arange(LMIN, LMAX, 1)

# compute power spectrum
cl_kappa_gal = hp.anafast(kappa_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky_cross
cl_gal = hp.anafast(gal_masked, lmax=LMAX-1, pol=False)/fsky_gal
cl_kappa = hp.anafast(kappa_masked, lmax=LMAX-1, pol=False)/fsky_cmb

# bin in ell
ell_bins = np.linspace(1, 3000, 30)
ell_binned, cl_kappa_gal_binned = bin_mat(ell_data, cl_kappa_gal, ell_bins)
ell_binned, cl_kappa_binned = bin_mat(ell_data, cl_kappa, ell_bins)
ell_binned, cl_gal_binned = bin_mat(ell_data, cl_gal, ell_bins)

# compute N(z)
z_edges = np.linspace(0., 0.5, 1001)
if galaxy_sample == "WISE":
    dNdz = np.ones(len(z_edges)-1)
elif galaxy_sample == "MGS":
    dNdz, _ = np.histogram(Z, bins=z_edges)
z = 0.5*(z_edges[1:] + z_edges[:-1])

# define Cosmology object
cosmo_dic = {'h': 0.6736, 'Omega_c': 0.26447, 'Omega_b': 0.04930, 'A_s': 2.083e-9, 'n_s': 0.9649, 'T_CMB': 2.7255, 'Neff': 2.0328, 'm_nu': 0.06, 'm_nu_type': 'single', 'transfer_function': 'boltzmann_class'}
cosmo = ccl.Cosmology(**cosmo_dic)

# calculate theoretical Cls
# modified /home/boryanah/anaconda3/lib/python3.7/site-packages/pyccl/tracers.py this is for the CMB tracer
# modified /home/boryanah/anaconda3/lib/python3.7/site-packages/pyccl/boltzmann.py this is for the cosmology matching abacus summit
z_source = 1089.3
#ell = np.arange(0, 3000, 1)
cmbl = ccl.CMBLensingTracer(cosmo, z_source=z_source)#, z_min=z_min, z_max=z_max)
cls_cmb_th = ccl.angular_cl(cosmo, cmbl, cmbl, ell_binned)

# set bias
#b = bias/ccl.background.growth_factor(cosmo, 1./(1+z))
b = np.ones_like(z)

# create CCL tracer object for galaxy clustering
gal = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z,b))

# calculate theoretical Cls
cls_gal_th = ccl.angular_cl(cosmo, gal, gal, ell_binned)
cls_cross_th = ccl.angular_cl(cosmo, gal, cmbl, ell_binned)

# save result
np.savez(f"data/cl_kappa_gal_{galaxy_sample}.npz", cl_kappa_gal_binned=cl_kappa_gal_binned, cl_kappa_binned=cl_kappa_binned, cl_gal_binned=cl_gal_binned, ell_binned=ell_binned, cls_cmb_th=cls_cmb_th, cls_gal_th=cls_gal_th, cls_cross_th=cls_cross_th)

# plot
plt.figure(1)
plt.plot(ell_binned, cl_kappa_gal_binned, color='r', label="cross binned")
plt.plot(ell_binned, cls_cross_th, color='r', ls='--', label="cross theory")
plt.plot(ell_binned, cl_gal_binned, color='g', label="gal binned")
plt.plot(ell_binned, cls_gal_th, color='g', ls='--', label="gal theory")
plt.plot(ell_binned, cl_kappa_binned, color='b', label="kappa binned")
plt.plot(ell_binned, cls_cmb_th, color='b', ls='--', label="kappa theory")
plt.legend()
plt.yscale('log')

plt.figure(2)
#plt.plot(ell_binned, cl_kappa_gal_binned/cls_cross_th, color='r', label="cross binned")
plt.plot(ell_binned, cl_gal_binned/cls_gal_th, color='g', label="gal binned")
#plt.plot(ell_binned, cl_kappa_binned/cls_cmb_th, color='b', label="kappa binned")
plt.legend()
#plt.yscale('log')
plt.savefig(f"figs/kappa_{galaxy_sample}.png") 
plt.show()
