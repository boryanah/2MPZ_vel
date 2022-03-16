import gc
import os
import time

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from util import *
from utils_power import bin_mat

# galaxy and cmb sample
#cmb_sample = "ACT_BN"; source_arcmin = 8.
#cmb_sample = "ACT_D56"; source_arcmin = 8.
#cmb_sample = "ACT_DR5_f090"; noise_uK = 45; source_arcmin = 8.
#cmb_sample = "ACT_DR5_f150"; noise_uK = 45; source_arcmin = 8.
cmb_sample = "Planck_healpix"; 

data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"

# filename of CMB map
if cmb_sample == "ACT_BN":
    fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"
    #msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_BN_compsep_ps_{source_arcmin:.1f}arcmin.fits"
elif cmb_sample == "ACT_D56":
    fn = "../cmb_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"
    #msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_D56_compsep_ps_{source_arcmin:.1f}arcmin.fits"
elif cmb_sample == "ACT_DR5_f090":
    fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits" # DR5 f090
    #msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f090_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f090_ivar_{noise_uK:d}uK.fits"
elif cmb_sample == "ACT_DR5_f150":
    fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f150_daynight_map.fits" # DR5 f150
    #msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f150_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f150_ivar_{noise_uK:d}uK.fits"
elif cmb_sample == "Planck_healpix":
    mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    #msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
    msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"

    # read maps and convert units
    mp = hp.read_map(mp_fn, verbose=True)
    msk = hp.read_map(msk_fn, verbose=True)
    mp *=  1.e6 # uK
    assert len(msk) == len(mp)

    # healpix params
    npix = len(msk)
    nside = hp.npix2nside(npix)

    # apply apodization and compute fsky
    smooth_scale_mask = 0.05 #0., 0.05 # in radians
    if smooth_scale_mask > 0.:
        msk = msk*hp.smoothing(msk, fwhm=smooth_scale_mask, iter=5, verbose=False)     
    fsky = np.sum(msk)*1./len(msk)
    print("fsky = ", fsky)
    
    # power parameters
    LMIN = 0
    LMAX = 3*nside+1
    ell_data = np.arange(LMIN, LMAX, 1)
    ell_data_binned = np.linspace(100, 4000, 400)

    # compute power spectrum using anafast dividing by fsky
    cmb_masked = mp*msk
    hp.mollview(msk)
    hp.mollview(cmb_masked)
    plt.show()
    cl_data = hp.anafast(cmb_masked, lmax=LMAX-1, pol=False) # uK^2
    cl_data /= fsky

    # bin power spectrum
    ell_data_binned, cl_data_binned = bin_mat(ell_data, cl_data, ell_data_binned)
    
    # load cmb power from theory
    camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
    cl_th = camb_theory[0, 0, :3000]
    ell_th = np.arange(cl_th.size)

    # for plotting
    power = 2.
    
    # plot stuff
    plt.figure(1)
    plt.plot(ell_data, cl_data*ell_data**power, label="unbinned")
    plt.plot(ell_data_binned, cl_data_binned*ell_data_binned**power, label="binned")
    plt.plot(ell_th, cl_th*ell_th**power, lw=3, color='k')
    plt.legend()
    plt.xlim([0, 3000])
    plt.show()
    
    # save power
    np.save("camb_data/Planck_power.npy", cl_data)
    np.save("camb_data/Planck_ell.npy", ell_data)
    quit()
    
# reading fits file
mp = enmap.read_fits(fn)
if "DR5" in cmb_sample:
    mp = mp[0]
    gc.collect()
msk = enmap.read_fits(msk_fn)
assert mp.shape == msk.shape

binned_power, centers = compute_power(mp, mask=msk, test=True)
np.save(f"camb_data/{cmb_sample}_binned_power.npy", binned_power)
np.save(f"camb_data/{cmb_sample}_centers.npy", centers)
