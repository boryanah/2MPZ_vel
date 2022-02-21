import gc
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from util import *

# galaxy and cmb sample
#cmb_sample = "ACT_BN"; source_arcmin = 8.
#cmb_sample = "ACT_D56"; source_arcmin = 8.
#cmb_sample = "ACT_DR5_f090"; noise_uK = 45; source_arcmin = 8.
cmb_sample = "ACT_DR5_f150"; noise_uK = 45; source_arcmin = 8.

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
