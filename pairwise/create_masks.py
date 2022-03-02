import gc
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils

from util import *

cmb_sample = "ACT_BN"
cmb_sample = "ACT_D56"
cmb_sample = "ACT_DR5_f150"
#cmb_sample = "ACT_DR5_f090"
cmb_sample = "Planck"

mp_dir = "/mnt/marvin1/boryanah/2MPZ_vel/cmb_data/"
noise_max = 45 # uK (45, 65) 

if cmb_sample == "ACT_D56":
    mJy_min = 0.015
else:
    mJy_min = 0.100
theta_src = 35. # arcmin (5., 8., 10., 35., None) excision size

# filename of CMB map
if cmb_sample == "ACT_BN":
    fn = mp_dir+"tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
    msk_fn = mp_dir+"act_dr4.01_s14s15_BN_compsep_mask.fits"

elif cmb_sample == "ACT_D56":
    fn = mp_dir+"tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
    msk_fn = mp_dir+"act_dr4.01_s14s15_D56_compsep_mask.fits"

elif cmb_sample == "Planck":
    fn = mp_dir+"Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits" 
    msk_fn = None

elif cmb_sample == "ACT_DR5_f090":
    fn = mp_dir+"act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits"
    ivar_fn = mp_dir+"act_dr5.01_s08s18_AA_f090_daynight_ivar.fits"

elif cmb_sample == "ACT_DR5_f150":
    fn = mp_dir+"act_planck_dr5.01_s08s18_AA_f150_daynight_map.fits"
    ivar_fn = mp_dir+"act_dr5.01_s08s18_AA_f150_daynight_ivar.fits"

if theta_src is not None:
    ps_fn = mp_dir+f"masks/source_masks/act_dr4.01_mask_s13s16_{mJy_min:.3f}mJy_{theta_src:.1f}arcmin.fits"

if cmb_sample in ["ACT_BN", "ACT_D56"]:
    # BN and D56

    if theta_src == None: print("You already have that mask"); exit()

    # reading fits file
    #mp = enmap.read_fits(fn)
    msk = enmap.read_fits(msk_fn)

    # load ps map and take small part of the mask corresponding to the cmb map dimensions
    ps = enmap.read_fits(ps_fn)
    box = msk.box()
    if "BN" in cmb_sample:
        box -= 1.e-6
    #s_ps = ps.submap(box)
    s_ps = enmap.submap(ps, box)
    print("ones percentage = ", np.sum(s_ps)*100./np.prod(s_ps.shape))
    print(s_ps.shape, s_ps.min(), np.median(s_ps), s_ps.max())
    print(msk.shape, msk.min(), np.median(msk), msk.max())

    #shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin)#, proj='car')
    #zeros = enmap.zeros(shape, wcs)
    #print(zeros.shape)

    new_msk = msk.copy()
    new_msk[s_ps == 0.] = 0 

    # save map
    enmap.write_fits(mp_dir+f"comb_mask_{cmb_sample}_compsep_ps_{theta_src:.1f}arcmin.fits", new_msk)

elif cmb_sample:
    assert msk_fn == None

    
    # reading fits file
    mp = enmap.read_fits(fn)[0]
    msk = mp*0.+1.

    # load ps map and take small part of the mask corresponding to the cmb map dimensions
    ps = enmap.read_fits(ps_fn)
    box = msk.box()
    if "Planck" in cmb_sample:
        box -= 1.e-6
    s_ps = enmap.submap(ps, box)
    print("ones percentage = ", np.sum(s_ps)*100./np.prod(s_ps.shape))
    print("s_ps = ", s_ps.shape, s_ps.min(), np.median(s_ps), s_ps.max())
    print("msk = ", msk.shape, msk.min(), np.median(msk), msk.max())

    #shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin)#, proj='car')
    #zeros = enmap.zeros(shape, wcs)
    #print(zeros.shape)

    new_msk = msk.copy()
    new_msk[s_ps == 0.] = 0 
    print("percentage masked = ", np.sum(new_msk == 0.)*100./np.product(new_msk.shape))
    
    # save map
    enmap.write_fits(mp_dir+f"ps_mask_{cmb_sample}_{theta_src:.1f}arcmin.fits", new_msk)
    
else:
    # DR5 (f090 and f150)
    
    # reading fits file
    mp = enmap.read_fits(fn)[0]
    box = mp.box()-1.e-6
    del mp
    gc.collect()
    ivar = enmap.read_fits(ivar_fn)[0]
    s_std = enmap.submap(ivar, box)**(-0.5)
    del ivar
    gc.collect()
    new_msk = (s_std < noise_max).astype(int)
    del s_std
    gc.collect()
    
    if theta_src is not None:
        # load ps map and take small part of the mask corresponding to the cmb map dimensions
        ps = enmap.read_fits(ps_fn)
        #s_ps = ps.submap(box)
        s_ps = enmap.submap(ps, box)
        print("ones percentage = ", np.sum(s_ps)*100./np.prod(s_ps.shape))
        print(s_ps.shape, s_ps.min(), np.median(s_ps), s_ps.max())
        print(s_std.shape, s_std.min(), np.median(s_std), s_std.max())
        print("lower noise pcent = ", np.sum(new_msk)*100./np.prod(new_msk.shape))
        new_msk[s_ps == 0] = 0 

        # save map
        enmap.write_fits(mp_dir+f"comb_mask_{cmb_sample}_ivar_{noise_max}uK_ps_{theta_src:.1f}arcmin.fits", new_msk)
    else:
        # save map
        enmap.write_fits(mp_dir+f"comb_mask_{cmb_sample}_ivar_{noise_max}uK.fits", new_msk)
    
#shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin)#, proj='car')
#zeros = enmap.zeros(shape, wcs)
#print(zeros.shape)
    
save = 1
if save:
    #fig_name = (fn.split('/')[-1]).split('.fits')[0]
    #eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
    eshow(new_msk, f"{cmb_sample}_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
    plt.close()



