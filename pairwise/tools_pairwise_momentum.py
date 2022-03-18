import os
import gc
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
from pixell import enmap, enplot, utils, reproject
from classy import Class
from astropy.io import fits
from astropy.io import ascii
from astropy import units as u
from astropy.coordinates import SkyCoord

from util import eshow

def get_P_D_A(Cosmo, RA, DEC, Z):
    # transform to cartesian coordinates (checked)
    CX = np.cos(RA*utils.degree)*np.cos(DEC*utils.degree)
    CY = np.sin(RA*utils.degree)*np.cos(DEC*utils.degree)
    CZ = np.sin(DEC*utils.degree)

    # stack together normalized positions
    P = np.vstack((CX, CY, CZ)).T

    # comoving distance to observer and angular size
    CD = np.zeros(len(Z))
    D_A = np.zeros(len(Z))
    for i in range(len(Z)):
        if Z[i] < 0.: continue
        lum_dist = Cosmo.luminosity_distance(Z[i])
        CD[i] = lum_dist/(1.+Z[i]) # Mpc # pretty sure of the units since classylss has Mpc/h and multiplies by h 
        D_A[i] = lum_dist/(1.+Z[i])**2 # Mpc
    P = P*CD[:, None]
    return P, D_A

def load_cmb_sample(cmb_sample, data_dir, source_arcmin, noise_uK, save=False):
    # filename of CMB map
    if cmb_sample == "ACT_BN":
        fn = data_dir+"/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_BN_compsep_ps_{source_arcmin:.1f}arcmin.fits"
        #msk_fn = data_dir+"/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"
    elif cmb_sample == "ACT_D56":
        fn = data_dir+"/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_D56_compsep_ps_{source_arcmin:.1f}arcmin.fits"
        #msk_fn = data_dir+"/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"
    elif cmb_sample == "ACT_DR5_f090":
        fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f090_daynight_map.fits" # DR5 f090
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f090_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    elif cmb_sample == "ACT_DR5_f150":
        fn = data_dir+"/cmb_data/act_planck_dr5.01_s08s18_AA_f150_daynight_map.fits" # DR5 f150
        msk_fn = data_dir+f"/cmb_data/comb_mask_ACT_DR5_f150_ivar_{noise_uK:d}uK_ps_{source_arcmin:.1f}arcmin.fits"
    elif cmb_sample == "Planck":
        fn = data_dir+"/cmb_data/Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits" # pixell
        #fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits" # hp
        #msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all_GalPlane-apo0_2048_R2.00.fits"
        msk_fn = data_dir+"/cmb_data/Planck_HFI_Mask_PointSrc_all_GalPlane-apo0_2048_R2.00_uK.fits"
        #msk_fn = data_dir+f"/cmb_data/ps_mask_Planck_{source_arcmin:.1f}arcmin.fits"
        
    """
    # generate Planck map
    res_arcmin = 0.5
    DEC_center, RA_center = 0., 0.
    nx = int(180./(res_arcmin/60.)) # DEC
    ny = int(360./(res_arcmin/60.)) # RA
    print("nx, ny = ", nx, ny)
    shape, wcs = enmap.geometry(shape=(nx, ny), res=res_arcmin*utils.arcmin, pos=(DEC_center, RA_center))
    #mp = reproject.enmap_from_healpix(fn, shape, wcs, ncomp=1, unit=1.e-6, lmax=6000, rot="gal,equ")
    mp = reproject.enmap_from_healpix(msk_fn, shape, wcs, ncomp=1, unit=1.e-6, lmax=6000, rot="gal,equ")
    mp = mp.astype(np.float32)
    print("pshape, pwcs = ", mp.shape, mp.wcs)
    print("box = ", enmap.box(mp.shape, mp.wcs)/utils.degree)
    print("pbox = ", enmap.box(mp.shape, mp.wcs)/utils.degree)
    #enmap.write_fits(data_dir+f"/cmb_data/Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits", mp)
    enmap.write_fits(data_dir+f"/cmb_data/Planck_HFI_Mask_PointSrc_all_GalPlane-apo0_2048_R2.00_uK.fits", mp)
    """
    
    # reading fits file
    mp = enmap.read_fits(fn)

    if "DR5" in cmb_sample:
        mp = mp[0] # three maps are available
        gc.collect()
    elif "Planck" in cmb_sample:
        mp = mp[0] # saved as (1, 10800, 21600)
    if msk_fn is None:
        msk = mp*0.+1.
        msk[mp == 0.] = 0.
    else:
        msk = enmap.read_fits(msk_fn)
        
    if "Planck" in cmb_sample:
        msk = msk[0] # saved as (1, 10800, 21600)
    print("msk == 0", np.sum(np.isclose(msk, 0.))/np.product(msk.shape), msk.shape, msk.min(), msk.max(), msk.mean())
    
    # save map
    if save:
        fig_name = (fn.split('/')[-1]).split('.fits')[0]
        mp_box = np.rad2deg(mp.box())
        print("decfrom, decto, rafrom, rato = ", mp_box[0, 0], mp_box[1, 0], mp_box[0, 1], mp_box[1, 1])
        eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
        eshow(msk, fig_name+"_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
        plt.close()
    return mp, msk

def load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random):

    # filename of galaxy map
    if galaxy_sample == "2MPZ":
        gal_fn = data_dir+"/2mpz_data/2MPZ_FULL_wspec_coma_complete.fits"#2MPZ.fits
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "2MPZ_Biteau":
        #gal_fn = data_dir+"/2mpz_data/2MPZ_Biteau.npz"
        gal_fn = data_dir+"/2mpz_data/2MPZ_Biteau_radec.npz"
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "WISExSCOS":
        gal_fn = data_dir+"wisexscos_data/WIxSC.fits"
        mask_fn = data_dir+"/2mpz_data/WISExSCOSmask.fits"
    elif galaxy_sample == "DECALS":
        gal_fn = data_dir+"dels_data/Legacy_Survey_DECALS_galaxies-selection.fits"
        mask_fn = data_dir+"dels_data/Legacy_footprint_final_mask.fits"
    elif galaxy_sample == "BOSS_North":
        gal_fn = data_dir+"/boss_data/galaxy_DR12v5_CMASS_North.fits"
    elif galaxy_sample == "BOSS_South":
        gal_fn = data_dir+"/boss_data/galaxy_DR12v5_CMASS_South.fits"
    elif galaxy_sample == "eBOSS_SGC":
        gal_fn = data_dir+"/eboss_data/eBOSS_ELG_clustering_data-SGC-vDR16.fits"
    elif galaxy_sample == "eBOSS_NGC":
        gal_fn = data_dir+"/eboss_data/eBOSS_ELG_clustering_data-NGC-vDR16.fits"
    elif "SDSS" in galaxy_sample:
        gal_fn = data_dir+"/sdss_data/V21_DR15_Catalog_v4.txt"
    
    # load galaxy sample
    if galaxy_sample == '2MPZ':
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
        DEC = hdul[1].data['DEC'].flatten()/utils.degree # -90, 90
        print("DEC min/max", DEC.min(), DEC.max())
        ZPHOTO = hdul[1].data['ZPHOTO'].flatten()
        ZSPEC = hdul[1].data['ZSPEC'].flatten()
        """
        plt.figure(figsize=(9, 7))
        plt.plot([ZPHOTO.min(), ZPHOTO.max()], [ZPHOTO.min(), ZPHOTO.max()], ls='--', lw=3, color='black')
        plt.scatter(ZSPEC[ZSPEC > 0.], ZPHOTO[ZSPEC > 0.], marker='x', color='red', s=5)
        plt.show()
        """
        #mode = "ZPHOTO"
        #mode = "ZSPEC" # complete for K < 11.65
        mode = "ZMIX"
        if mode == "ZPHOTO":
            Z = ZPHOTO.copy()
        elif mode == "ZSPEC":
            Z = ZSPEC.copy()
        elif mode == "ZMIX":
            Z = ZPHOTO.copy()
            Z[ZSPEC > 0.] = ZSPEC[ZSPEC > 0.]
        K_rel = hdul[1].data['KCORR'].flatten() # might be unnecessary since 13.9 is the standard
        B = hdul[1].data['B'].flatten() # -90, 90
        L = hdul[1].data['L'].flatten() # 0, 360
        if want_random != -1:
            np.random.seed(want_random)
            factor = 3
            N_rand = len(RA)*factor
            Z = np.repeat(Z, factor)
            K_rel = np.repeat(K_rel, factor)
            """
            RA = np.repeat(RA, factor)
            DEC = np.repeat(DEC, factor)
            inds_ra = np.arange(len(RA), dtype=int)
            inds_dec = np.arange(len(RA), dtype=int)
            np.random.shuffle(inds_ra)
            np.random.shuffle(inds_dec)
            RA = RA[inds_ra]
            DEC = DEC[inds_dec]
            """
            costheta = np.random.rand(N_rand)*2.-1.
            theta = np.arccos(costheta)
            DEC = theta*(180./np.pi) # 0, 180
            DEC -= 90.
            RA = np.random.rand(N_rand)*360.
            print("RA/DEC range", RA.min(), RA.max(), DEC.min(), DEC.max())
            c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
            B = c_icrs.galactic.b.value
            L = c_icrs.galactic.l.value

        #choice = (K_rel < 13.9) & (Z > 0.0) # original is 13.9
        #choice = (K_rel < 13.9) & (Z > 0.0) & (Z < 0.3)  # original is 13.9
        #choice = (K_rel < 13.9) & (Z > 0.0) & (Z < 0.0773)
        choice = (K_rel < 13.9) & (Z > 0.0)
        #choice = (K_rel < 11.65) & (Z > 0.0) & (Z < 0.3)
        #choice = (K_rel < 13.9) & (Z < 0.15); Z[Z < 0.] = 0.
        
        lum_cut = True
        if lum_cut:
            K_abs = np.zeros_like(K_rel)+100. # make it faint
            for i in range(len(Z)):
                if Z[i] < 0.: continue
                lum_dist = Cosmo.luminosity_distance(Z[i]) # Mpc
                E_z = Z[i]
                K_z = -6.*np.log10(1. + Z[i])
                K_abs[i] = K_rel[i] - 5.*np.log10(lum_dist) - 25. + K_z + E_z
            K_perc = np.percentile(K_abs, 40.)#33.)
            print(K_abs.min(), K_perc, K_abs.max())
            
            choice &= (K_abs < K_perc)
        
        # apply 2MPZ mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
    elif '2MPZ_Biteau' == galaxy_sample:
        data = np.load(gal_fn)
        RA = data['RA']
        DEC = data['DEC']
        #Z = data['Z']
        Z = data['Z_hdul'] # TESTING
        Mstar = data['M_star']
        d_L = data['d_L']
        B = data['B']
        L = data['L']
        choice = np.ones(len(Z), dtype=bool)

        #c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
        #B = c_icrs.galactic.b.value
        #L = c_icrs.galactic.l.value

        # apply 2MPZ mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
        
        mass_cut = True
        if mass_cut:
            Mstar_perc = np.percentile(Mstar, 30.)
            print("Mstar threshold = ", Mstar_perc)
            Mstar_perc = 10.3 # TESTING
            print("Mstar threshold = ", Mstar_perc)
            choice &= (Mstar > Mstar_perc)
        dL_max = 350. # Mpc sample complete 0.0773
        dL_min = 100. # Mpc 0.0229
        #dL_min = 0. # Mpc 0.
        choice &= (d_L < dL_max) & (d_L > dL_min)
        
        # could add mask
    elif galaxy_sample == 'WISExSCOS':
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
        DEC = hdul[1].data['DEC'].flatten()/utils.degree # -90, 90
        B = hdul[1].data['B'].flatten() # 0, 360
        L = hdul[1].data['L'].flatten() # -90, 90
        print("DEC min/max", DEC.min(), DEC.max())
        Z = hdul[1].data['ZPHOTO_CORR'].flatten()
        choice = (Z > 0.1) & (Z < 0.35)
        
        # apply mask
        B *= utils.degree
        L *= utils.degree
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.

    elif galaxy_sample == "DECALS":
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90
        Z = hdul[1].data['PHOTOZ_3DINFER'].flatten()
        choice = np.ones(len(Z), dtype=bool)
        
        """
        #c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
        #B = c_icrs.galactic.b.value
        #L = c_icrs.galactic.l.value

        # apply mask
        #B *= utils.degree
        #L *= utils.degree
        B = RA*utils.degree # TESTING
        L = DEC*utils.degree # TESTING
        x = np.cos(B)*np.cos(L)
        y = np.cos(B)*np.sin(L)
        z = np.sin(B)
        mask = hp.read_map(mask_fn) # ring, not nested
        npix = len(mask)
        nside = hp.npix2nside(npix)
        ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
        choice &= mask[ipix] == 1.
        """
        print("DEC min/max", DEC.min(), DEC.max())
        print("RA min/max", RA.min(), RA.max())
        
    elif 'BOSS' in galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 36
        Z = hdul[1].data['Z'].flatten()
        choice = np.ones(len(Z), dtype=bool)
    elif 'eBOSS' in galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 36
        Z = hdul[1].data['Z'].flatten()
        choice = np.ones(len(Z), dtype=bool)
    elif 'SDSS' in galaxy_sample:
        data = ascii.read(gal_fn)
        RA = data['ra'] # 0, 360
        DEC = data['dec'] # -90, 90 # -10, 36
        Z = data['z']
        L = data['lum']
        L_lo, L_hi = get_sdss_lum_lims(galaxy_sample)
        choice = (L_lo < L) & (L_hi >= L)
        if "ACT_BN" in cmb_sample or "ACT_D56" in cmb_sample:
            choice &= data['S16ILC'] == 1.
        elif "ACT_DR5_f090" in cmb_sample or "ACT_DR5_f150" in cmb_sample:
            choice &= data['S18coadd'] == 1.
        print("galaxies satisfying luminosty cut = ", np.sum(choice))

    # galaxy indices before applying any cuts
    index = np.arange(len(Z), dtype=int)
    print("Zmin/max/med = ", Z.min(), Z.max(), np.median(Z))
    print("RAmin/RAmax = ", RA.min(), RA.max())
    print("DECmin/DECmax = ", DEC.min(), DEC.max())

    # make magnitude and RA/DEC cuts to  match ACT
    DEC_choice = (DEC <= cmb_box['decto']) & (DEC > cmb_box['decfrom'])
    if cmb_sample == 'ACT_D56':
        RA_choice = (RA <= cmb_box['rafrom']) | (RA > cmb_box['rato'])
    elif cmb_sample == 'ACT_BN':
        RA_choice = (RA <= cmb_box['rafrom']) & (RA > cmb_box['rato'])
    else:
        # DR5 has RA from -180 to 180, so cmb_box is not used
        RA_choice = np.ones_like(DEC_choice)
    choice &= DEC_choice & RA_choice
    RA = RA[choice]
    DEC = DEC[choice]
    Z = Z[choice]
    index = index[choice]

    if galaxy_sample == "2MPZ":
        if mode == "ZMIX":
            ZSPEC = ZSPEC[choice]
            assert len(ZSPEC) == len(Z)
            print("percentage zspec available = ", np.sum(ZSPEC > 0.)*100./len(ZSPEC))
    print("number of galaxies = ", np.sum(choice))

    return RA, DEC, Z, index

def get_sdss_lum_lims(galaxy_sample):
    if "all" in galaxy_sample:
        L_lo = 0.
        L_hi = 1.e20
    if "L43" in galaxy_sample:
        L_lo = 4.3e10
        if "L43D" in galaxy_sample:
            L_hi = 6.1e10
        else:
            L_hi = 1.e20
    elif "L61" in galaxy_sample:
        L_lo = 6.1e10
        if "L61D" in galaxy_sample:
            L_hi = 7.9e10
        else:
            L_hi = 1.e20
    elif "L79" in galaxy_sample:
        L_lo = 7.9e10
        L_hi = 1.e20
    return L_lo, L_hi
