import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class
from astropy.io import fits
from astropy.io import ascii

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

def load_cmb_sample(cmb_sample, save=False):
    # filename of CMB map
    if cmb_sample == "ACT_BN":
        fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
        msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"
    elif cmb_sample == "ACT_D56":
        fn = "../cmb_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
        msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"

    # reading fits file
    mp = enmap.read_fits(fn)
    msk = enmap.read_fits(msk_fn)

    # save map
    if save:
        fig_name = (fn.split('/')[-1]).split('.fits')[0]
        eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
        eshow(msk, fig_name+"_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
        plt.close()
    return mp, msk

def load_galaxy_sample(galaxy_sample, cmb_sample, cmb_box):

    # filename of galaxy map
    if galaxy_sample == "2MPZ":
        gal_fn = "../galaxy_tests/2mpz_data/2MPZ.fits"
    elif galaxy_sample == "BOSS_North":
        gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_North.fits"
    elif galaxy_sample == "BOSS_South":
        gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_South.fits"
    elif "SDSS" in galaxy_sample:
        gal_fn = "../galaxy_tests/sdss_data/V21_DR15_Catalog_v4.txt"
    
    # load galaxy sample
    if galaxy_sample == '2MPZ':
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
        DEC = hdul[1].data['DEC'].flatten()/utils.degree # -180, 180
        Z = hdul[1].data['ZPHOTO'].flatten()
        #Z = hdul[1].data['ZSPEC'].flatten()
        K_rel = hdul[1].data['KCORR'].flatten() # might be unnecessary since 13.9 is the standard
        choice = (K_rel < 13.9) & (Z > 0.0) # original is 13.9
    elif 'BOSS' in galaxy_sample:
        hdul = fits.open(gal_fn)
        RA = hdul[1].data['RA'].flatten() # 0, 360
        DEC = hdul[1].data['DEC'].flatten() # -180, 180 # -10, 36
        Z = hdul[1].data['Z'].flatten()
        choice = np.ones(len(Z), dtype=bool)
    elif 'SDSS' in galaxy_sample:
        data = ascii.read(gal_fn)
        RA = data['ra'] # 0, 360
        DEC = data['dec'] # -180, 180 # -10, 36
        Z = data['z']
        L = data['lum']
        L_lo, L_hi = get_sdss_lum_lims(galaxy_sample)
        choice = (L_lo < L) & (L_hi >= L)
        if "ACT_BN" in cmb_sample or "ACT_D56" in cmb_sample:
            choice &= data['S16ILC'] == 1.
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
    choice &= DEC_choice & RA_choice
    RA = RA[choice]
    DEC = DEC[choice]
    Z = Z[choice]
    index = index[choice]
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