import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class
from astropy.io import fits
from astropy.io import ascii

from util import *
from estimator import pairwise_momentum

# ask sqrt 2, 2 stuff and flux and resolution, how to combine samples

# settings
#rApMinArcmin = 1. # minimum aperture for bins
#rApMaxArcmin = 10. # maximum aperture for bins
#resCutoutArcmin = 0.5 # stamp resolution
resCutoutArcmin = 0.05 # stamp resolution
projCutout = 'cea' # projection
want_jackknife = False
h = 0.7; Omega_Lambda = 0.7; Omega_cdm = 0.25; Omega_b = 0.05

# galaxy and cmb sample
#cmb_sample = "ACT_BN"
#cmb_sample = "None"
cmb_sample = "Planck"
#cmb_sample = "ACT_D56"
#galaxy_sample = "BOSS_South" # goes with D56
#galaxy_sample = "BOSS_North" # goes with BN # slow
#galaxy_sample = "2MPZ" # both ACT maps
#galaxy_sample = "SDSS_L43D"
#galaxy_sample = "SDSS_L61D"
#galaxy_sample = "SDSS_L43"
#galaxy_sample = "SDSS_L61"
#galaxy_sample = "SDSS_L79"
#galaxy_sample = "SDSS_all"
#galaxy_sample = "SDSS_S16ILC"
galaxy_sample = "MGS"
print(f"Producing: {galaxy_sample:s}_{cmb_sample:s}")

# filename of galaxy map
if galaxy_sample == "2MPZ":
    gal_fn = "../galaxy_tests/2mpz_data/2MPZ.fits"
elif galaxy_sample == "BOSS_North":
    gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_North.fits"
elif galaxy_sample == "BOSS_South":
    gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_South.fits"
elif galaxy_sample == "MGS":
    gal_fn = "/mnt/marvin1/boryanah/2MPZ_vel/sdss_data/post_catalog.dr72bright0.fits"
elif "SDSS" in galaxy_sample:
    gal_fn = "../galaxy_tests/sdss_data/V21_DR15_Catalog_v4.txt"
    if "all" in galaxy_sample or "S16ILC" in galaxy_sample:
        L_lo = 0.
        L_hi = 1e20
    if "L43" in galaxy_sample:
        L_lo = 4.3e10
        if "D" in galaxy_sample:
            L_hi = 6.1e10
        else:
            L_hi = 1.e20
    elif "L61" in galaxy_sample:
        L_lo = 6.1e10
        if "D" in galaxy_sample:
            L_hi = 7.9e10
        else:
            L_hi = 1.e20
    elif "L79" in galaxy_sample:
        L_lo = 7.9e10
        L_hi = 1.e20
    
# filename of CMB map
if cmb_sample == "ACT_BN":
    fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"

elif cmb_sample == "ACT_D56":
    fn = "../cmb_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"

elif cmb_sample == "Planck":
    fn = "/mnt/marvin1/boryanah/2MPZ_vel//cmb_data/Planck_COM_CMB_IQU-smica_2048_R3.00_uK.fits" # healpy (because no "Planck_"), galactic
    msk_fn = "/mnt/marvin1/boryanah/2MPZ_vel//cmb_data/Planck_HFI_Mask_PointSrc_Gal70.fits" # combined healpy (equatorial), galactic

    
if cmb_sample is not "None":
    # reading fits file
    mp = enmap.read_fits(fn)
    msk = enmap.read_fits(msk_fn)

    
    # save map
    save = 0
    if save:
        fig_name = (fn.split('/')[-1]).split('.fits')[0]
        eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
        eshow(msk, fig_name+"_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
        plt.close()

    # map info
    #print("box (map) = ", enmap.box(mp.shape, mp.wcs)/utils.degree)
    decfrom = np.rad2deg(mp.box())[0, 0]
    rafrom = np.rad2deg(mp.box())[0, 1]
    decto = np.rad2deg(mp.box())[1, 0]
    rato = np.rad2deg(mp.box())[1, 1]
    rafrom = rafrom%360.
    rato = rato%360.
    print("decfrom, decto, rafrom, rato = ", decfrom, decto, rafrom, rato)

# size of the clusters and median z of 2MPZ
if galaxy_sample == '2MPZ':
    goal_size = 1.1*h # Mpc/h (comoving)
    zmed = 0.08
elif 'BOSS' in galaxy_sample:
    goal_size = 1.1*h # Mpc/h (comoving)
    zmed = 0.5
elif 'MGS' in galaxy_sample:
    goal_size = 1.1*h # Mpc/h (comoving)
    zmed = 0.5
elif 'SDSS' in galaxy_sample:
    goal_size = 1.1*h # Mpc/h (comoving)
    zmed = 0.5
sigma_z = 0.01

# create instance of the class "Class"
Cosmo = Class()
param_dict = {"h": h, "Omega_Lambda": Omega_Lambda, "Omega_cdm": Omega_cdm, "Omega_b": Omega_b}
Cosmo.set(param_dict)
Cosmo.compute()

# compute angular size distance at median z of 2MPZ
D_A_zmed = Cosmo.luminosity_distance(zmed)/(1.+zmed)**2 # Mpc/h
theta_arcmin_zmed = goal_size/D_A_zmed / utils.arcmin
#theta_arcmin_zmed = (0.5*goal_size/D_A_zmed) / utils.arcmin 
print("theta_arcmin_zmed = ", theta_arcmin_zmed)

# load 2MPZ data
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
elif 'MGS' in galaxy_sample:
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
    choice = (L_lo < L) & (L_hi >= L)
    if "S16ILC" in galaxy_sample:
        choice &= data['S16ILC'] == 1.
    print("galaxies satisfying luminosty cut = ", np.sum(choice))
    #choice = np.ones(len(Z), dtype=bool)
print("Zmin/max/med = ", Z.min(), Z.max(), np.median(Z))
print("RAmin/RAmax = ", RA.min(), RA.max())
print("DECmin/DECmax = ", DEC.min(), DEC.max())

if cmb_sample is not "None":
    # make magnitude and RA/DEC cuts to  match ACT
    DEC_choice = (DEC <= decto) & (DEC > decfrom)
    if cmb_sample == 'ACT_D56':
        RA_choice = (RA <= rafrom) | (RA > rato)
    elif cmb_sample == 'ACT_BN':
        RA_choice = (RA <= rafrom) & (RA > rato)
    else:
        RA_choice = np.ones_like(DEC_choice)
    choice &= DEC_choice & RA_choice
RA = RA[choice]
DEC = DEC[choice]
Z = Z[choice]

# transfer into pixell coordinates
RA[RA > 180.] -= 360.

if cmb_sample is "None":
    box = np.array([[DEC.min(), RA.max()],[DEC.max(), RA.min()]]) * utils.degree
    shape, wcs = enmap.geometry(pos=box, res=0.5 * utils.arcmin)#, proj='car')
else:
    shape, wcs = mp.shape, mp.wcs
    #imap = enmap.zeros((3,) + shape, wcs=wcs)

# compute the aperture photometry for each galaxy
r = theta_arcmin_zmed * utils.arcmin
srcs = ([DEC*utils.degree, RA*utils.degree])
mask = enmap.distance_from(shape, wcs, srcs, rmax=r) >= r
eshow(mask, f'{galaxy_sample}_{cmb_sample}_galaxies', **{"colorbar":True, "ticks": 5, "downgrade": 4})
plt.close()
