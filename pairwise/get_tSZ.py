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
rApMaxArcmin = 10. # maximum aperture for bins
#resCutoutArcmin = 0.5 # stamp resolution
resCutoutArcmin = 0.25 # stamp resolution
projCutout = 'cea' # projection
want_jackknife = False
h = 0.7; Omega_Lambda = 0.7; Omega_cdm = 0.25; Omega_b = 0.05

# galaxy and cmb sample
cmb_sample = "ACT_BN"
#cmb_sample = "ACT_D56"
#galaxy_sample = "BOSS_South" # goes with D56
#galaxy_sample = "BOSS_North" # goes with BN # slow
#galaxy_sample = "2MPZ" # both ACT maps
#galaxy_sample = "SDSS_L43D"
#galaxy_sample = "SDSS_L61D"
#galaxy_sample = "SDSS_L43"
galaxy_sample = "SDSS_L61"
#galaxy_sample = "SDSS_L79"
print(f"Producing: {galaxy_sample:s}_{cmb_sample:s}")

# filename of galaxy map
if galaxy_sample == "2MPZ":
    gal_fn = "../galaxy_tests/2mpz_data/2MPZ.fits"
elif galaxy_sample == "BOSS_North":
    gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_North.fits"
elif galaxy_sample == "BOSS_South":
    gal_fn = "../galaxy_tests/boss_data/galaxy_DR12v5_CMASS_South.fits"
elif "SDSS" in galaxy_sample:
    gal_fn = "../galaxy_tests/sdss_data/V21_DR15_Catalog_v4.txt"
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
    
# filename of CMB map
if cmb_sample == "ACT_BN":
    #fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
    fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_comptony_map_v1.2.0_joint.fits" # BN
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_BN_compsep_mask.fits"
elif cmb_sample == "ACT_D56":
    fn = "../cmb_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56
    msk_fn = "../cmb_tests/cmb_data/act_dr4.01_s14s15_D56_compsep_mask.fits"
    
# reading fits file
mp = enmap.read_fits(fn)
msk = enmap.read_fits(msk_fn)
if cmb_sample == "ACT_BN":
    pass
elif cmb_sample == "ACT_D56":
    pass
    
# save map
save = 0
if save:
    fig_name = (fn.split('/')[-1]).split('.fits')[0]
    #eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
    eshow(msk, fig_name+"_mask", **{"colorbar":True, "ticks": 5, "downgrade": 4})
    plt.close()

# map info
#print("shape and dtype = ", mp.shape, mp.dtype)
#print("wcs = ", mp.wcs)
#print("box (map) = ", enmap.box(mp.shape, mp.wcs)/utils.degree)
#print("box (msk) = ", enmap.box(msk.shape, msk.wcs)/utils.degree)
decfrom = np.rad2deg(mp.box())[0, 0]
rafrom = np.rad2deg(mp.box())[0, 1]
decto = np.rad2deg(mp.box())[1, 0]
rato = np.rad2deg(mp.box())[1, 1]
rafrom = rafrom%360. # option 0
rato = rato%360. # option 0
#rafrom = (180.-rafrom)%360. # option 1
#rato = (180.-rato)%360. # option 1
#rafrom = (-rafrom)%360. # option 2
#rato = (-rato)%360. # option 2
print("decfrom, decto, rafrom, rato = ", decfrom, decto, rafrom, rato)

# size of the clusters and median z of 2MPZ
if galaxy_sample == '2MPZ':
    goal_size = 1.5*h # Mpc/h
    zmed = 0.08
elif 'BOSS' in galaxy_sample:
    goal_size = 1.1*h # Mpc/h
    zmed = 0.5
elif 'SDSS' in galaxy_sample:
    goal_size = 1.1*h # Mpc/h
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
elif 'SDSS' in galaxy_sample:
    data = ascii.read(gal_fn)
    RA = data['ra'] # 0, 360
    DEC = data['dec'] # -180, 180 # -10, 36
    Z = data['z']
    L = data['lum']
    choice = (L_lo < L) & (L_hi >= L) 
    #choice = np.ones(len(Z), dtype=bool)
print("Zmin/max/med = ", Z.min(), Z.max(), np.median(Z))
# transform to cartesian coordinates (checked)
CX = np.cos(RA*utils.degree)*np.cos(DEC*utils.degree)
CY = np.sin(RA*utils.degree)*np.cos(DEC*utils.degree)
CZ = np.sin(DEC*utils.degree)
index = np.arange(len(Z), dtype=int)
print("RAmin/RAmax = ", RA.min(), RA.max())
print("DECmin/DECmax = ", DEC.min(), DEC.max())

# make magnitude and RA/DEC cuts to  match ACT
DEC_choice = (DEC <= decto) & (DEC > decfrom)
if cmb_sample == 'ACT_D56':
    RA_choice = (RA <= rafrom) | (RA > rato)
elif cmb_sample == 'ACT_BN':
    RA_choice = (RA <= rafrom) & (RA > rato)
choice &= DEC_choice & RA_choice
RA = RA[choice]
DEC = DEC[choice]
print("number of galaxies = ", np.sum(choice))

stack = 0
count = 0
for i in range(len(RA)):
    if i % 1000 == 0: print("i = ", i)
    if i == 10000: break
    ra = RA[i]
    dec = DEC[i]
    # Extract stamps by reprojecting to a tangent plane projection
    if ra > 180.: ra -= 360. # option 0

    # extract stamp
    opos, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)
    if np.sum(stampMask) == 0.: continue
    stack += stampMap
    count += 1
stack /= count
print(count)
plt.imshow(stack)
plt.savefig(f"figs/{galaxy_sample}_{cmb_sample:s}_tSZ_stack.png")
plt.show()

plots = enplot.plot(enmap.upgrade(stack, 5), grid=True)
enplot.write("figs/tSZ", plots)
