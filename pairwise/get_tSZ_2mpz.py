import os
import time

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class
from astropy.io import fits

from util import *
from estimator import pairwise_momentum

# TODO: masking and coordinates and is this a typo T_AB bar

# filename of CMB map
galaxy_sample = "2MPZ"
cmb_sample = "ACT_BN"
if cmb_sample == "ACT_BN":
    #fn = "../CMB_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
    fn = "../CMB_tests/cmb_data/tilec_single_tile_BN_comptony_map_v1.2.0_joint.fits" # BN
elif cmb_sample == "ACT_D56":
    fn = "../CMB_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56

# reading fits file
mp = enmap.read_fits(fn)
if cmb_sample == "ACT_BN":
    pass #mp = mp[600:-600, 1500:-1500] # BN
elif cmb_sample == "ACT_D56":
    pass
    
# save map
save = 0
if save:
    fig_name = (fn.split('/')[-1]).split('.fits')[0]
    eshow(mp, fig_name, **{"colorbar":True, "range": 300, "ticks": 5, "downgrade": 4})
    plt.close()

# settings for pixell
#rApMinArcmin = 1. # minimum aperture for bins
rApMaxArcmin = 10. # maximum aperture for bins
resCutoutArcmin = 0.25 # stamp resolution
projCutout = 'cea' # projection

# map info
print("shape and dtype = ", mp.shape, mp.dtype)
print("wcs = ", mp.wcs)
print("box = ", enmap.box(mp.shape, mp.wcs)/utils.degree)
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
goal_size = 1.1 # Mpc/h
zmed = 0.08
sigma_z = 0.01
want_shuffle = False

# create instance of the class "Class"
Cosmo = Class()

# pass cosmology parameters
h = 0.7
param_dict = {"h": h, "Omega_Lambda": 0.7, "Omega_cdm": 0.25, "Omega_b": 0.05}
Cosmo.set(param_dict)

# run class
Cosmo.compute()

# compute angular size distance at median z of 2MPZ
D_A_zmed = Cosmo.luminosity_distance(zmed)/(1.+zmed)**2 # Mpc/h
theta_arcmin = goal_size/2./D_A_zmed / utils.arcmin
print("theta_arcmin = ", theta_arcmin)

# load 2MPZ data
hdul = fits.open("../2MPZ_tests/2mpz_data/2MPZ.fits")
RA = hdul[1].data['RA'].flatten()/utils.degree # 0, 360
DEC = hdul[1].data['DEC'].flatten()/utils.degree # -180, 180
K_rel = hdul[1].data['KCORR'].flatten()
Z_photo = hdul[1].data['ZPHOTO'].flatten()
CX = hdul[1].data['CX'].flatten()
CY = hdul[1].data['CY'].flatten()
CZ = hdul[1].data['CZ'].flatten()
index = np.arange(len(Z_photo), dtype=int)
print("RAmin/RAmax = ", RA.min(), RA.max())
print("DECmin/DECmax = ", DEC.min(), DEC.max())

# shuffling
if want_shuffle:
    inds = np.arange(len(RA), dtype=int)
    np.random.shuffle(inds)
    RA = RA[inds]

# make magnitude and RA/DEC cuts to  match ACT
K_choice = (K_rel < 13.9) # original is 13.9
DEC_choice = (DEC <= decto) & (DEC > decfrom)
if cmb_sample == "ACT_D56":
    RA_choice = (RA <= rafrom) | (RA > rato) # option 0
elif cmb_sample == "ACT_BN":
    RA_choice = (RA <= rafrom) & (RA > rato) # option 0
choice = K_choice & DEC_choice & RA_choice
RA = RA[choice]
DEC = DEC[choice]
Z_photo = Z_photo[choice]
CX = CX[choice]
CY = CY[choice]
CZ = CZ[choice]
print("number of galaxies = ", np.sum(choice))

# Write code to get stack

print(mp.shape)
stack = 0
for i in range(len(RA)):
    if i % 1000 == 0: print("i = ", i)
    #if i == 10: break
    ra = RA[i]
    dec = DEC[i]
    # Extract stamps by reprojecting to a tangent plane projection
    if ra > 180.: ra -= 360. # option 0

    #stampMap = reproject.thumbnails(mp, coords=[ra, dec], r=20., res=0.5, order=1)
    # extract stamp
    opos, stampMap = extractStamp(mp, ra, dec, rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False)
    if stampMap is None: continue 
    #print(stampMap.shape)
    #print(stack)
    stack += stampMap
stack /= len(RA)
plt.imshow(stack)
plt.savefig(f"figs/{galaxy_sample}_{cmb_sample:s}_tSZ_stack.png")
plt.show()

plots = enplot.plot(enmap.upgrade(stack, 5), grid=True)
enplot.write("figs/tSZ", plots)

