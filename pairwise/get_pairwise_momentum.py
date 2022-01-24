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
ACT_type = "BN"
#ACT_type = "D56"
if ACT_type == "BN":
    fn = "../CMB_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
elif ACT_type == "D56":
    fn = "../CMB_tests/cmb_data/tilec_single_tile_D56_cmb_map_v1.2.0_joint.fits" # D56

# reading fits file
mp = enmap.read_fits(fn)
if ACT_type == "BN":
    mp = mp[600:-600, 1500:-1500] # BN
elif ACT_type == "D56":
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
#rafrom = (180.-rafrom)%360. # option 1
#rato = (180.-rato)%360. # option 1
rafrom = (-rafrom)%360. # option 2
rato = (-rato)%360. # option 2
print("decfrom, decto, rafrom, rato = ", decfrom, decto, rafrom, rato)

# size of the clusters and median z of 2MPZ
goal_size = 1.1 # Mpc/h
zmed = 0.08
sigma_z = 0.01

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

# make magnitude and RA/DEC cuts to  match ACT
K_choice = (K_rel < 11.0) # original is 13.9
DEC_choice = (DEC <= decto) & (DEC > decfrom)
#RA_choice = (RA <= rato) | (RA > rafrom) # option 1
RA_choice = (RA <= rato) & (RA > rafrom) # option 2
choice = K_choice & DEC_choice & RA_choice
np.save(f"../pairwise/data/2MPZ_index_{ACT_type:s}.npy", index[choice])
RA = RA[choice]
DEC = DEC[choice]
Z_photo = Z_photo[choice]
CX = CX[choice]
CY = CY[choice]
CZ = CZ[choice]
print("number of galaxies = ", np.sum(choice))

# compute the aperture photometry for each galaxy
T_APs = np.zeros(len(RA))
for i in range(len(RA)):
    if i % 1000 == 0: print("i = ", i)
    ra = RA[i]
    dec = DEC[i]
    #ra = (180.-ra) % 360.# - 360. # option 1
    ra = (-ra) % 360.# - 360. # option 2

    # extract stamp
    opos, stampMap = extractStamp(mp, ra, dec, rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False)

    # record T_AP
    T_APs[i] = calc_T_AP(stampMap, theta_arcmin)

# get the redshift-weighted one 
Ws = np.exp(-(Z_photo[:, None]-Z_photo[None, :])**2./(2.*sigma_z**2.))
bar_T_APs = np.einsum('i,ij', T_APs, Ws)/np.sum(Ws, axis=1)

# final quantity: temperature decrement around each galaxy
delta_Ts = T_APs - bar_T_APs
np.save("data/delta_Ts.npy", delta_Ts)

# stack together normalized positions
P = np.vstack((CX, CY, CZ)).T

# comoving distance to observer
CD = np.zeros(len(CX))
for i in range(len(CX)):
    if i%1000 == 0: print(i, len(CX)-1)
    if Z_photo[i] < 0.: continue
    CD[i] = Cosmo.luminosity_distance(Z_photo[i])/(1.+Z_photo[i]) # Mpc/h
P = P*CD[:, None]

# define bins in Mpc/h
rbins = np.linspace(0., 100., 11)
rbinc = (rbins[1:]+rbins[:-1])*.5
is_log_bin = False
nthread = 32 # 64 threads and 32 cores
boxsize = 0. # this is not used since we are on the sky

# change dtype
dtype = np.float32
P = P.astype(dtype)
delta_Ts = delta_Ts.astype(dtype)

# calculate the pairwise velocity
print("starting...")
t1 = time.time()
DD, PV = pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
print("done!")
print(time.time()-t1)

# save arrays
galaxy_sample = f"2MPZ_ACT_{ACT_type:s}"
np.save(f"data/{galaxy_sample:s}_DD.npy", DD)
np.save(f"data/{galaxy_sample:s}_PV.npy", PV)
np.save("data/rbinc.npy", rbinc)

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc, PV)
plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu{\rm K}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.show()
