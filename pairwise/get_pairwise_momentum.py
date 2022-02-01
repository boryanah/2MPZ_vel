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
    fn = "../cmb_tests/cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits" # BN
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
Z = Z[choice]
CX = CX[choice]
CY = CY[choice]
CZ = CZ[choice]
index = index[choice]
print("number of galaxies = ", np.sum(choice))

# stack together normalized positions
P = np.vstack((CX, CY, CZ)).T

# comoving distance to observer and angular size
CD = np.zeros(len(CX))
D_A = np.zeros(len(CX))
for i in range(len(CX)):
    if Z[i] < 0.: continue
    lum_dist = Cosmo.luminosity_distance(Z[i])
    CD[i] = lum_dist/(1.+Z[i]) # Mpc/h
    D_A[i] = lum_dist/(1.+Z[i])**2 # Mpc/h
P = P*CD[:, None]

# scale aperture radius with redshift
theta_arcmin = theta_arcmin_zmed*(D_A/D_A_zmed)
print("aperture radius min/max = ", theta_arcmin.min(), theta_arcmin.max())

show_theta = False
if show_theta:
    import plotparams
    plotparams.buba()
    plt.figure(figsize=(9, 7))
    tbins = np.linspace(theta_arcmin.min(), theta_arcmin.max(), 51)
    tbinc = (tbins[1:]+tbins[:-1])*.5
    hist, _ = np.histogram(theta_arcmin, bins=tbins)
    plt.plot(tbinc, hist)
    plt.xlabel(r'$\Theta$')
    plt.savefig("figs/theta_hist.png")
    plt.show()

# size of the stamps divided by two
rApMaxArcmin = np.ceil(theta_arcmin*np.sqrt(2.)) # since sqrt2 is the max radius
print("max radius of outer flux ceiled = ", rApMaxArcmin.min(), rApMaxArcmin.max())

delta_T_fn = f"data/{galaxy_sample:s}_{cmb_sample:s}_delta_Ts.npy"
if os.path.exists(delta_T_fn):
    delta_Ts = np.load(delta_T_fn)
    index_new = np.load(f"../pairwise/data/{galaxy_sample}_{cmb_sample}_index.npy")
    choice = np.in1d(index, index_new)
    Z = Z[choice]
    P = P[choice]
else:
    # compute the aperture photometry for each galaxy
    T_APs = np.zeros(len(RA))
    zeros_everywhere = 0
    for i in range(len(RA)):
        if i % 1000 == 0: print("i = ", i)
        ra = RA[i]
        dec = DEC[i]
        if ra > 180.: ra -= 360. # option 0

        # extract stamp
        opos, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=rApMaxArcmin[i], resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)

        # skip if mask is zero everywhere
        if np.sum(stampMask) == 0.: zeros_everywhere += 1; continue
        
        # record T_AP
        T_APs[i] = calc_T_AP(stampMap, theta_arcmin[i], mask=stampMask)

    # apply cuts because of masking
    choice = T_APs != 0.
    index = index[choice]
    Z = Z[choice]
    T_APs = T_APs[choice]
    P = P[choice]
    np.save(f"../pairwise/data/{galaxy_sample}_{cmb_sample}_index.npy", index)
    
    # get the redshift-weighted one 
    # fast and elegant but resource-consuming
    #Ws = np.exp(-(Z[:, None]-Z[None, :])**2./(2.*sigma_z**2.))
    #bar_T_APs = np.einsum('i,ij', T_APs, Ws)/np.sum(Ws, axis=1)
    # slow and not elegant
    bar_T_APs = np.zeros(len(T_APs))
    for i in range(len(T_APs)):
        z = Z[i]
        w = np.exp(-(z-Z)**2./(2.*sigma_z**2.))
        bar_T_APs[i] = np.sum(T_APs*w)/np.sum(w)
    
    # final quantity: temperature decrement around each galaxy
    delta_Ts = T_APs - bar_T_APs

    np.save(delta_T_fn, delta_Ts)
    print("percentage of objects for which the mask is zeros everywhere = ", zeros_everywhere*100./len(RA))

# define bins in Mpc/h
rbins = np.linspace(0., 100., 16)
rbinc = (rbins[1:]+rbins[:-1])*.5
is_log_bin = False
nthread = 32 # 64 threads and 32 cores
boxsize = 0. # this is not used since we are on the sky

# change dtype
dtype = np.float32
P = P.astype(dtype)
delta_Ts = delta_Ts.astype(dtype)
assert P.shape[0] == len(delta_Ts)

if want_jackknife:
    # number of jackknife samples
    n_jack = 100

    # initialize
    PV_2D = np.zeros((len(rbinc), n_jack))
    DD_2D = np.zeros((len(rbinc), n_jack))
    inds = np.arange(len(delta_Ts))
    np.random.shuffle(inds)
    n_jump = len(inds)//n_jack
    for i in range(n_jack):
        # calculate the pairwise velocity
        print("starting...")
        t1 = time.time()
        new_inds = np.delete(inds, np.arange(n_jump*i, n_jump*(i+1)))
        DD, PV = pairwise_momentum(P[new_inds], delta_Ts[new_inds], rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        print("done!")
        print(time.time()-t1)
        PV_2D[:, i] = PV
        DD_2D[:, i] = DD

    # compute errorbars
    DD_mean = np.mean(DD_2D, axis=1)
    PV_mean = np.mean(PV_2D, axis=1)    
    DD_err = np.sqrt(n_jack-1)*np.std(DD_2D, axis=1)
    PV_err = np.sqrt(n_jack-1)*np.std(PV_2D, axis=1)

    # save arrays
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_DD_mean.npy", DD_mean)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_PV_mean.npy", PV_mean)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_DD_err.npy", DD_err)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_PV_err.npy", PV_err)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_rbinc.npy", rbinc)
else:
    print("starting...")
    t1 = time.time()
    # calculate the pairwise velocity
    DD, PV = pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
    print("done!")
    print(time.time()-t1)
        
    # save arrays
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_DD.npy", DD)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_PV.npy", PV)
    np.save(f"data/{galaxy_sample:s}_{cmb_sample}_rbinc.npy", rbinc)

    
# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.zeros(len(PV)), 'k--')
plt.plot(rbinc, PV)
plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu{\rm K}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.show()

"""
dtype=(numpy.record, [('RA', '>f8'), ('DEC', '>f8'), ('RUN', '>i4'), ('RERUN', 'S12'), ('CAMCOL', '>i4'), ('FIELD', '>i4'), ('ID', '>i4'), ('ICHUNK', '>i4'), ('IPOLY', '>i4'), ('ISECT', '>i4'), ('FRACPSF', '>f4', (5,)), ('EXPFLUX', '>f4', (5,)), ('DEVFLUX', '>f4', (5,)), ('PSFFLUX', '>f4', (5,)), ('MODELFLUX', '>f4', (5,)), ('FIBER2FLUX', '>f4', (5,)), ('R_DEV', '>f4', (5,)), ('EXTINCTION', '>f4', (5,)), ('PSF_FWHM', '>f4', (5,)), ('AIRMASS', '>f4'), ('SKYFLUX', '>f4', (5,)), ('EB_MINUS_V', '>f4'), ('IMAGE_DEPTH', '>f4', (5,)), ('IMATCH', '>i4'), ('Z', '>f4'), ('WEIGHT_FKP', '>f4'), ('WEIGHT_CP', '>f4'), ('WEIGHT_NOZ', '>f4'), ('WEIGHT_STAR', '>f4'), ('WEIGHT_SEEING', '>f4'), ('WEIGHT_SYSTOT', '>f4'), ('NZ', '>f4'), ('COMP', '>f4'), ('PLATE', '>i4'), ('FIBERID', '>i4'), ('MJD', '>i4'), ('FINALN', '>i4'), ('TILE', '>i2', (3,)), ('SPECTILE', '>i4'), ('ICOLLIDED', '>i4'), ('INGROUP', '>i4'), ('MULTGROUP', '>i4')]))
>>> hdul[1].data['RA'].shape
(230831,)
"""
