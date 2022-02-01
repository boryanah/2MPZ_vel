import time
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
import healpy as hp
from classy import Class

from estimator import pairwise_velocity_sky as pairwise_velocity
import plotparams
plotparams.buba()

galaxy_sample = "2mpz"
#galaxy_sample = "mock_2015"
#galaxy_sample = "mock_box"
#galaxy_sample = "mock_rsd_box"
#galaxy_sample = "random"

if galaxy_sample == "2mpz":
    # load mock data
    hdul = fits.open("../2MPZ_tests/2mpz_data/2MPZ.fits")
    B = hdul[1].data['B'].flatten() # -90, 90
    L = hdul[1].data['L'].flatten() # 0, 360
    theta = (90.-B)*np.pi/180.
    phi = L.copy()*np.pi/180.
    print("Lmin/Lmax = ", L.min(), L.max())
    print("Bmin/Bmax = ", B.min(), B.max())

    B *= np.pi/180.
    L *= np.pi/180.
    CX = np.cos(B)*np.cos(L)
    CY = np.cos(B)*np.sin(L)
    CZ = np.sin(B)
    
    K_rel = hdul[1].data['KCORR'].flatten()
    Z_photo = hdul[1].data['ZPHOTO'].flatten()

    # create instance of the class "Class"
    Cosmo = Class()

    # pass input parameters
    param_dict = {"h": 0.6736, "Omega_Lambda": 0.7, "Omega_cdm": 0.25, "Omega_b": 0.05}
    Cosmo.set(param_dict)

    # run class
    Cosmo.compute()

    # stack together normalized positions
    P = np.vstack((CX, CY, CZ)).T

    # comoving distance to observer
    fn_CD = "../2MPZ_tests/2mpz_data/CD.npy"
    if os.path.exists(fn_CD):
        CD = np.load(fn_CD)
    else:
        CD = np.zeros_like(CX)
        for i in range(len(CX)):
            if i%10000 == 0: print(i)
            if Z_photo[i] < 0.: continue
            CD[i] = Cosmo.luminosity_distance(Z_photo[i])/(1.+Z_photo[i])
        np.save(fn_CD, CD)
    P = P*CD[:, None]

    # velocity along line of sight
    fn_V_los = "../2MPZ_tests/2mpz_data/V_los.npy"
    if os.path.exists(fn_V_los):
        V_los = np.load(fn_V_los)
    else:
        # load CMB temperature map
        hdul = fits.open("../CMB_tests/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits")
        temp = hdul[1].data['I_STOKES']
        nest = True
        nside = 2048
        npix = hp.nside2npix(nside)
        ring2nest = hp.ring2nest(nside, np.arange(npix)) # tells me for each pixel in RING, where do I find the pixel in NESTED
        temp = temp[ring2nest]; nest = False # convert to RING
        
        # define high-pass filter
        ls = np.arange(3*nside)
        hp_filter = np.zeros_like(ls)
        ell_filter = 3000
        hp_filter[ls > ell_filter] = 1

        # apply high-pass filter
        alms_temp = hp.sphtfunc.map2alm(temp, pol=False) # expects input in RING
        alms_temp_high = hp.sphtfunc.almxfl(alms_temp, hp_filter)
        temp_high = hp.sphtfunc.alm2map(alms_temp_high, nside) # returns only in RING

        """
        # ACT MAP
        fn = "cmb_data/tilec_single_tile_BN_cmb_map_v1.2.0_joint.fits"
        lmax = 6000
        mp = enmap.read_fits(fn)
        shape = mp.shape
        npix = enmap.npix(shape)
        nside = curvedsky.npix2nside(npix) # ~2000
        # aperture photometry 100,000 galaxies
        # pixell cut out circles cluster studies 
        mp = reproject.healpix_from_enmap(mp, lmax=lmax, nside=nside)
        """
        #hp.mollview(temp_high, min=-5.e-5, max=5.e-5)
        #plt.show()
        
        # for each galaxy in 2MPZ, find pixel
        ipix = hp.pixelfunc.vec2pix(nside, CX, CY, CZ, nest=nest)
        #ipix2 = hp.ang2pix(nside, theta, phi, nest=nest) # have checked that this agrees with ipix when using B, L
        # take value of pixel from CMB high-pass map and name V_los
        V_los = temp_high[ipix]
        np.save(fn_V_los, V_los)
    
    # let's make cuts to reduce number of objects
    choice = (K_rel < 13.0) # 260,000 gals, 6 mins; original cut is 13.9, but we are currently challenged computationally
    V_los = V_los[choice]
    P = P[choice]
    print("number of galaxies = ", np.sum(choice))


elif galaxy_sample == "mock_2015":
    # load mock data
    hdul = fits.open("../2MPZ_tests/mock_data/mock_1024_002_HAMonly_noMW.fits")
    X = hdul[1].data['PX']
    Y = hdul[1].data['PY']
    Z = hdul[1].data['PZ']
    VX = hdul[1].data['VX']
    VY = hdul[1].data['VY']
    VZ = hdul[1].data['VZ']
    K_rel = hdul[1].data['K_REL']

    # comoving distance to observer
    P = np.vstack((X, Y, Z)).T
    dist = np.sqrt(np.sum(P**2, axis=1)) # min 1 Mpc/h and max 1050 Mpc/h, comes from putting three boxes in each dimension at z = 0 next to each other

    # LOS velocity, defined as hat P dot v
    V_los = (X*VX + Y*VY + Z*VZ)/dist

    # let's make cuts to reduce number of objects
    #choice = K_rel < 12.5 # original cut is 13.9, but we are currently challenged computationally
    #choice = K_rel < 12. # original cut is 13.9, but we are currently challenged computationally
    #choice = (K_rel < 13.5) & (K_rel > 13.0) # original cut is 13.9, but we are currently challenged computationally
    choice = (K_rel < 13.5) # 2417 secs, 634481 gals
    V_los = V_los[choice]
    P = P[choice]
    print("number of galaxies = ", np.sum(choice))

elif galaxy_sample == "mock_box":
    Lbox = 2000.
    P = np.load("mock_data/pos_gal_down.npy")
    V = np.load("mock_data/vel_gal_down.npy")

    # downsample cause computationally challenged
    inds = np.arange(P.shape[0])
    np.random.shuffle(inds)
    inds = inds[:300000]
    P = P[inds]
    V = V[inds]
    
    X = P[:, 0]; Y = P[:, 1]; Z = P[:, 2]
    VX = V[:, 0]; VY = V[:, 1]; VZ = V[:, 2]
    
    # comoving distance to observer
    dist = np.sqrt(np.sum(P**2, axis=1)) # min 1 Mpc/h and max 1050 Mpc/h, comes from putting three boxes in each dimension at z = 0 next to each other

    # LOS velocity, defined as hat P dot v
    V_los = (X*VX + Y*VY + Z*VZ)/dist
    
elif galaxy_sample == "mock_rsd_box":
    Lbox = 2000.
    P = np.load("mock_data/pos_gal_rsd_down.npy")
    V = np.load("mock_data/vel_gal_rsd_down.npy")

    # downsample cause computationally challenged
    inds = np.arange(P.shape[0])
    np.random.shuffle(inds)
    inds = inds[:300000]
    P = P[inds]
    V = V[inds]
    
    X = P[:, 0]; Y = P[:, 1]; Z = P[:, 2]
    VX = V[:, 0]; VY = V[:, 1]; VZ = V[:, 2]

    
    
    # comoving distance to observer
    dist = np.sqrt(np.sum(P**2, axis=1)) # min 1 Mpc/h and max 1050 Mpc/h, comes from putting three boxes in each dimension at z = 0 next to each other
    
    # LOS velocity, defined as hat P dot v
    V_los = (X*VX + Y*VY + Z*VZ)/dist

elif galaxy_sample == "random":
    Lbox = 2000.
    N = 10000
    X = np.random.rand(N)*Lbox-Lbox/2.
    Y = np.random.rand(N)*Lbox-Lbox/2.
    Z = np.random.rand(N)*Lbox-Lbox/2.
    V_typical = 200.
    VX = np.random.rand(N)*V_typical-V_typical/2.
    VY = np.random.rand(N)*V_typical-V_typical/2.
    VZ = np.random.rand(N)*V_typical-V_typical/2.

    # comoving distance to observer
    P = np.vstack((X, Y, Z)).T
    dist = np.sqrt(np.sum(P**2, axis=1)) # min 1 Mpc/h and max 1050 Mpc/h, comes from putting three boxes in each dimension at z = 0 next to each other

    # LOS velocity, defined as hat P dot v
    V_los = (X*VX + Y*VY + Z*VZ)/dist
    
save_nz = False
if save_nz:
    fsky = 1.#0.8
    cbins = np.linspace(1., 1050., 101)
    cbinc = (cbins[1:] + cbins[:-1]) *.5
    hist, _ = np.histogram(dist, bins=cbins)
    hist = hist/fsky
    hist = hist.astype(int)

    plt.plot(cbinc, hist)
    plt.yscale('log')
    plt.show()

    np.save(f"data_nz/{galaxy_sample:s}_Nz.npy", hist)
    np.save(f"data_nz/{galaxy_sample:s}_cbins.npy", cbins)
    print(hist)
    print(np.sum(hist))


# define bins
rbins = np.linspace(0., 30., 90)
rbinc = (rbins[1:]+rbins[:-1])*.5
is_log_bin = False
periodic = False
nthread = 32 # 64 threads and 32 cores
boxsize = 0. # this is not used since we are on the sky

# change dtype
dtype = np.float32
P = P.astype(dtype)
V_los = V_los.astype(dtype)

# run estimator
print("starting...")
t1 = time.time()
DD, PV = pairwise_velocity(P, V_los, boxsize, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
print("done!")
print(time.time()-t1)

np.save(f"data_pairwise/{galaxy_sample:s}_DD.npy", DD)
np.save(f"data_pairwise/{galaxy_sample:s}_PV.npy", PV)
np.save("data_pairwise/rbinc.npy", rbinc)

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc, PV/100.)
plt.ylabel(r"$v_{12}(r) \ [100 {\rm km}/s]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.show()


show_nz = False
if show_nz:
    # get redshift distribution (same as in paper)
    zbins = np.linspace(0, 0.4, 41)
    zbinc = (zbins[1:]+zbins[:-1])*.5
    hist_z, _ = np.histogram(hdul[1].data['Z_TRUE'], bins=zbins)

    plt.figure(figsize=(9, 7))
    plt.plot(zbinc, hist_z)
    plt.yscale('log')
    plt.show()



"""
hdul.info()
hdul[1].data
dtype=(numpy.record, [('ID', '>i8'), ('RA', '>f4'), ('DEC', '>f4'), ('Z_OBS', '>f4'), ('Z_TRUE', '>f4'), ('Z_COSMO', '>f4'), ('K_REL', '>f4'), ('K_ABS', '>f4'), ('PX', '>f4'), ('PY', '>f4'), ('PZ', '>f4'), ('VX', '>f4'), ('VY', '>f4'), ('VZ', '>f4')]))
hdul[1].data['Z_TRUE'].shape # (1285393,)
hdul[1].data['Z_TRUE'].min() # -0.0007482307
hdul[1].data['Z_TRUE'].max() # 0.38544536
hdul[1].data['K_REL'].max() # 13.999999
hdul[1].data['K_REL'].min() # 3.9206426
np.sum(hdul[1].data['K_REL'] < 13.9) # 1116309
np.sum(hdul[1].data['K_REL'] < 13.5) # 634481
hdul[1].data['K_ABS'].min() # -26.187534
hdul[1].data['K_ABS'].max() # -17.0
"""
