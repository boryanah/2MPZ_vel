#!/usr/bin/env python3
"""
Usage
-----
$ ./multi_pairwise_momentum.py --help
TODO: tzav_each and asymm_pair missing from jackknife
"""
import os
import gc
import time
import argparse

import numpy as np
import random
import numba
numba.config.THREADING_LAYER = 'safe'
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class
import healpy as hp
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy import interpolate
from colossus.halo import splashback
from colossus.cosmology import cosmology

from util import extractStamp, calc_T_AP, eshow, get_tzav_fast, cutoutGeometry, gaussian_filter, tophat_filter, GNFW_filter, kSZ_filter, calc_T_MF
from estimator import pairwise_vel_asymm as pairwise_vel # pairwise_momentum, 
from numba_2pcf.cf import numba_pairwise_vel#, numba_2pcf
from tools_pairwise_momentum import load_cmb_sample, load_galaxy_sample
import kmeans_radec
from kmeans_radec import KMeans, kmeans_sample

from contextlib import closing
import multiprocessing
from itertools import repeat

# settings
DEFAULTS = {}
DEFAULTS['resCutoutArcmin'] = 0.05 # stamp resolution
DEFAULTS['projCutout'] = 'cea' # projection
DEFAULTS['galaxy_sample'] = "SDSS_L79"
DEFAULTS['cmb_sample'] = "ACT_BN"
DEFAULTS['error_estimate'] = "none"
DEFAULTS['n_sample'] = 100
DEFAULTS['mask_type'] = 'mtype0'
DEFAULTS['data_dir'] = "/mnt/marvin1/boryanah/2MPZ_vel/"

# cosmological parameters
wb = 0.02225
wc = 0.1198
s8 = 0.83
ns = 0.964
h = 67.3/100.
COSMO_DICT = {'h': h,
              'omega_b': wb,
              'omega_cdm': wc,
              'A_s': 2.100549e-09, # doesn't affect calculation
              #'sigma8': s8,
              'n_s': ns}
colossus_cosmo = {'flat': True, 'H0': h*100., 'Om0': (wb+wc)/h**2, 'Ob0': wb/h**2, 'sigma8': s8, 'ns': ns}
cosmo = cosmology.setCosmology('TNG_cosmo', colossus_cosmo)

def get_regions(mask, n_regions, nside, unassigned=hp.UNSEEN):
    """ Generates `n_regions` regions of roughly equal area
    for a given sky mask `mask`, assuming HEALPix "RING"
    ordering. Returns a HEALPix map where each pixel holds
    the index of the region it is assigned to. Unassigned
    pixels will take the `unassigned` value.
    """
    npix = len(mask)
    hp.npix2nside(npix)
    ipix = np.arange(npix)
    ra, dec = hp.pix2ang(nside, ipix, lonlat=True)
    goodpix = mask > 0
    km = kmeans_sample(np.array([ra[goodpix], dec[goodpix]]).T,
                       n_regions, maxiter=100, tol=1.0e-5,
                       verbose=False)
    map_ids = np.full(npix, unassigned)
    map_ids[ipix[goodpix]] = km.labels
    return map_ids

def get_premask(RA, DEC, msk):
    """
    Given RA, DEC of the galaxies, compute premask
    """
    coords = np.deg2rad(np.array((DEC, RA)))
    ypix, xpix = enmap.sky2pix(msk.shape, msk.wcs, coords)
    print("xpix min/max, mask_x = ", xpix.min(), xpix.max(), msk.shape[1])
    print("ypix min/max, mask_y = ", ypix.min(), ypix.max(), msk.shape[0])
    xpix, ypix = xpix.astype(int), ypix.astype(int)
    premask = np.zeros(len(RA), dtype=bool)
    inside = (xpix >= 0) & (ypix >= 0) & (xpix < msk.shape[1]) & (ypix < msk.shape[0])
    print("inside percentage = ", np.sum(inside)*100./len(inside))
    xpix[~inside] = 0; ypix[~inside]= 0; # it is ok to be conservative near edge
    premask[msk[ypix, xpix] == 1.] = True
    premask[~inside] = False
    return premask

def get_premask_healpix(RA, DEC, msk):
    # transform coords
    c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs')
    B = c_icrs.galactic.b.value # -90, 90
    L = c_icrs.galactic.l.value # 0, 360
    vec = hp.ang2vec(theta=L, phi=B, lonlat=True)
    print("number before premasking = ", len(RA))
    
    # apply premasking 
    npix = len(msk)
    nside = hp.npix2nside(npix)
    ipix = hp.pixelfunc.vec2pix(nside, *vec.T)
    choice = msk[ipix] == 1.
    return choice

def compute_aperture(mp, msk, ra, dec, Theta_arcmin, r_max_arcmin, resCutoutArcmin, projCutout, matched_filter=None, power=None):
    
    # extract stamp
    opos, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=r_max_arcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)

    # skip if mask is zero everywhere
    if (np.sum(stampMask) == 0.) or (stampMask is None): T_AP = 0.; return T_AP
    
    if matched_filter is not None:
        # record T_MF
        dT = calc_T_MF(stampMap, fmap=matched_filter, mask=stampMask, power=power)
    else:
        # record T_AP
        dT = calc_T_AP(stampMap, Theta_arcmin, mask=stampMask)

    return dT

def jackknife_pairwise_momentum(P, delta_Ts, rbins, is_log_bin, dtype, nthread, n_jack=1000):

    # initialize
    PV_2D = np.zeros((len(rbins)-1, n_jack))
    DD_2D = np.zeros((len(rbins)-1, n_jack))
    inds = np.arange(len(delta_Ts))
    np.random.shuffle(inds)
    n_jump = len(inds)//n_jack
    for i in range(n_jack):
        # calculate the pairwise velocity
        t1 = time.time()
        new_inds = np.delete(inds, np.arange(n_jump*i, n_jump*(i+1)))
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        assert is_log_bin == False
        #assert rbins[0] == 0.
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread) # slow
        #PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, pos2=None, bias2=None)
        print("jackknife sample took = ", i, time.time()-t1)
        PV_2D[:, i] = PV
        DD_2D[:, i] = DD

    # compute errorbars
    DD_mean = np.mean(DD_2D, axis=1)
    PV_mean = np.mean(PV_2D, axis=1)    
    DD_err = np.sqrt(n_jack-1)*np.std(DD_2D, axis=1)
    PV_err = np.sqrt(n_jack-1)*np.std(PV_2D, axis=1)

    return PV_2D, PV_mean, PV_err

def bootstrap_pairwise_momentum(P, delta_Ts, rbins, is_log_bin, dtype, nthread, n_boot=1000):

    # initialize
    PV_2D = np.zeros((len(rbins)-1, n_boot))
    DD_2D = np.zeros((len(rbins)-1, n_boot))
    inds = np.arange(len(delta_Ts))
    for i in range(n_boot):
        # calculate the pairwise velocity
        t1 = time.time()
        new_inds = np.random.choice(inds, len(inds))
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        assert is_log_bin == False
        #assert rbins[0] == 0.
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread) # slow
        #PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, pos2=None, bias2=None)
        print("bootstrap sample took = ", i, time.time()-t1)
        PV_2D[:, i] = PV
        DD_2D[:, i] = DD

    # compute errorbars
    DD_mean = np.mean(DD_2D, axis=1)
    PV_mean = np.mean(PV_2D, axis=1)    
    DD_err = np.std(DD_2D, axis=1)
    PV_err = np.std(PV_2D, axis=1)

    return PV_2D, PV_mean, PV_err

def _init(shared_arr_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr
    shared_arr = shared_arr_

def shared_to_numpy(shared_arr, dtype, shape):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape)

def create_shared_array(dtype, shape):
    """Create a new shared array. Return the shared array pointer, and a NumPy array view to it.
    Note that the buffer values are not initialized.
    """
    dtype = np.dtype(dtype)
    # Get a ctype type from the NumPy dtype.
    cdtype = np.ctypeslib.as_ctypes_type(dtype)
    # Create the RawArray instance.
    try:
        shared_arr = multiprocessing.RawArray(cdtype, shape[0]*shape[1])
    except:
        shared_arr = multiprocessing.RawArray(cdtype, shape[0]) # sum(shape)) # B.H. bug?
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr

def _init3(shared_arr_, shared_map_, shared_mask_):
    # The shared array pointer is a global variable so that it can be accessed by the
    # child processes. It is a tuple (pointer, dtype, shape).
    global shared_arr, shared_map, shared_mask
    shared_arr = shared_arr_
    shared_map = shared_map_
    shared_mask = shared_mask_

def shared_to_numpy3(shared_arr, dtype, shape, shared_map, dtype_map, shape_map, shared_mask, dtype_mask, shape_mask):
    """Get a NumPy array from a shared memory buffer, with a given dtype and shape.
    No copy is involved, the array reflects the underlying shared buffer."""
    return np.frombuffer(shared_arr, dtype=dtype).reshape(shape), np.frombuffer(shared_map, dtype=dtype_map).reshape(shape_map), np.frombuffer(shared_mask, dtype=dtype_mask).reshape(shape_mask)

def parallel_aperture_shared(index_range, wcs, RA, DEC, theta_arcmin, rApMaxArcmin, resCutoutArcmin, projCutout, matched_filter, power):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    T_APs, new_mp, new_msk = shared_to_numpy3(*shared_arr, *shared_map, *shared_mask)
    # WARNING: need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    new_mp = enmap.ndmap(new_mp, wcs)
    new_msk = enmap.ndmap(new_msk, wcs) # this is necessary since this operation sheds the ndmap coating
    #print(T_APs[:10], new_mp.shape, new_msk.shape)
    #print(new_mp[:2], new_msk[:2])
    
    for i in range(i0, i1):
        if i % 1000 == 0: print(i)
        T_APs[i] = compute_aperture(new_mp, new_msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin[i], resCutoutArcmin, projCutout, matched_filter=matched_filter, power=power)

def parallel_jackknife_pairwise(index_range, inds, P, delta_Ts, rbins, is_log_bin, dtype, nthread, n_sample):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    PV_jack = shared_to_numpy(*shared_arr)
    # WARNING: need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    n_jump = len(inds)//n_sample
    for i in range(i0, i1):
        # calculate the pairwise velocity
        t1 = time.time()
        new_inds = np.delete(inds, np.arange(n_jump*i, n_jump*(i+1)))
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        assert is_log_bin == False
        #assert rbins[0] == 0.
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread) # slow
        #PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, pos2=None, bias2=None)
        print("jackknife sample took = ", i, time.time()-t1)
        PV_jack[:, i] = PV

def parallel_bootstrap_pairwise(index_range, rngs, inds, P, delta_Ts, Z, rbins, is_log_bin, dtype, nthread, sigma_z, tzav_each, tau, P_cross, bias_cross):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    PV_boot = shared_to_numpy(*shared_arr)
    # WARNING: need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    for i in range(i0, i1):
        # calculate the pairwise velocity
        t1 = time.time()
        rng = np.random.default_rng(rngs[i])
        new_inds = rng.choice(inds, len(inds))
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        Z_new = Z[new_inds]
        if tau is not None:
            tau_new = tau[new_inds]
        else:
            tau_new = None
        # recompute the z averaging for each sample
        if tzav_each:
            bar_T_new = get_tzav_fast(delta_Ts_new, Z_new, sigma_z) # this is called delta_Ts, but it is actually T_APs
            delta_Ts_new = delta_Ts_new - bar_T_new
        assert is_log_bin == False
        #assert rbins[0] == 0.
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        #PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, tau=tau_new, pos2=P_cross, bias2=bias_cross) # og
        #PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, tau=tau_new, pos2=P_new, bias2=bias_cross) # TESTING
        print("bootstrap sample took = ", i, time.time()-t1)
        #print(i, PV)
        PV_boot[:, i] = PV

def parallel_kmeansjack_pairwise(index_range, regions, inds, P, delta_Ts, Z, rbins, is_log_bin, dtype, nthread, sigma_z, tzav_each, tau, regions_cross, inds_cross, P_cross, bias_cross):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    PV_kmeans = shared_to_numpy(*shared_arr)
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    for i in range(i0, i1):
        # calculate the pairwise velocity
        t1 = time.time()
        new_inds = inds[regions != i]
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        Z_new = Z[new_inds]
        if tau is not None:
            tau_new = tau[new_inds]
        else:
            tau_new = None
        if regions_cross is not None:
            new_inds = inds_cross[regions_cross != i]
            P_cross_new = P_cross[new_inds]
            bias_cross_new = bias_cross[new_inds]
        
        # recompute the z averaging for each sample
        if tzav_each:
            bar_T_new = get_tzav_fast(delta_Ts_new, Z_new, sigma_z) # this is called delta_Ts, but it is actually T_APs
            delta_Ts_new = delta_Ts_new - bar_T_new
        assert is_log_bin == False
        PV = pairwise_vel(P_new, delta_Ts_new, rbins, nthread, periodic=False, box=None, tau=tau_new, pos2=P_cross_new, bias2=bias_cross_new) 
        print("kmeansjack sample took = ", i, time.time()-t1)
        PV_kmeans[:, i] = PV
        

def main(galaxy_sample, cmb_sample, resCutoutArcmin, projCutout, want_error, n_sample, data_dir, Theta, mask_type, asymm_pair=-1, bias_weight=False, tau_weight=False, vary_Theta=False, want_plot=False, want_MF=False, want_random=-1, want_premask=False, not_parallel=False, tzav_each=False, force=False):
    print(f"Producing: {galaxy_sample}_{cmb_sample}")
    sigma_z = 0.01
    vary_str = "vary" if vary_Theta else "fixed"
    MF_str = "MF" if want_MF else ""
    mask_str = "_premask" if want_premask else ""
    if mask_type == 'mtype0' or '': mask_type = ''; source_arcmin = 8.; noise_uK = 65
    else:
        if mask_type == 'mtype1': source_arcmin = 10.; noise_uK = 65
        elif mask_type == 'mtype2': source_arcmin = 35.; noise_uK = 65
        elif mask_type == 'mtype3': source_arcmin = 8.; noise_uK = 45
        elif mask_type == 'mtype4': source_arcmin = 10.; noise_uK = 45
        elif mask_type == 'mtype5': source_arcmin = 35.; noise_uK = 45
        mask_type = f'_{mask_type}'
    if want_random != -1:
        print("Requested using random galaxy positions, forcing 2MPZ-like sample")
        galaxy_sample = "2MPZ"
        rand_str = f"_rand{want_random:d}"
    else:
        rand_str = ""
    if bias_weight:
        assert ("SDSS" in galaxy_sample) or ("MGS" in galaxy_sample), "Missing an estimate of the bias"
        assert asymm_pair >= 0, "If you want to take into account bias, need asymmetric estimator"
        bias_str = "_bweight"
    else:
        bias_str = ""
    if asymm_pair >= 0:
        if asymm_pair == 0:
            #assert bias_weight or tau_weight, "Asymmetric estimator makes no difference"
            asymm_str = "_auto"
        else:
            asymm_str = "_cross"
    else:
        assert not bias_weight, "Weighting for bias is inherently asymmetric so can't do"
        asymm_str = ""
    if tau_weight:
        assert ("SDSS" in galaxy_sample) or ("MGS" in galaxy_sample), "Need an estimate of the bias"
        tau_str = "_tweight"
    else:
        tau_str = ""
        
    # file names and power spectrum for matched filter
    if want_MF:
        root_name = f"{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_{vary_str}Th{Theta:.2f}"
        if "DR5" in cmb_sample:
            binned_power = np.load(f"camb_data/{cmb_sample}_{noise_uK:d}uK_binned_power.npy")
            centers = np.load(f"camb_data/{cmb_sample}_{noise_uK:d}uK_centers.npy")
        else:
            binned_power = np.load(f"camb_data/{cmb_sample.split('_healpix')[0]}_binned_power.npy") # since Planck power spectrum 
            centers = np.load(f"camb_data/{cmb_sample.split('_healpix')[0]}_centers.npy")
    else:
        root_name = f"{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}"
    print(f"ROOT NAME = {root_name}")
    delta_T_fn = f"data/{root_name}_delta_Ts.npz"
    root_name += f"{rand_str}{asymm_str}{tau_str}{bias_str}"

    
    # create instance of the class "Class"
    Cosmo = Class()
    Cosmo.set(COSMO_DICT)
    Cosmo.compute()

    if "healpix" in cmb_sample:
        cmb_box = {'decfrom': -90., 'decto': 90., 'rafrom': 0., 'rato': 360.}
        if not os.path.exists(delta_T_fn):
            print("Missing files for computing pairwise estimator on healpix-generated delta Ts. This script can't generate those, need to run `get_delta_Ts_Planck.py` with the same settings.")
            quit()
    else:
        # load CMB map and mask
        mp, msk = load_cmb_sample(cmb_sample, data_dir, source_arcmin, noise_uK)
    
        # map info (used for cutting galaxies when using DR4)
        cmb_box = {}
        mp_box = np.rad2deg(mp.box())
        cmb_box['decfrom'], cmb_box['decto'], cmb_box['rafrom'], cmb_box['rato'] = mp_box[0, 0], mp_box[1, 0], mp_box[0, 1] % 360., mp_box[1, 1] % 360.
        print("cmb bounds = ", cmb_box.items())
        
    # load galaxies
    if asymm_pair >= 0:
        RA, DEC, Z, P, D_A, index, tau, RA_cross, DEC_cross, Z_cross, P_cross, D_A_cross, index_cross, bias_cross = load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random, return_asymm=asymm_pair, return_bias=bias_weight, return_tau=tau_weight)
    else:
        RA, DEC, Z, P, D_A, index, tau = load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random, return_tau=tau_weight)
        
    if "healpix" in cmb_sample and want_premask:        
        # load mask
        msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
        msk = hp.read_map(msk_fn, verbose=True)
        choice = get_premask_healpix(RA, DEC, msk)
        index = index[choice]
        RA = RA[choice]
        DEC = DEC[choice]
        Z = Z[choice]
        P = P[choice]
        D_A = D_A[choice]
        if tau_weight:
            tau = tau[choice]
        print("number after premasking = ", len(RA))
    
    # if files don't exist, need to compute
    if os.path.exists(delta_T_fn) and not force:
        data = np.load(delta_T_fn)
        delta_Ts = data['delta_Ts']
        index_new = data['index']
        choice = np.in1d(index, index_new)
        del index, index_new; gc.collect()
        Z = Z[choice]
        P = P[choice]
        if tau_weight:
            tau = tau[choice]
        if asymm_pair >= 0:
            if "healpix" in cmb_sample:
                premask = get_premask_healpix(RA_cross, DEC_cross, msk)
            else:
                premask = get_premask(RA_cross, DEC_cross, msk)
            index_cross = index_cross[premask]
            P_cross = P_cross[premask]
            bias_cross = bias_cross[premask]
            del premask; gc.collect()
        del data, choice; gc.collect()
    else:
        # size of the clusters and median redshift
        zmed = np.median(Z)
        if "SDSS" in galaxy_sample or "BOSS" in galaxy_sample:
            goal_size = 1.1 # Mpc (comoving)
        elif "2MPZ" in galaxy_sample:
            goal_size = 0.5 # Mpc (comoving) 0.5 or 0.6 or 0.792 see target.py in hydro_tests/
        elif "MGS" in galaxy_sample:
            goal_size = 1. # Mpc (comoving)
        elif "BGS" in galaxy_sample:
            goal_size = 1. # Mpc (comoving)
        goal_size *= 1./(1+zmed) # Mpc (proper)
        print("median redshift = ", zmed)
            
        # compute angular size distance at median redshift
        D_A_zmed = Cosmo.luminosity_distance(zmed)/(1.+zmed)**2 # Mpc (see note on units in tools)
        theta_arcmin_zmed = (goal_size/D_A_zmed) / utils.arcmin
        #theta_arcmin_zmed = (0.5*goal_size/D_A_zmed) / utils.arcmin 
        print("theta_arcmin_zmed = ", theta_arcmin_zmed)

        # scale aperture radius with redshift
        if Theta is None:
            Theta = theta_arcmin_zmed
        if vary_Theta:
            theta_arcmin = Theta*(D_A/D_A_zmed)
        else:
            theta_arcmin = Theta*np.ones(len(D_A))
        print("aperture radius min/max = ", theta_arcmin.min(), theta_arcmin.max())

        """
        # tau is actually halo mass # TESTING!!!!!!!!!!!!!!!!
        goal_size = np.zeros_like(Z)
        for i in range(len(Z)):
            # returns proper units
            Rsp, Msp, mask = splashback.splashbackRadius(Z[i], '200m', R=None, M=tau[i], c=None, model='diemer20', statistic='median', rspdef='sp-apr-mn')
            goal_size[i] = Rsp/1000. # Mpc/h
            #assert np.sum(mask) == len(mask)
            print(mask)
        goal_size /= h # Mpc
        print("proper median size in Mpc = ", np.median(goal_size))
        print("proper min/max size in Mpc = ", np.min(goal_size), np.max(goal_size))
        print("number larger than 2 Mpc = ", np.sum(goal_size > 2.))
        goal_size[goal_size > 2.] = 2.
        theta_arcmin = (goal_size/D_A) / utils.arcmin
        print("median theta in arcmin = ", np.median(theta_arcmin))
        print("aperture radius min/max = ", theta_arcmin.min(), theta_arcmin.max())
        """
        
        # size of the stamps divided by two
        rApMaxArcmin = theta_arcmin # in util.py multiplied by sqrt2 * 2 and ceil for size of canvas
        print("max radius of outer flux ceiled = ", rApMaxArcmin.min(), rApMaxArcmin.max())

        # convert to pixell coordinates
        RA[RA > 180.] -= 360.

        # apply cmb mask before computing temperature decrements
        if want_premask: 
            premask = get_premask(RA, DEC, msk)
            index = index[premask]
            RA = RA[premask]
            DEC = DEC[premask]
            P = P[premask]
            Z = Z[premask]
            if tau_weight:
                tau = tau[premask]
            #msk = mp*0. + 1. # from now on, no more masking # TESTING!!!!!
            print("number of galaxies (after premasking) = ", len(RA))
            del premask
    
        # matched filter or not
        if want_MF:
            assert vary_Theta == False, "stamps must be same sized" 
            assert np.isclose(np.mean(theta_arcmin), np.max(theta_arcmin)), "stamps must be same sized"
            assert np.isclose(np.mean(theta_arcmin), np.min(theta_arcmin)), "stamps must be same sized"

            # create empty map
            theta_arcmin[:] = Theta # 15. arcmin (total length is np.ceil(rApMaxArcmin*2*np.sqrt(2)))
            stamp = cutoutGeometry(projCutout=projCutout, rApMaxArcmin=np.mean(theta_arcmin), resCutoutArcmin=resCutoutArcmin) # const

            # get ell and distance maps
            modlmap = stamp.modlmap()
            modrmap = stamp.modrmap()

            # choose filter
            #fmap = gaussian_filter(modlmap)
            #fmap = tophat_filter(modrmap, np.mean(theta_arcmin)) # constant
            fmap = GNFW_filter(modrmap, np.mean(theta_arcmin)) # constant
            #fmap = kSZ_filter(modrmap)
            power = interpolate.interp1d(centers, binned_power, bounds_error=False, fill_value=None)(modlmap)
            power[modlmap == 0.] = 0.
        else:
            fmap = None
            power = None
            
        # parallel computation or not
        if not_parallel:
            # non-parallelized version                
            T_APs = np.zeros(len(RA))
            for i in range(len(RA)):
                if i % 1000 == 0: print("i = ", i)
                T_AP = compute_aperture(mp, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin[i], resCutoutArcmin, projCutout, matched_filter=fmap, power=power)
                T_APs[i] = T_AP
        else:
            # For simplicity, make sure the total size is a multiple of the number of processes.
            n_processes = os.cpu_count()
            n = len(RA) // n_processes
            index_ranges = []
            for k in range(n_processes-1):
                index_ranges.append((k * n, (k + 1) * n))
            index_ranges.append(((k + 1) * n, len(RA)))
            index_ranges = np.array(index_ranges)

            # Initialize a shared array.
            dtype = np.float64; shape = (len(RA),)
            shared_arr, T_APs = create_shared_array(dtype, shape)
            T_APs.flat[:] = np.zeros(len(RA))

            # Initialize a shared array.
            dtype_map = np.float64; shape_map = (mp.shape[0], mp.shape[1])
            shared_map, new_mp = create_shared_array(dtype_map, shape_map)
            new_mp.flat[:] = mp[:, :]
            
            # Initialize a shared array.
            dtype_mask = np.float64; shape_mask = (msk.shape[0], msk.shape[1])
            shared_mask, new_msk = create_shared_array(dtype_mask, shape_mask)
            new_msk.flat[:] = msk[:, :]
            
            # compute the aperture photometry for each galaxy
            # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
            with closing(multiprocessing.Pool(n_processes, initializer=_init3, initargs=((shared_arr, dtype, shape), (shared_map, dtype_map, shape_map), (shared_mask, dtype_mask, shape_mask),))) as p:   
                # Call parallel_function in parallel.
                p.starmap(parallel_aperture_shared, zip(index_ranges, repeat(mp.wcs), repeat(RA), repeat(DEC), repeat(theta_arcmin), repeat(rApMaxArcmin), repeat(resCutoutArcmin), repeat(projCutout), repeat(fmap), repeat(power)))
            # Close the processes.
            p.join()
            del new_mp, new_msk

        # apply cuts because of masking
        choice = T_APs != 0.
        T_APs = T_APs[choice]
        index = index[choice]
        Z = Z[choice]
        P = P[choice]
        if tau_weight:
            tau = tau[choice]
        print("percentage T_APs == 0. ", np.sum(~choice)*100./len(RA))
            
        # get the redshift-weighted apertures and temperature decrement around each galaxy
        if tzav_each:
            delta_Ts = T_APs
        else:
            bar_T_APs = get_tzav_fast(T_APs, Z, sigma_z)
            delta_Ts = T_APs - bar_T_APs
            del bar_T_APs
            
        # save apertures and indices
        np.savez(delta_T_fn, delta_Ts=delta_Ts, index=index)

        del mp, msk
        gc.collect()
    print("number of galaxies (after masking) = ", len(delta_Ts))
    
    # define bins in Mpc
    rbins = np.linspace(0., 150., 16) # Mpc # og
    print("rbins", rbins)
    #rbins = np.linspace(0., 100., 5)
    #rbins[0] = 1.e-6
    rbinc = (rbins[1:]+rbins[:-1])*.5 # Mpc
    nthread = 8 #os.cpu_count()//4
    is_log_bin = False
    
    # change dtype to speed up calculation
    dtype = np.float32
    P = P.astype(dtype)
    delta_Ts = delta_Ts.astype(dtype)
    Z = Z.astype(dtype)
    if tau_weight:
        tau = tau.astype(dtype)
    else:
        tau = None
        
    if asymm_pair >= 0:
        P_cross = P_cross.astype(dtype)
        bias_cross = bias_cross.astype(dtype)
    else:
        # to implement for jackknife, just add the two variables in the parallel functions
        P_cross = None
        bias_cross = None
    assert P.shape[0] == len(delta_Ts) == len(Z)
    
    if want_error == "jackknife":
        # For simplicity, make sure the total size is a multiple of the number of processes.
        n_processes = 25 #os.cpu_count() 
        n = n_sample // n_processes
        index_ranges = []
        for k in range(n_processes-1):
            index_ranges.append((k * n, (k + 1) * n))
        index_ranges.append(((k + 1) * n, n_sample))
        index_ranges = np.array(index_ranges)

        # Initialize a shared array.
        dtype = np.float64; shape = ((len(rbins)-1, n_sample))
        shared_arr, PV_jack = create_shared_array(dtype, shape)
        PV_jack.flat[:] = np.zeros(shape, dtype=dtype)

        # randomize indices for performing jackknifes
        inds = np.arange(len(delta_Ts))
        np.random.shuffle(inds)
        
        # compute the pairwise signal for each sample
        # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
        with closing(multiprocessing.Pool(n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:   
            # Call parallel_function in parallel.
            p.starmap(parallel_jackknife_pairwise, zip(index_ranges, repeat(inds), repeat(P), repeat(delta_Ts), repeat(rbins), repeat(is_log_bin), repeat(dtype), repeat(nthread), repeat(n_sample)))
        # Close the processes.
        p.join()

        # compute jackknifes
        #PV_jack, PV_mean, PV_err = jackknife_pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)

        # save arrays
        np.save(f"data/{root_name}_PV_jack.npy", PV_jack) # TODO: need to save raw file too
        np.save(f"data/rbinc.npy", rbinc)
    elif want_error == "bootstrap":
        assert is_log_bin == False
        #assert rbins[0] == 0.

        #DD, PV = pairwise_momentum(P, delta_Ts, rbins, is_log_bin, dtype=dtype, nthread=nthread)
        
        t = time.time()
        # if want to compute z averaging for each sample (TODO: jackknife case)
        if tzav_each:
            # need to compute this just for the first one
            bar_T_APs = get_tzav_fast(delta_Ts, Z, sigma_z) 
            delta_Ts_new = delta_Ts - bar_T_APs
        else:
            delta_Ts_new = delta_Ts
            
        #table =  numba_2pcf(P, box=1000., Rmax=np.max(rbins), nbin=len(rbins)-1, nthread=-1, n1djack=None, pg_kwargs=None, corrfunc=False)
        #PV = numba_pairwise_vel(P.copy(), delta_Ts_new.copy(), box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        PV = pairwise_vel(P, delta_Ts_new, rbins, nthread, periodic=False, box=None, tau=tau, pos2=P_cross, bias2=bias_cross)
        print("first calculation took = ", time.time()-t)
        print(PV)

        # save arrays
        np.save(f"data/{root_name}_PV.npy", PV)

        # For simplicity, make sure the total size is a multiple of the number of processes.
        n_processes = 10 #os.cpu_count() 
        n = n_sample // n_processes
        index_ranges = []
        for k in range(n_processes-1):
            index_ranges.append((k * n, (k + 1) * n))
        index_ranges.append(((k + 1) * n, n_sample))
        index_ranges = np.array(index_ranges)

        # Initialize a shared array.
        dtype = np.float64; shape = ((len(rbins)-1, n_sample))
        shared_arr, PV_boot = create_shared_array(dtype, shape)
        PV_boot.flat[:] = np.zeros(shape, dtype=dtype)

        # randomize indices for performing bootstraps
        inds = np.arange(len(delta_Ts))

        # create the RNG that you want to pass around
        seed = 98765
        rng = np.random.default_rng(seed)
        # get the SeedSequence of the passed RNG
        ss = rng.bit_generator._seed_seq
        # create n_sample initial independent states
        rngs = ss.spawn(n_sample)
        
        # compute the pairwise signal for each sample
        # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
        with closing(multiprocessing.Pool(n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:   
            # Call parallel_function in parallel.
            p.starmap(parallel_bootstrap_pairwise, zip(index_ranges, repeat(rngs), repeat(inds), repeat(P), repeat(delta_Ts), repeat(Z), repeat(rbins), repeat(is_log_bin), repeat(dtype), repeat(nthread), repeat(sigma_z), repeat(tzav_each), repeat(tau), repeat(P_cross), repeat(bias_cross)))
        # Close the processes.
        p.join()

        # save arrays
        np.save(f"data/{root_name}_PV_boot.npy", PV_boot)
        np.save(f"data/rbinc.npy", rbinc)

    elif want_error == "kmeansjack":
        assert is_log_bin == False
        
        t = time.time()
        # if want to compute z averaging for each sample (TODO: jackknife case)
        if tzav_each:
            # need to compute this just for the first one
            bar_T_APs = get_tzav_fast(delta_Ts, Z, sigma_z) 
            delta_Ts_new = delta_Ts - bar_T_APs
        else:
            delta_Ts_new = delta_Ts
        PV = pairwise_vel(P, delta_Ts_new, rbins, nthread, periodic=False, box=None, tau=tau, pos2=P_cross, bias2=bias_cross)
        print("first calculation took = ", time.time()-t)
        print(PV)

        # save arrays
        np.save(f"data/{root_name}_PV.npy", PV)

        # For simplicity, make sure the total size is a multiple of the number of processes.
        n_processes = 10 #os.cpu_count() 
        n = n_sample // n_processes
        index_ranges = []
        for k in range(n_processes-1):
            index_ranges.append((k * n, (k + 1) * n))
        index_ranges.append(((k + 1) * n, n_sample))
        index_ranges = np.array(index_ranges)

        # Initialize a shared array.
        dtype = np.float64; shape = ((len(rbins)-1, n_sample))
        shared_arr, PV_kmeans = create_shared_array(dtype, shape)
        PV_kmeans.flat[:] = np.zeros(shape, dtype=dtype)

        # randomize indices for performing kmeans jackknifes
        inds = np.arange(len(delta_Ts))
        if asymm_pair >= 0:
            inds_cross = np.arange(P_cross.shape[0])
        else:
            inds_cross = None

        # create mask
        nside = 128
        mask = np.zeros(12*nside**2)
        ipix = hp.vec2pix(nside, P[:, 0], P[:, 1], P[:, 2])
        mask[ipix] = 1.
        if asymm_pair >= 0:
            ipix_cross = hp.vec2pix(nside, P_cross[:, 0], P_cross[:, 1], P_cross[:, 2])
            mask[ipix_cross] = 1.
        
        # regions
        regions_map = get_regions(mask, n_sample, nside)
        regions = regions_map[ipix]
        if asymm_pair >= 0:
            regions_cross = regions_map[ipix_cross]
            del ipix_cross
        else:
            regions_cross = None
        del regions_map, ipix, mask; gc.collect()
        
        # compute the pairwise signal for each sample
        # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
        with closing(multiprocessing.Pool(n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:   
            # Call parallel_function in parallel.
            p.starmap(parallel_kmeansjack_pairwise, zip(index_ranges, repeat(regions), repeat(inds), repeat(P), repeat(delta_Ts), repeat(Z), repeat(rbins), repeat(is_log_bin), repeat(dtype), repeat(nthread), repeat(sigma_z), repeat(tzav_each), repeat(tau), repeat(regions_cross), repeat(inds_cross), repeat(P_cross), repeat(bias_cross)))
        # Close the processes.
        p.join()

        # save arrays
        np.save(f"data/{root_name}_PV_kmeans.npy", PV_kmeans)
        np.save(f"data/rbinc.npy", rbinc)
    else:
        # calculate the pairwise velocity
        #DD_old, PV_old = pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        
        t = time.time()
        assert is_log_bin == False
        #assert rbins[0] == 0. # not needed when you use corrfunc
        #table = numba_pairwise_vel(P, delta_Ts, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)
        #DD = table['npairs']
        #PV = table['pairwise']
        PV = pairwise_vel(P, delta_Ts, rbins, nthread, periodic=False, box=None, pos2=None, bias2=None)
        print("calculation took = ", time.time()-t)
        # save arrays
        np.save(f"data/{root_name}_PV.npy", PV)
        np.save(f"data/rbinc.npy", rbinc)

    # plot pairwise velocity
    if want_plot:
        plt.figure(figsize=(9, 7))
        plt.plot(rbinc, np.zeros(len(PV)), 'k--')
        plt.plot(rbinc, PV)
        plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu{\rm K}]$")
        plt.xlabel(r"$r \ [{\rm Mpc}]$")
        plt.show()

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    # not too sure what this does
    multiprocessing.freeze_support()
    
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--galaxy_sample', '-gal', help='Which galaxy sample do you want to use?',
                        default=DEFAULTS['galaxy_sample'])#, choices=["BOSS_South", "BOSS_North", "2MPZ", "SDSS_L43D", "SDSS_L61D", "2MPZ_Biteau", "SDSS_L43", "SDSS_L61", "SDSS_L79", "SDSS_all", "MGS", "MGS_grp", "eBOSS_SGC", "eBOSS_NGC"])
    parser.add_argument('--cmb_sample', '-cmb', help='Which CMB sample do you want to use?',
                        default=DEFAULTS['cmb_sample'],
                        choices=["ACT_BN", "ACT_D56", "ACT_DR5_f090", "ACT_DR5_f150", "ACT_DR6", "Planck", "Planck_healpix"])
    parser.add_argument('--resCutoutArcmin', help='Resolution of the cutout', type=float, default=DEFAULTS['resCutoutArcmin'])
    parser.add_argument('--projCutout', help='Projection of the cutout', default=DEFAULTS['projCutout'])
    parser.add_argument('--want_error', '-error', help='Perform jackknife/bootstrap/none error computation of pairwise momentum', default=DEFAULTS['error_estimate'], choices=["none", "jackknife", "bootstrap", "kmeansjack"])
    parser.add_argument('--n_sample', help='Number of samples when performing jackknife/bootstrap', type=int, default=DEFAULTS['n_sample'])
    parser.add_argument('--Theta', '-Th', help='Aperture radius in arcmin', type=float, default=None)
    parser.add_argument('--vary_Theta', '-vary', help='Vary the aperture radius', action='store_true')
    parser.add_argument('--data_dir', help='Directory where the data is stored', default=DEFAULTS['data_dir'])
    parser.add_argument('--want_plot', '-plot', help='Plot the final pairwise momentum function', action='store_true')
    parser.add_argument('--want_MF', '-MF', help='Want to use matched filter', action='store_true')
    parser.add_argument('--want_random', '-rand', help='Random seed to shuffle galaxy positions (-1 does not randomize)', type=int, default=-1)
    parser.add_argument('--tau_weight', '-tau', help='Luminosity weighting of the temperature decrements', action='store_true')
    parser.add_argument('--bias_weight', '-bias', help='Bias weighting of the temperature decrements', action='store_true')
    parser.add_argument('--asymm_pair', '-asymm', help='Want asymmetric estimator (-1 normal pairwise; 0 asymmetric auto; 1 asymmetric using all galaxies)', type=int, default=-1)
    parser.add_argument('--want_premask', '-mask', help='Mask galaxies with CMB mask before taking temperature decrements', action='store_true')
    parser.add_argument('--mask_type', '-mtype', help='Type of CMB mask to apply', choices=['', 'mtype0', 'mtype1', 'mtype2', 'mtype3', 'mtype4', 'mtype5'], default=DEFAULTS['mask_type'])
    parser.add_argument('--not_parallel', help='Do serial computation of aperture rather than parallel', action='store_true')
    parser.add_argument('--tzav_each', help='Apply the delta T averaging procedure to each jackknife/bootstrap sample (currently AP/MF output is not marked in any way)', action='store_true')
    parser.add_argument('--force', help='Run even if AP/MF files exist', action='store_true')
    args = vars(parser.parse_args())

    main(**args)

