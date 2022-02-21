#!/usr/bin/env python3
"""
Usage
-----
$ ./multi_pairwise_momentum.py --help
"""
import os
import gc
import time
import argparse

import numpy as np
import numba
numba.config.THREADING_LAYER = 'safe'
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class

from util import extractStamp, calc_T_AP, eshow, get_tzav_fast, cutoutGeometry, gaussian_filter, calc_T_MF
from estimator import pairwise_momentum
from numba_2pcf.cf import numba_pairwise_vel
from tools_pairwise_momentum import get_P_D_A, load_cmb_sample, load_galaxy_sample, get_sdss_lum_lims

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
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        assert is_log_bin == False
        assert rbins[0] == 0.
        PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
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
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        assert is_log_bin == False
        assert rbins[0] == 0.
        PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
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
        #DD, PV = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        assert is_log_bin == False
        assert rbins[0] == 0.
        PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        print("jackknife sample took = ", i, time.time()-t1)
        PV_jack[:, i] = PV

def parallel_bootstrap_pairwise(index_range, inds, P, delta_Ts, rbins, is_log_bin, dtype, nthread):
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
        new_inds = np.random.choice(inds, len(inds))
        P_new = P[new_inds]
        delta_Ts_new = delta_Ts[new_inds]
        #DD_old, PV_old = pairwise_momentum(P_new, delta_Ts_new, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        assert is_log_bin == False
        assert rbins[0] == 0.
        PV = numba_pairwise_vel(P_new, delta_Ts_new, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)['pairwise']
        print("bootstrap sample took = ", i, time.time()-t1)
        PV_boot[:, i] = PV

def main(galaxy_sample, cmb_sample, resCutoutArcmin, projCutout, want_error, n_sample, data_dir, Theta, mask_type, vary_Theta=False, want_plot=False, want_MF=False, want_random=-1, want_premask=False, not_parallel=False):
    print(f"Producing: {galaxy_sample}_{cmb_sample}")
    vary_str = "vary" if vary_Theta else "fixed"
    MF_str = "MF" if want_MF else ""
    mask_str = "_premask" if want_premask else ""
    if mask_type == 'mtype0' or '': mask_type = ''; source_arcmin = 8.; noise_uK = 65.
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
    
    # load CMB map and mask
    mp, msk = load_cmb_sample(cmb_sample, data_dir, source_arcmin, noise_uK)
    
    # map info (used for cutting galaxies when using DR4)
    cmb_box = {}
    mp_box = np.rad2deg(mp.box())
    cmb_box['decfrom'] = mp_box[0, 0]
    cmb_box['decto'] = mp_box[1, 0]
    cmb_box['rafrom'] = mp_box[0, 1] % 360.
    cmb_box['rato'] = mp_box[1, 1] % 360.
    print("cmb bounds = ", cmb_box.items())

    # load galaxies
    RA, DEC, Z, index = load_galaxy_sample(galaxy_sample, cmb_sample, data_dir, cmb_box, want_random)

    # create instance of the class "Class"
    Cosmo = Class()
    Cosmo.set(COSMO_DICT)
    Cosmo.compute()
    
    # get cartesian coordinates and angular size distance
    P, D_A = get_P_D_A(Cosmo, RA, DEC, Z)

    # size of the clusters and median redshift
    zmed = np.median(Z)
    if "SDSS" in galaxy_sample:
        goal_size = 1.1 # Mpc (comoving)
    elif "2MPZ" in galaxy_sample:
        goal_size = 0.6 # Mpc (comoving) 0.5 or 0.6 or 0.792 see target.py in hydro_tests/
    goal_size *= 1./(1+zmed) # Mpc (proper)
    sigma_z = 0.01
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
    
    # size of the stamps divided by two
    #rApMaxArcmin = np.ceil(theta_arcmin*np.sqrt(2.)) # since sqrt2 is the max radius
    rApMaxArcmin = theta_arcmin # in util.py multiplied by sqrt2 * 2 and ceil for size of canvas
    print("max radius of outer flux ceiled = ", rApMaxArcmin.min(), rApMaxArcmin.max())

    # convert to pixell coordinates
    RA[RA > 180.] -= 360.

    # apply cmb mask before computing temperature decrements
    if want_premask:
        coords = np.deg2rad(np.array((DEC, RA)))
        ypix, xpix = enmap.sky2pix(msk.shape, msk.wcs, coords)
        print(xpix.min(), xpix.max(), msk.shape[1])
        print(ypix.min(), ypix.max(), msk.shape[0])
        xpix, ypix = xpix.astype(int), ypix.astype(int)
        premask = np.zeros(len(RA), dtype=bool)
        inside = (xpix >= 0) & (ypix >= 0) & (xpix < msk.shape[1]) & (ypix < msk.shape[0])
        print("inside percentage = ", np.sum(inside)*100./len(inside))
        xpix[~inside] = 0; ypix[~inside]= 0; # it is ok to be conservative near edge
        premask[msk[ypix, xpix] == 1.] = True
        premask[~inside] = False
        index = index[premask]
        RA = RA[premask]
        DEC = DEC[premask]
        P = P[premask]
        Z = Z[premask]
        msk = msk*0. + 1. # from now on, no more masking
        print("number of galaxies (after premasking) = ", len(RA))
        del premask, xpix, ypix, inside
    
    if want_MF:
        delta_T_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_delta_Ts.npy"
        index_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_index.npy"
        binned_power = np.load(f"camb_data/{cmb_sample}_{noise_uK:d}uK_binned_power.npy")
        centers = np.load(f"camb_data/{cmb_sample}_{noise_uK:d}uK_centers.npy")
        power = np.vstack((centers, binned_power)).T
    else:
        delta_T_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_delta_Ts.npy"
        index_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_index.npy"
        power = None
    if os.path.exists(delta_T_fn) and os.path.exists(index_fn):
        delta_Ts = np.load(delta_T_fn)
        index_new = np.load(index_fn)
        choice = np.in1d(index, index_new)
        Z = Z[choice]
        P = P[choice]
    else:
        if want_MF:
            size_stamp = 15. # arcmin (total length is np.ceil(size_stamp*2*np.sqrt(2)))
            theta_arcmin[:] = size_stamp # all cutouts should be that big
            stamp = cutoutGeometry(projCutout=projCutout, rApMaxArcmin=size_stamp, resCutoutArcmin=resCutoutArcmin)
            modlmap = stamp.modlmap() 
            fmap = gaussian_filter(modlmap)
        else:
            fmap = None

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
        print("percentage T_APs == 0. ", np.sum(~choice)*100./len(RA))

        # get the redshift-weighted apertures and temperature decrement around each galaxy
        bar_T_APs = get_tzav_fast(T_APs, Z, sigma_z)
        delta_Ts = T_APs - bar_T_APs

        # save apertures and indices
        np.save(delta_T_fn, delta_Ts)
        np.save(index_fn, index)
    del mp, msk
    gc.collect()
    print("number of galaxies (after masking) = ", len(delta_Ts))
    
    # define bins in Mpc
    rbins = np.linspace(0., 150., 16) # Mpc
    rbinc = (rbins[1:]+rbins[:-1])*.5 # Mpc
    nthread = os.cpu_count()//4
    is_log_bin = False

    # change dtype to speed up calculation
    dtype = np.float32
    P = P.astype(dtype)
    delta_Ts = delta_Ts.astype(dtype)
    assert P.shape[0] == len(delta_Ts)

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
        if want_MF:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_PV_jack.npy", PV_jack)
        else:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_PV_jack.npy", PV_jack)
        np.save(f"data/rbinc.npy", rbinc)
    elif want_error == "bootstrap":
        t = time.time()
        assert is_log_bin == False
        assert rbins[0] == 0.
        table = numba_pairwise_vel(P, delta_Ts, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)
        print("first calculation took = ", time.time()-t)
        DD = table['npairs']
        PV = table['pairwise']
        # save arrays
        if want_MF:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_PV.npy", PV)
        else:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_PV.npy", PV)

        
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
        
        # compute the pairwise signal for each sample
        # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
        with closing(multiprocessing.Pool(n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:   
            # Call parallel_function in parallel.
            p.starmap(parallel_bootstrap_pairwise, zip(index_ranges, repeat(inds), repeat(P), repeat(delta_Ts), repeat(rbins), repeat(is_log_bin), repeat(dtype), repeat(nthread)))
        # Close the processes.
        p.join()
        
        # compute bootstraps
        #PV_boot, PV_mean, PV_err = bootstrap_pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)

        # save arrays
        if want_MF:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_PV_boot.npy", PV_boot)
        else:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_PV_boot.npy", PV_boot)
        np.save(f"data/rbinc.npy", rbinc)
    else:
        # calculate the pairwise velocity
        #DD_old, PV_old = pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        
        t = time.time()
        assert is_log_bin == False
        assert rbins[0] == 0.
        table = numba_pairwise_vel(P, delta_Ts, box=None, Rmax=np.max(rbins), nbin=len(rbins)-1, corrfunc=False, nthread=nthread, periodic=False)
        DD = table['npairs']
        PV = table['pairwise']
        print("calculation took = ", time.time()-t)
        # save arrays
        if want_MF:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_PV.npy", PV)
        else:
            np.save(f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_PV.npy", PV)
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
                        default=DEFAULTS['galaxy_sample'],
                        choices=["BOSS_South", "BOSS_North", "2MPZ", "SDSS_L43D", "SDSS_L61D",
                                 "SDSS_L43", "SDSS_L61", "SDSS_L79", "SDSS_all"])
    parser.add_argument('--cmb_sample', '-cmb', help='Which CMB sample do you want to use?',
                        default=DEFAULTS['cmb_sample'],
                        choices=["ACT_BN", "ACT_D56", "ACT_DR5_f090", "ACT_DR5_f150"])
    parser.add_argument('--resCutoutArcmin', help='Resolution of the cutout', type=float, default=DEFAULTS['resCutoutArcmin'])
    parser.add_argument('--projCutout', help='Projection of the cutout', default=DEFAULTS['projCutout'])
    parser.add_argument('--want_error', '-error', help='Perform jackknife/bootstrap/none error computation of pairwise momentum', default=DEFAULTS['error_estimate'], choices=["none", "jackknife", "bootstrap"])
    parser.add_argument('--n_sample', help='Number of samples when performing jackknife/bootstrap', type=int, default=DEFAULTS['n_sample'])
    parser.add_argument('--Theta', '-Th', help='Aperture radius in arcmin', type=float, default=None)
    parser.add_argument('--vary_Theta', '-vary', help='Vary the aperture radius', action='store_true')
    parser.add_argument('--data_dir', help='Directory where the data is stored', default=DEFAULTS['data_dir'])
    parser.add_argument('--want_plot', '-plot', help='Plot the final pairwise momentum function', action='store_true')
    parser.add_argument('--want_MF', '-MF', help='Want to use matched filter', action='store_true')
    parser.add_argument('--want_random', '-rand', help='Random seed to shuffle galaxy positions (-1 does not randomize)', type=int, default=-1)
    parser.add_argument('--want_premask', '-mask', help='Mask galaxies with CMB mask before taking temperature decrements', action='store_true')
    parser.add_argument('--mask_type', '-mtype', help='Type of CMB mask to apply', choices=['', 'mtype0', 'mtype1', 'mtype2', 'mtype3', 'mtype4', 'mtype5'], default=DEFAULTS['mask_type'])
    parser.add_argument('--not_parallel', help='Do serial computation of aperture rather than parallel', action='store_true')
    args = vars(parser.parse_args())

    main(**args)
