#!/usr/bin/env python3
"""
Usage
-----
$ ./multi_pairwise_momentum.py --help
"""
import os
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt

from pixell import enmap, enplot, utils
from classy import Class

from util import extractStamp, calc_T_AP, eshow, get_tzav_fast
from estimator import pairwise_momentum
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

def compute_aperture(mp, msk, ra, dec, Theta_arcmin, r_max_arcmin, resCutoutArcmin, projCutout):

    # extract stamp
    opos, stampMap, stampMask = extractStamp(mp, ra, dec, rApMaxArcmin=r_max_arcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout, pathTestFig='figs/', test=False, cmbMask=msk)

    # skip if mask is zero everywhere
    if (np.sum(stampMask) == 0.) or (stampMask is None): T_AP = 0.
        
    # record T_AP
    T_AP = calc_T_AP(stampMap, Theta_arcmin, mask=stampMask)

    return T_AP

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
        DD, PV = pairwise_momentum(P[new_inds], delta_Ts[new_inds], rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
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
    n_jump = len(inds)//n_boot
    for i in range(n_boot):
        # calculate the pairwise velocity
        t1 = time.time()
        new_inds = np.random.choice(inds, len(inds))
        DD, PV = pairwise_momentum(P[new_inds], delta_Ts[new_inds], rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
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
    shared_arr = multiprocessing.RawArray(cdtype, sum(shape))
    # Get a NumPy array view.
    arr = shared_to_numpy(shared_arr, dtype, shape)
    return shared_arr, arr


def parallel_function(index_range, wcs, mp, msk, RA, DEC, theta_arcmin, rApMaxArcmin, resCutoutArcmin, projCutout):
    """This function is called in parallel in the different processes. It takes an index range in
    the shared array, and fills in some values there."""
    i0, i1 = index_range
    T_APs = shared_to_numpy(*shared_arr)
    # WARNING: need to make sure that two processes do not write to the same part of the array.
    # Here, this is the case because the ranges are not overlapping, and all processes receive
    # a different range.
    mp = enmap.ndmap(mp, wcs)
    msk = enmap.ndmap(msk, wcs)
    #dT = np.zeros(i1-i0) # i get why this was failing cause its indexing is from 0 to i1-i0
    for i in range(i0, i1):
        if i % 1000 == 0: print(i)
        #dT[i] = compute_aperture(mp, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin[i], resCutoutArcmin, projCutout)
        T_APs[i] = compute_aperture(mp, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin[i], resCutoutArcmin, projCutout)
    #T_APs[i0:i1] = dT


def main(galaxy_sample, cmb_sample, resCutoutArcmin, projCutout, want_error, Theta, vary_Theta=False, want_plot=False):
    print(f"Producing: {galaxy_sample:s}_{cmb_sample:s}")
    vary_str = "vary" if vary_Theta else "fixed"
    
    # load CMB map and mask
    mp, msk = load_cmb_sample(cmb_sample)

    # map info
    cmb_box = {}
    mp_box = np.rad2deg(mp.box())
    cmb_box['decfrom'] = mp_box[0, 0]
    cmb_box['decto'] = mp_box[1, 0]
    cmb_box['rafrom'] = mp_box[0, 1] % 360.
    cmb_box['rato'] = mp_box[1, 1] % 360.
    print("cmb bounds = ", cmb_box.items())

    # load galaxies
    RA, DEC, Z, index = load_galaxy_sample(galaxy_sample, cmb_sample, cmb_box)

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
        goal_size = 0.6 # Mpc (comoving) 0.5 or 0.6 see target.py in hydro_tests/
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
    rApMaxArcmin = theta_arcmin # in util.py multiplied by sqrt2
    print("max radius of outer flux ceiled = ", rApMaxArcmin.min(), rApMaxArcmin.max())

    # convert to pixell coordinates
    RA[RA > 180.] -= 360.

    delta_T_fn = f"data/{galaxy_sample:s}_{cmb_sample:s}_{vary_str:s}Th{Theta:.2f}_delta_Ts.npy"
    index_fn = f"../pairwise/data/{galaxy_sample}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_index.npy"
    if os.path.exists(delta_T_fn) and os.path.exists(index_fn):
        delta_Ts = np.load(delta_T_fn)
        index_new = np.load(index_fn)
        choice = np.in1d(index, index_new)
        Z = Z[choice]
        P = P[choice]
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

        # compute the aperture photometry for each galaxy
        # Create a Pool of processes and expose the shared array to the processes, in a global variable (_init() function)
        with closing(multiprocessing.Pool(n_processes, initializer=_init, initargs=((shared_arr, dtype, shape),))) as p:   
            # Call parallel_function in parallel.
            p.starmap(parallel_function, zip(index_ranges, repeat(mp.wcs), repeat(mp), repeat(msk), repeat(RA), repeat(DEC), repeat(theta_arcmin), repeat(rApMaxArcmin), repeat(resCutoutArcmin), repeat(projCutout)))
        # Close the processes.
        p.join()

        # non-parallelized version
        #T_APs = np.zeros(len(RA))
        #for i in range(len(RA)):
        #    if i % 1000 == 0: print("i = ", i)
        #    T_AP = compute_aperture(mp, msk, RA[i], DEC[i], theta_arcmin[i], rApMaxArcmin[i], resCutoutArcmin, projCutout)
        #    T_APs[i] = T_AP

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

    # define bins in Mpc
    rbins = np.linspace(0., 150., 16) # Mpc
    rbinc = (rbins[1:]+rbins[:-1])*.5 # Mpc
    nthread = os.cpu_count()
    is_log_bin = False

    # change dtype to speed up calculation
    dtype = np.float32
    P = P.astype(dtype)
    delta_Ts = delta_Ts.astype(dtype)
    assert P.shape[0] == len(delta_Ts)

    if want_error == "jackknife":
        # compute jackknifes
        PV_jack, PV_mean, PV_err = jackknife_pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)

        # save arrays
        np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_jack.npy", PV_jack)
        np.save(f"data/rbinc.npy", rbinc)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_jack_mean.npy", PV_mean)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_jack_err.npy", PV_err)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_DD_jack_mean.npy", DD_mean)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_DD_jack_err.npy", DD_err)
    elif want_error == "bootstrap":
        # compute bootstraps
        PV_boot, PV_mean, PV_err = bootstrap_pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)

        # save arrays
        np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_boot.npy", PV_boot)
        np.save(f"data/rbinc.npy", rbinc)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_boot_mean.npy", PV_mean)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV_boot_err.npy", PV_err)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_DD_boot_mean.npy", DD_mean)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_DD_boot_err.npy", DD_err)
    else:
        # calculate the pairwise velocity
        t = time.time()
        DD, PV = pairwise_momentum(P, delta_Ts, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
        print("calculation took = ", time.time()-t)

        # save arrays
        np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_PV.npy", PV)
        np.save(f"data/rbinc.npy", rbinc)
        #np.save(f"data/{galaxy_sample:s}_{cmb_sample}_{vary_str:s}Th{Theta:.2f}_DD.npy", DD)

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
                        choices=["ACT_BN", "ACT_D56"])
    parser.add_argument('--resCutoutArcmin', help='Resolution of the cutout', type=float, default=DEFAULTS['resCutoutArcmin'])
    parser.add_argument('--projCutout', help='Projection of the cutout', default=DEFAULTS['projCutout'])
    parser.add_argument('--want_error', '-error', help='Perform jackknife/bootstrap/none error computation of pairwise momentum', default=DEFAULTS['error_estimate'], choices=["none", "jackknife", "bootstrap"])
    parser.add_argument('--Theta', '-Th', help='Aperture radius in arcmin', type=float, default=None)
    parser.add_argument('--vary_Theta', '-vary', help='Vary the aperture radius', action='store_true')
    parser.add_argument('--want_plot', '-plot', help='Plot the final pairwise momentum function', action='store_true')
    args = vars(parser.parse_args())

    main(**args)
