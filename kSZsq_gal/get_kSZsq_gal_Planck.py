#!/usr/bin/env python3
"""
Usage
-----
$ ./get_kSZsq_gal_Planck.py --help
"""
import os
import sys
sys.path.append('../pairwise')
import time
import argparse

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from healpy.projector import GnomonicProj

from astropy.coordinates import SkyCoord
from astropy import units as u
from pixell import enmap, utils, powspec
from classy import Class

from tools_pairwise_momentum import load_galaxy_sample
from utils_power import bin_mat

# defaults
DEFAULTS = {}
DEFAULTS['galaxy_sample'] = "2MPZ"

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

def main(galaxy_sample, apod_mw, apod_star, apod_gal, want_joint=False):

    # constants
    coord = 'G' # milky way at 0 0  (GE not working if not doing Gnomonic)
    cmb_sample = "Planck_healpix"
    nest = False # Ordering converted to RING (default)
    cmb_box = {'decfrom': -90., 'decto': 90., 'rafrom': 0., 'rato': 360.}
    data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
    plot_moll = False

    # apodization scale and mask choice
    apod_mw *= utils.arcmin # (10 arcmin in paper)
    apod_star *= utils.arcmin # (2 arcmin for point source masks)
    apod_gal = apod_gal*utils.arcmin if want_joint else 0. # if treated separately, then galaxies don't need a mask
    print("smoothing scales in arcmins = ", apod_mw/utils.arcmin, apod_star/utils.arcmin, apod_mw/utils.arcmin)
    
    # load CMB map
    #cmb_mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    #cmb_mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits"
    #cmb_mp_fn = data_dir+"/cmb_data/dust/COM_CMB_IQU-smica-nosz-nodust_2048_R3.00_full.fits"
    #cmb_mp_fn = data_dir+"/cmb_data/LGMCA/WPR2_CMB_muK.fits"
    #cmb_mp_fn = data_dir+"/cmb_data/dust/WPR2_CMB_nodust_K.fits" # new!
    cmb_mp_fn = data_dir+f"/cmb_data/dust/WPR2_CMB_elldep_nodust_K_{galaxy_sample}.fits" # newest!!!
    cmb_mp = hp.read_map(cmb_mp_fn, verbose=True)
    
    # load CMB masks (pt src makes big difference)
    #mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits" # 70% MW + point sources 
    #mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal60.fits" # 60% MW + point sources
    #mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal60.fits" # 60% MW
    mw_msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits" # 70% MW
    mw_msk = hp.read_map(mw_msk_fn, verbose=True)
    star_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all.fits" # point sources
    star_msk = hp.read_map(star_msk_fn, verbose=True)

    # scrape info from masks
    npix = len(mw_msk)
    nside = hp.npix2nside(npix)
    if 'muK' not in cmb_mp_fn:
        cmb_mp *=  1.e6 # uK
    assert len(mw_msk) == len(cmb_mp)

    # power parameters
    LMIN = 0
    LMAX = 3*nside+1
    ell_data = np.arange(LMIN, LMAX, 1)
    
    if not os.path.exists(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits"):
        if galaxy_sample == "WISE":
            # load counts from Simo's file
            gal_counts = hp.read_map(data_dir+"wise_data/WISEgals_2048.fits")
            gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
            gal_msk = hp.ud_grade(gal_msk, nside)
        else:
            # create instance of the class "Class"
            Cosmo = Class()
            Cosmo.set(COSMO_DICT)
            Cosmo.compute() 

            # load galaxies
            RA, DEC, Z, index, gal_msk = load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random=-1, return_mask=True)
            if gal_msk is not None:
                gal_msk = hp.ud_grade(gal_msk, nside)
            c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
            B = c_icrs.galactic.b.value # -90, 90
            L = c_icrs.galactic.l.value # 0, 360
            vec = hp.ang2vec(theta=L, phi=B, lonlat=True)
            print("number before premasking = ", len(RA))

            # get pixel of each galaxy
            ipix = hp.pixelfunc.vec2pix(nside, *vec.T)
            #x = np.cos(B*utils.degree)*np.cos(L*utils.degree)
            #y = np.cos(B*utils.degree)*np.sin(L*utils.degree)
            #z = np.sin(B*utils.degree) # equiv to vec
            #ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
            
            # apply premasking
            """
            choice = gal_msk[ipix] == 1.
            index = index[choice]
            RA = RA[choice]
            DEC = DEC[choice]
            Z = Z[choice]
            B = B[choice]
            L = L[choice]
            print("number after premasking = ", len(RA))
            """
            
            # compute galaxy overdensity
            gal_counts = np.zeros(npix, dtype=int)
            ipix_un, counts = np.unique(ipix, return_counts=True)
            gal_counts[ipix_un] = counts
            
        # compute overdensity
        gal_mean = np.sum(gal_counts*gal_msk)/np.sum(gal_msk)
        gal_delta = gal_counts/gal_mean - 1.
        print("total number of galaxies = ", np.sum(gal_counts*gal_msk))
        print("mean density = ", np.sum(gal_delta*gal_msk)/np.sum(gal_msk))
        if plot_moll:
            hp.mollview(np.log10(gal_delta+1.), title="Delta gal", cmap='seismic')
            hp.mollview(gal_counts, title="Counts gal", cmap='seismic')
            #plt.show()

        # compute shotnoise
        gal_nbar = gal_mean/hp.nside2pixarea(nside)
        shotnoise = 1/gal_nbar # galaxy shot noise
        print("number of galaxies per deg^2 = ", gal_nbar/(180.**2/np.pi**2))
        print("shotnoise = ", shotnoise)

        # save map
        hp.write_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits", gal_delta)    
    else:
        # load map
        gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits")

        # load mask
        if galaxy_sample == "WISE":
            gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
        else:
            assert galaxy_sample == "DECALS"
            gal_msk = hp.read_map(data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits")
        gal_msk = hp.ud_grade(gal_msk, nside)
            
    if plot_moll:
        hp.mollview(gal_msk, title="Galaxy mask")
        #plt.show()

    # apply smoothing
    if apod_mw > 0.:
        mw_msk = hp.smoothing(mw_msk, fwhm=apod_mw, iter=5, verbose=False)
        mw_msk[mw_msk < 0.] = 0.; mw_msk[mw_msk > 1.] = 1.
    if apod_gal > 0.:
        gal_msk = hp.smoothing(gal_msk, fwhm=apod_gal, iter=5, verbose=False)
        gal_msk[gal_msk > 1.] = 1.; gal_msk[gal_msk < 0.] = 0.;
    if apod_star > 0.:
        star_msk = hp.smoothing(star_msk, fwhm=apod_star, iter=5, verbose=False)
        star_msk[star_msk > 1.] = 1.; star_msk[star_msk < 0.] = 0.;

    """
    # if we have already applied star mask
    if 'PointSrc' in mw_msk_fn:
        print("hopefully yes")
        star_msk *= 0.
        star_msk += 1.
    """
    
    # create combined mask
    if want_joint:
        cmb_msk = mw_msk*star_msk*gal_msk
        gal_msk = cmb_msk
    else:
        cmb_msk = mw_msk*star_msk

    # compute fsky for pure cmb, pure gal and mixed cmb+gal
    fsky_cmb = np.sum(cmb_msk**2)*1./len(cmb_msk) # needs to be squared
    fsky_gal = np.sum(gal_msk**2)*1./len(gal_msk) # needs to be squared
    fsky_mix = np.sum(cmb_msk*gal_msk)*1./len(cmb_msk) # needs to be squared 
    print("fsky = ", fsky_mix)
    
    # mask galaxy and CMB field
    gal_masked = gal_delta * gal_msk
    cmb_masked = cmb_mp * cmb_msk

    # load filter
    fl_ksz = np.load("kszsq_gal_data/Planck_filter_taper_kSZ.npy") # F(ell)
    ell_ksz = np.load("kszsq_gal_data/Planck_ell_kSZ.npy")
    fl_ksz /= np.max(fl_ksz)
    
    # apply filter
    pix_area = 4*np.pi/npix
    cmb_fltr_masked = hp.alm2map(hp.almxfl(hp.map2alm(cmb_masked, iter=3), fl_ksz), nside)
    if plot_moll:
        hp.mollview(cmb_fltr_masked, title="CMB filtered")
        #plt.show()
    
    # measure cross power spectrum
    cl_kszsq_gal = hp.anafast(cmb_fltr_masked**2, gal_masked, lmax=LMAX-1, pol=False)/fsky_mix
    cl_ksz_gal = hp.anafast(cmb_fltr_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky_mix
    cl_gal = hp.anafast(gal_masked, lmax=LMAX-1, pol=False)/fsky_gal
    cl_kszsq = hp.anafast(cmb_fltr_masked**2, lmax=LMAX-1, pol=False)/fsky_cmb
    cl_ksz = hp.anafast(cmb_fltr_masked, lmax=LMAX-1, pol=False)/fsky_cmb
    np.savez(f"kszsq_gal_data/cl_all_{galaxy_sample}_Planck.npz", cl_gal=cl_gal, cl_kszsq=cl_kszsq, cl_ksz_gal=cl_ksz_gal, cl_kszsq_gal=cl_kszsq_gal, cl_ksz=cl_ksz, ell=ell_data, fsky_mix=fsky_mix, fsky_cmb=fsky_cmb, fsky_gal=fsky_gal)

    # bin power spectra
    ell_binned = np.linspace(300, 2900, 14)
    _, cl_kszsq_gal_binned = bin_mat(ell_data, cl_kszsq_gal, ell_binned)
    _, cl_kszsq_binned = bin_mat(ell_data, cl_kszsq, ell_binned)
    _, cl_ksz_binned = bin_mat(ell_data, cl_ksz, ell_binned)
    _, cl_gal_binned = bin_mat(ell_data, cl_gal, ell_binned)
    ell_binned, cl_ksz_gal_binned = bin_mat(ell_data, cl_ksz_gal, ell_binned)
    print(ell_binned)
    
    # difference between ell bins
    diff = np.diff(ell_binned)
    diff = np.append(diff, diff[-1])
    print(diff.shape, ell_binned.shape)
    
    # compute gaussian errorbars # could maybe take out cl_kszsq_gal_binned**2
    cl_kszsq_gal_err = np.sqrt(1./((2.*ell_binned+1)*fsky_mix*diff)*(cl_kszsq_gal_binned**2+cl_gal_binned*cl_kszsq_binned))
    cl_ksz_gal_err = np.sqrt(1./((2.*ell_binned+1)*fsky_mix*diff)*(cl_ksz_gal_binned**2+cl_gal_binned*cl_ksz_binned))
    factor = ell_binned*(ell_binned+1)/(2.*np.pi)

    if plot_moll:
        plt.figure(1, figsize=(9, 7))
        plt.axhline(0., ls='--', c='black')
        plt.errorbar(ell_binned, cl_kszsq_gal_binned*factor, yerr=cl_kszsq_gal_err*factor, ls='', marker='o', capsize=4., color='dodgerblue')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell (\ell + 1) C_\ell^{T^2_{\rm clean}\times \delta_g}/2 \pi \ [\mu K^2]$')
        plt.savefig(f'kszsq_gal_figs/cl_kszsq_gal_{galaxy_sample}.png')
        #plt.show()

        plt.figure(2, figsize=(9, 7))
        plt.axhline(0., ls='--', c='black')
        plt.errorbar(ell_binned, cl_ksz_gal_binned*1.e5*ell_binned/np.pi, yerr=cl_ksz_gal_err*1.e5*ell_binned/np.pi, ls='', marker='o', capsize=4., color='dodgerblue')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell C_\ell^{T_{\rm clean}\times \delta_g}/\pi \ [\mu K]$')
        plt.savefig(f'kszsq_gal_figs/cl_ksz_gal_{galaxy_sample}.png')
        #plt.show()

        plt.figure(3, figsize=(9, 7))
        #plt.axhline(shotnoise, ls='--', c='black')
        plt.plot(ell_data, cl_gal)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{gg}$')
        plt.savefig('kszsq_gal_figs/cl_gal.png')
        plt.show()
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--galaxy_sample', '-gal', help='Which galaxy sample do you want to use?',
                        default=DEFAULTS['galaxy_sample'],
                        choices=["WISE", "DECALS"])
    parser.add_argument('--want_joint', '-joint', help='Treat CMB and galaxies with the same combined mask (MW, point src and galaxy) or separately', action='store_true')
    parser.add_argument('--apod_mw', help='Apodization scale for MW mask', type=float, default=10.)
    parser.add_argument('--apod_star', help='Apodization scale for point source mask', type=float, default=10.)
    parser.add_argument('--apod_gal', help='Apodization scale for galaxy mask (if joint=False, not used)', type=float, default=10.)
    args = vars(parser.parse_args())

    main(**args)
