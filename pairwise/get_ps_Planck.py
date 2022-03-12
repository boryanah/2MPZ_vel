#!/usr/bin/env python3
"""
Usage
-----
$ ./get_ps_Planck.py --help
"""

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

from util import get_tzav_fast
from tools_pairwise_momentum import load_galaxy_sample
from utils_power import bin_mat

# defaults
DEFAULTS = {}
DEFAULTS['resCutoutArcmin'] = 0.1 # stamp resolution
DEFAULTS['galaxy_sample'] = "2MPZ"
DEFAULTS['Theta'] = 3.

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

def main(galaxy_sample, resCutoutArcmin, Theta):

    # constants
    coord = 'G' # milky way at 0 0  (GE not working if not doing Gnomonic)
    cmb_sample = "Planck_healpix"
    nest = False # Ordering converted to RING # default
    vary_str = "fixed"
    mask_str = "_premask"
    mask_type = ""
    projCutout = 'cea'
    cmb_box = {'decfrom': -90., 'decto': 90., 'rafrom': 0., 'rato': 360.}
    sigma_z = 0.01
    data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
    plot = False
    want_random = -1
    if want_random != -1:
        print("Requested using random galaxy positions, forcing 2MPZ-like sample")
        galaxy_sample = "2MPZ"
        rand_str = f"_rand{want_random:d}"
    else:
        rand_str = ""
    rot = (0., 0., 0.)
    
    # choices
    radius = Theta*utils.arcmin # radians

    print("Theta, res, gal", Theta, resCutoutArcmin, galaxy_sample)

    # size of the cutouts
    xsize = int(np.ceil(2.*Theta*np.sqrt(2.)/resCutoutArcmin))+1
    ysize = int(np.ceil(2.*Theta*np.sqrt(2.)/resCutoutArcmin))+1
    dxDeg = xsize*(resCutoutArcmin/60.)
    dyDeg = ysize*(resCutoutArcmin/60.)
    print("size in deg = ", dxDeg)
    print("size in pixels = ", xsize)
    
    # load map and mask
    mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    mp = hp.read_map(mp_fn, verbose=True)
    mp *=  1.e6 # uK
    msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
    #msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"
    msk = hp.read_map(msk_fn, verbose=True)
    assert len(msk) == len(mp)
    
    # create instance of the class "Class"
    Cosmo = Class()
    Cosmo.set(COSMO_DICT)
    Cosmo.compute() 

    # load galaxies                                                                                                   
    RA, DEC, Z, index = load_galaxy_sample(Cosmo, galaxy_sample, cmb_sample, data_dir, cmb_box, want_random)
    c_icrs = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, frame='icrs') # checked
    B = c_icrs.galactic.b.value # -90, 90
    L = c_icrs.galactic.l.value # 0, 360
    vec = hp.ang2vec(theta=L, phi=B, lonlat=True)

    print("number before premasking = ", len(RA))
    # apply premasking 
    #x = np.cos(B*utils.degree)*np.cos(L*utils.degree)
    #y = np.cos(B*utils.degree)*np.sin(L*utils.degree)
    #z = np.sin(B*utils.degree) # equiv to vec
    npix = len(msk)
    nside = hp.npix2nside(npix)
    #ipix = hp.pixelfunc.vec2pix(nside, x, y, z)
    ipix = hp.pixelfunc.vec2pix(nside, *vec.T)
    choice = msk[ipix] == 1.
    index = index[choice]
    RA = RA[choice]
    DEC = DEC[choice]
    Z = Z[choice]
    B = B[choice]
    L = L[choice]
    print("number after premasking = ", len(RA))

    # apply apodization
    smooth_scale_mask = 0.05 #0.05 # in radians
    if smooth_scale_mask > 0.:
        msk = msk*hp.smoothing(msk, fwhm=smooth_scale_mask, iter=5, verbose=False)     
    fsky = np.sum(msk)*1./len(msk)
    print("fsky = ", fsky)
    
    # power parameters
    LMIN = 100
    LMAX = 3*nside+1
    ell = np.arange(0, LMAX, 1)

    # compute power spectrum using anafast dividing by fsky
    #cmb_masked = hp.ma(mp)
    cmb_masked = mp*msk
    #cmb_masked.mask = np.logical_not(msk)
    hp.mollview(cmb_masked)
    cl_tt = hp.anafast(cmb_masked, lmax=LMAX-1, pol=False) # uK^2
    cl_tt /= fsky
    print("fsky =", fsky)

    # compute power of galaxies
    gal_counts = np.zeros(npix, dtype=int)
    print("npix = ", npix)
    ipix_un, counts = np.unique(ipix, return_counts=True)
    gal_counts[ipix_un] = counts

    gal_mean = np.mean(gal_counts[msk.astype(bool)])
    gal_delta = gal_counts/gal_mean - 1.

    #gal_masked = hp.ma(gal_delta)
    #gal_masked.mask = np.logical_not(msk)
    gal_masked = gal_delta * msk
    hp.mollview(np.log10(gal_masked+2.), cmap='seismic')
    plt.show()
    
    # compute shotnoise
    gal_nbar = gal_mean/hp.nside2pixarea(nside)
    N_gg = 1/gal_nbar * np.ones_like(ell) #galaxy shot noise
    print("number of galaxies per deg^2 = ", gal_nbar/(180.**2/np.pi**2))
    print("shotnoise = ", N_gg)

    cl_gg = hp.anafast(gal_masked, lmax=LMAX - 1, pol=False)
    cl_gg /= fsky

    # load cmb power from theory
    camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
    cltt = camb_theory[0, 0, :3000]
    ls = np.arange(cltt.size)
    ell_ksz, cl_ksz = np.loadtxt("camb_data/cl_ksz_bat.dat", unpack=True)

    # bin power
    np.save("camb_data/Planck_power.npy", cl_tt)
    np.save("camb_data/Planck_ell.npy", ell)
    bins = np.linspace(100, 4000, 40)
    cl_tt_binned, ell_binned = bin_mat(ell, cl_tt, bins)

    # create kSZ filter
    #f_ell = np.interp(ell_binned, ell_ksz, cl_ksz)/cl_tt_binned
    f_ell = np.interp(ell, ell_ksz, cl_ksz)/np.interp(ell, ell_binned, cl_tt_binned)
    f_ell[(ell < 100) | (ell > 3000)] = 0.
    
    plot_filter = True
    if plot_filter:
        #plt.plot(ell, f_ell*ell**2, label='filter')
        #plt.plot(ell_ksz[ell_ksz < 3000], (cl_ksz)[ell_ksz < 3000])
        #plt.plot(ell, cl_tt*ell**2)
        plt.plot(ell_binned, cl_tt_binned*ell_binned**2)
        plt.plot(ls, cltt*ls**2, lw=3, color='k') # D_ell = C_ell l (l+1)/2pi (muK^2)
        plt.legend()
        plt.show()

    pix_area = 4*np.pi/npix
    ipix_area = 1./pix_area
    cmb_fltr_masked = hp.alm2map(hp.almxfl(hp.map2alm(cmb_masked, iter=3), f_ell), nside)*ipix_area
    
    # measure cross power spectrum
    cl_kszsq_gal = hp.anafast(cmb_fltr_masked**2, gal_masked, lmax=LMAX-1, pol=False)
    cl_kszsq_gal_binned, ell_binned = bin_mat(ell, cl_kszsq_gal, bins)
    
    plt.figure(1)
    plt.plot(ell_binned, cl_kszsq_gal_binned*ell_binned*(ell_binned+1.)/(np.pi*2.))
    
    # plot and compare with theory
    plt.figure(2)
    plt.plot(ell, cl_tt*ell*(ell+1.)/(np.pi*2.))
    plt.plot(ls, cltt*ls*(ls+1)/(np.pi*2.), lw=3, color='k') # D_ell = C_ell l (l+1)/2pi (muK^2)
    #plt.show()

    plt.figure(3)
    plt.plot(ell, (cl_gg-N_gg)*ell*(ell+1.)/(np.pi*2.), ls='--')
    plt.plot(ell, cl_gg*ell*(ell+1.)/(np.pi*2.))
    plt.show()
    quit()

    np.save(delta_T_fn, delta_Ts)
    np.save(index_fn, index)

class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":
    
    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--Theta', '-Th', help='Aperture radius in arcmin', type=float, default=DEFAULTS['Theta'])
    parser.add_argument('--resCutoutArcmin', help='Resolution of the cutout', type=float, default=DEFAULTS['resCutoutArcmin'])
    parser.add_argument('--galaxy_sample', '-gal', help='Which galaxy sample do you want to use?',
                        default=DEFAULTS['galaxy_sample'],
                        choices=["BOSS_South", "BOSS_North", "2MPZ", "SDSS_L43D", "SDSS_L61D",
                                 "SDSS_L43", "SDSS_L61", "SDSS_L79", "SDSS_all", "eBOSS_SGC", "eBOSS_NGC"])
    args = vars(parser.parse_args())

    main(**args)
