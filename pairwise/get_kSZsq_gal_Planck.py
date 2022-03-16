#!/usr/bin/env python3
"""
Usage
-----
$ ./get_kSZsq_gal_Planck.py --help
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
    msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
    mp = hp.read_map(mp_fn, verbose=True)
    msk = hp.read_map(msk_fn, verbose=True)
    mp *=  1.e6 # uK
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

    # power parameters
    LMIN = 0
    LMAX = 3*nside+1
    ell_data = np.arange(LMIN, LMAX, 1)
    ell_data_binned = np.linspace(100, 3000, 300)
    
    # apply apodization
    smooth_scale_mask = 0. #0.05 # in radians
    if smooth_scale_mask > 0.:
        msk = msk*hp.smoothing(msk, fwhm=smooth_scale_mask, iter=5, verbose=False)     
    fsky = np.sum(msk)*1./len(msk)
    print("fsky = ", fsky)

    # compute galaxy overdensity
    #nside = 128; npix = hp.nside2npix(nside); ipix = hp.pixelfunc.vec2pix(nside, *vec.T) # TESTING
    gal_counts = np.zeros(npix, dtype=int)
    ipix_un, counts = np.unique(ipix, return_counts=True)
    gal_counts[ipix_un] = counts

    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #msk[gal_counts == 0.] = 0.
    
    gal_mean = np.mean(gal_counts*msk)
    gal_delta = gal_counts.copy()
    gal_delta[msk > 0.] = gal_counts[msk > 0.]/gal_mean - 1.
    """
    # TESTING
    gal_mean = np.mean(gal_counts)
    gal_delta = gal_counts.copy()
    gal_delta = gal_counts/gal_mean - 1.
    """
    hp.mollview(np.log10(gal_delta+2.), cmap='seismic')
    hp.mollview(gal_delta, cmap='seismic')
    hp.mollview(gal_counts, cmap='seismic')
    plt.show()
    
    
    # mask galaxy and CMB field
    gal_masked = gal_delta * msk
    cmb_masked = mp * msk
    
    # compute shotnoise
    gal_nbar = gal_mean/hp.nside2pixarea(nside)
    N_gg = 1/gal_nbar * np.ones_like(ell_data) #galaxy shot noise
    print("number of galaxies per deg^2 = ", gal_nbar/(180.**2/np.pi**2))
    print("shotnoise = ", N_gg[0])

    # load filter
    fl_ksz = np.load("camb_data/Planck_filter_kSZ.npy") # F(ell)
    ell_ksz = np.load("camb_data/Planck_ell_kSZ.npy")
    fl_ksz /= np.max(fl_ksz)
    
    # apply filter
    pix_area = 4*np.pi/npix
    cmb_fltr_masked = hp.alm2map(hp.almxfl(hp.map2alm(cmb_masked, iter=3), fl_ksz), nside)
    hp.mollview(cmb_fltr_masked)
    plt.show()
    
    # measure cross power spectrum
    cl_kszsq_gal = hp.anafast(cmb_fltr_masked**2, gal_masked, lmax=LMAX-1, pol=False)/fsky
    cl_ksz_gal = hp.anafast(cmb_fltr_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky
    cl_kszsq = hp.anafast(gal_masked, lmax=LMAX-1, pol=False)/fsky
    cl_gal = hp.anafast(cmb_fltr_masked**2, lmax=LMAX-1, pol=False)/fsky
    np.savez(f"kszsq_gal_data/cl_all_{galaxy_sample}_Planck.npz", cl_gal=cl_gal, cl_kszsq=cl_kszsq, cl_ksz_gal=cl_ksz_gal, cl_kszsq_gal=cl_kszsq_gal, ell=ell_data)
    ell_data_binned = np.linspace(100, 3000, 15)
    ell_data_binned, cl_kszsq_gal_binned = bin_mat(ell_data, cl_kszsq_gal, ell_data_binned)
    
    plt.figure(1, figsize=(9, 7))
    plt.plot(ell_data_binned, cl_kszsq_gal_binned*ell_data_binned*(ell_data_binned+1.)/(np.pi*2.))
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell (\ell + 1) C_\ell^{T^2_{\rm clean}\times \delta_g}/2 \pi \ [\mu K^2]$')
    plt.savefig('kszsq_gal_figs/cl_kszsq_gal.png')
    #plt.show()

    plt.figure(2, figsize=(9, 7))
    plt.axhline(N_gg[0], ls='--', c='black')
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
    parser.add_argument('--Theta', '-Th', help='Aperture radius in arcmin', type=float, default=DEFAULTS['Theta'])
    parser.add_argument('--resCutoutArcmin', help='Resolution of the cutout', type=float, default=DEFAULTS['resCutoutArcmin'])
    parser.add_argument('--galaxy_sample', '-gal', help='Which galaxy sample do you want to use?',
                        default=DEFAULTS['galaxy_sample'],
                        choices=["BOSS_South", "BOSS_North", "2MPZ", "SDSS_L43D", "SDSS_L61D", "2MPZ_Biteau", "WISExSCOS",
                                 "SDSS_L43", "SDSS_L61", "SDSS_L79", "SDSS_all", "eBOSS_SGC", "eBOSS_NGC", "DECALS"])
    args = vars(parser.parse_args())

    main(**args)
