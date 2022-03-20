#!/usr/bin/env python3
"""
Usage
-----
$ ./get_delta_Ts_Planck.py --help
"""

import time
import argparse

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from healpy.projector import GnomonicProj
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy import units as u
from pixell import enmap, utils
from classy import Class

from util import get_tzav_fast, GNFW_filter, calc_T_MF
from tools_pairwise_momentum import load_galaxy_sample

# defaults
DEFAULTS = {}
DEFAULTS['resCutoutArcmin'] = 0.1 # stamp resolution
DEFAULTS['galaxy_sample'] = "2MPZ"
DEFAULTS['Theta'] = 3.0
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

def main(galaxy_sample, resCutoutArcmin, Theta, data_dir, want_MF=False, want_random=-1):

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
    plot = False
    if want_random != -1:
        print("Requested using random galaxy positions, forcing 2MPZ-like sample")
        galaxy_sample = "2MPZ"
        rand_str = f"_rand{want_random:d}"
    else:
        rand_str = ""
    MF_str = "MF" if want_MF else ""
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
    mp *= 1.e6 # get in units of uK
    msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
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

    # TESTING!!!!!!!!!!!!!!!!!!!!!!!!
    """
    if Theta > 3.000001:
        # load filter
        fl_ksz = np.load("camb_data/Planck_filter_kSZ.npy")
        # TESTING
        fl_ksz /= np.max(fl_ksz)
        ell_ksz = np.load("camb_data/Planck_ell_kSZ.npy")

        # apply filter
        pix_area = 4*np.pi/npix
        ipix_area = 1./pix_area # normalization needed???
        mp = hp.alm2map(hp.almxfl(hp.map2alm(mp*msk, iter=3), fl_ksz), nside) #*ipix_area
        #hp.mollview(cmb_fltr_masked)
        #plt.show()
    """
    
    # initialize projection class
    GP = GnomonicProj(rot=rot, coord=coord, xsize=xsize, ysize=ysize, reso=resCutoutArcmin)

    # create cutout map and find the pixels correspoding to the inner and outer disks (only used for modrmap)
    shape, wcs = enmap.geometry(np.array([[-0.5*dxDeg,-0.5*dyDeg],[0.5*dxDeg,0.5*dyDeg]])*utils.degree, res=resCutoutArcmin*utils.arcmin, proj=projCutout)
    cutoutMap = enmap.zeros(shape, wcs)
    small_mp = enmap.zeros(shape, wcs)
    modrmap = cutoutMap.modrmap()
    modlmap = cutoutMap.modlmap()
    inner = modrmap < radius
    outer = (modrmap >= radius) & (modrmap < np.sqrt(2.)*radius)

    if want_MF:
        fmap = GNFW_filter(modrmap, theta_500=Theta)
        binned_power = np.load(f"camb_data/Planck_binned_power.npy")
        centers = np.load(f"camb_data/Planck_centers.npy")
        power = interpolate.interp1d(centers, binned_power, bounds_error=False, fill_value=0)(modlmap)
        power[modlmap == 0.] = 0.
        
    def vec2pix_func(x, y, z, nside=nside):
        pix = hp.vec2pix(nside, x, y, z)
        return pix

    if want_MF:
        delta_T_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_{vary_str}Th{Theta:.2f}_delta_Ts.npy"
        index_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{MF_str}_{vary_str}Th{Theta:.2f}_index.npy"
    else:
        delta_T_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_delta_Ts.npy"
        index_fn = f"data/{galaxy_sample}{mask_type}{mask_str}{rand_str}_{cmb_sample}_{vary_str}Th{Theta:.2f}_index.npy"

    # bilinear interpolation
    posmap = cutoutMap.posmap()
    lats = posmap[0]/utils.degree # declination in radians
    lons = posmap[1]/utils.degree # right ascension in radians
    pixmap = cutoutMap.pixmap()
    pixy = posmap[0]
    pixx = posmap[1]

    # TESTING!!!!!!!!!!!!!!!
    #RA = RA[:3000]
    
    # create empty array for the apertures
    T_APs = np.zeros(len(RA))
    for i in range(len(RA)):
        if i % 1000 == 0: print("i = ", i)

        # query radius in a map
        # faster way of doing things (no control over resolution) agrees for larger cutouts but not for smaller?
        """
        ipix_inner = hp.query_disc(nside, vec[i], radius, nest=nest) # radians
        ipix_outer = hp.query_disc(nside, vec[i], radius*np.sqrt(2.), nest=nest) # radians
        print("query radius map = ", np.mean(mp[ipix_inner]), len(ipix_inner))
        print("query radius msk = ", np.mean(msk[ipix_inner]))
        
        # record aperture
        T_APs[i] = np.mean(mp[ipix_inner]*msk[ipix_inner]) - np.mean(mp[ipix_outer]*msk[ipix_outer])
        #T_APs[i] = np.mean(mp[ipix_inner]) - np.mean(mp[ipix_outer])
        """

        # mirror-turned because of pixell (preferred: interpolates and changes resolution)
        interp_vals = hp.get_interp_val(mp, theta=L[i]+lons, phi=B[i]+lats, nest=nest, lonlat=True)
        small_mp[:, :] = interp_vals[:, ::-1]
        rot = (L[i], B[i], 0.) # psi is rotation along the line of sight
        small_msk = GP.projmap(map=msk, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        cutoutMap += small_mp
                
        # get gnomonic projection (no interpolation, so everything is a romb)
        """
        rot = (L[i], B[i], 0.) # psi is rotation along the line of sight
        small_mp = GP.projmap(map=mp, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        small_msk = GP.projmap(map=msk, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        assert small_mp.shape == cutoutMap.shape # x and y are switched!!!!
        cutoutMap += small_mp
        #print("proj map = ", np.mean(small_mp[inner]))
        #print("proj map = ", np.mean(small_msk[inner]))
        """
        
        # record aperture
        if want_MF:
            T_APs[i] = calc_T_MF(small_mp, fmap=fmap, mask=None, power=power, test=False, apod_pix=10)
            #T_APs[i] = calc_T_MF(small_mp, fmap=fmap, mask=None, power=power, test=True, apod_pix=0)
        else:
            T_APs[i] = np.mean((small_mp*small_msk)[inner]) - np.mean((small_mp*small_msk)[outer])
            #T_APs[i] = np.mean(small_mp[inner]) - np.mean(small_mp[outer])

        if plot:
            plt.figure()
            plt.imshow(small_mp)
            plt.colorbar()
            plt.savefig(f"figs/cutout_obj{i:04d}_Th{Theta:.2f}_res{resCutoutArcmin:.2f}_gal{galaxy_sample}_cmb{cmb_sample}.png")
            plt.close()
    plt.show()

    plt.figure(figsize=(12,12))
    #plt.figure()
    fig = plt.gcf()
    ax = fig.gca()
    plt.imshow(cutoutMap/len(RA))
    print((xsize*0.5, xsize*0.5), Theta/resCutoutArcmin, Theta/resCutoutArcmin*np.sqrt(2.))
    circle = plt.Circle((xsize*0.5, xsize*0.5), Theta/resCutoutArcmin, fill=False, color='red')
    #circle = plt.Circle((0, 0), (Theta/60.)/(dxDeg/2.)*xsize*0.5, fill=False, color='red')
    ax.add_patch(circle)
    circle = plt.Circle((xsize*0.5, xsize*0.5), Theta/resCutoutArcmin*np.sqrt(2.), fill=False, color='yellow')
    #circle = plt.Circle((0, 0), (Theta*np.sqrt(2.)/60.)/(dxDeg/2.)*xsize*0.5, fill=False, color='yellow')
    ax.add_patch(circle)
    plt.xlim([0, xsize])
    plt.ylim([0, xsize])
    ax.set_aspect('equal')        
    plt.colorbar()
    plt.savefig(f"figs/stack_Th{Theta:.2f}_res{resCutoutArcmin:.2f}_gal{galaxy_sample}_cmb{cmb_sample}.png")
    plt.close()
    
    # apply cuts because of masking
    choice = T_APs != 0.
    T_APs = T_APs[choice]
    index = index[choice]
    Z = Z[choice]
    print("percentage T_APs == 0. ", np.sum(~choice)*100./len(RA))

    # get the redshift-weighted apertures and temperature decrement around each galaxy                            
    bar_T_APs = get_tzav_fast(T_APs, Z, sigma_z)
    delta_Ts = T_APs - bar_T_APs

    # save apertures and indices                                                                                  
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
                        choices=["BOSS_South", "BOSS_North", "2MPZ", "SDSS_L43D", "SDSS_L61D", "2MPZ_Biteau",
                                 "SDSS_L43", "SDSS_L61", "SDSS_L79", "SDSS_all", "eBOSS_SGC", "eBOSS_NGC"])
    parser.add_argument('--want_MF', '-MF', help='Want to use matched filter', action='store_true')
    parser.add_argument('--want_random', '-rand', help='Random seed to shuffle galaxy positions (-1 does not randomize)', type=int, default=-1)
    parser.add_argument('--data_dir', help='Directory where the data is stored', default=DEFAULTS['data_dir'])
    args = vars(parser.parse_args())

    main(**args)
