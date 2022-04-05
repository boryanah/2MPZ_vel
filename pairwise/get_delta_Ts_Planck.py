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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from healpy.projector import GnomonicProj
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy import units as u
from pixell import enmap, utils, reproject
from classy import Class

from util import get_tzav_fast, GNFW_filter, calc_T_MF
from tools_pairwise_momentum import load_galaxy_sample
import rotfuncs

# defaults
DEFAULTS = {}
DEFAULTS['resCutoutArcmin'] = 0.1 #1.717 #0.05 #1.717 # stamp resolution
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

def main(galaxy_sample, resCutoutArcmin, Theta, data_dir, want_MF=False, plot=False, want_random=-1):

    # constants
    coord = 'G' # milky way at 0 0  (GE not working if not doing Gnomonic)
    cmb_sample = "Planck_healpix"
    nest = False # Ordering converted to RING # default
    vary_str = "fixed"
    mask_str = "_premask"
    mask_type = ""
    projCutout = 'cea'
    cmb_box = {'decfrom': -90., 'decto': 90., 'rafrom': 0., 'rato': 360.}
    sigma_z = 0.013
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
    dArcmin = np.ceil(2. * Theta * np.sqrt(2.))
    nx = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    ny = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    xsize = 2 * nx + 1
    ysize = 2 * ny + 1
    dxDeg = xsize * resCutoutArcmin / 60.
    dyDeg = ysize * resCutoutArcmin / 60.
    """
    xsize = int(np.ceil(Theta*np.sqrt(2.)/resCutoutArcmin))+1
    ysize = int(np.ceil(Theta*np.sqrt(2.)/resCutoutArcmin))+1
    dxDeg = xsize*(resCutoutArcmin/60.)
    dyDeg = ysize*(resCutoutArcmin/60.)
    """
    print("size of stamp in arcmin = ", dxDeg*60.)
    print("size of stamp in pixels = ", xsize)
    
    # load map and mask
    mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits"
    #mp_fn = data_dir+"/cmb_data/COM_CMB_IQU-smica_8192_R3.00_full.fits"
    mp = hp.read_map(mp_fn, verbose=True)
    mp *= 1.e6 # get in units of uK
    msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70.fits"
    #msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_Gal70_8192.fits"
    msk = hp.read_map(msk_fn, verbose=True)
    assert len(msk) == len(mp)

    # code used for bilinear interpolation
    """
    nside_new = 8192
    ipix = np.arange(hp.nside2npix(nside_new), dtype=int)
    lons, lats = hp.pix2ang(nside_new, ipix, lonlat=True, nest=nest)
    interp_vals = hp.get_interp_val(mp, theta=lons, phi=lats, nest=nest, lonlat=True)
    mp = interp_vals
    msk = hp.ud_grade(msk, nside_new)
    hp.write_map(data_dir+f"/cmb_data/COM_CMB_IQU-smica_{nside_new:d}_R3.00_full.fits", mp)
    hp.write_map(data_dir+f"/cmb_data/HFI_Mask_PointSrc_Gal70_{nside_new:d}.fits", msk)
    assert len(msk) == len(mp)
    """
    
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
    pix_area = 4*np.pi/npix
    pix_size_arcmin = np.sqrt(pix_area)*180./np.pi*60.
    choice = msk[ipix] == 1.
    index = index[choice]
    RA = RA[choice]
    DEC = DEC[choice]
    Z = Z[choice]
    B = B[choice]
    L = L[choice]
    vec = vec[choice]
    print("number after premasking = ", len(RA))
    print("average pixel size in arcmin = ", pix_size_arcmin)
    
    # filter to remove primary CMB
    """
    if Theta > 3.000001:
        # load filter
        fl_ksz = np.load("camb_data/Planck_filter_kSZ.npy")
        fl_ksz /= np.max(fl_ksz)
        ell_ksz = np.load("camb_data/Planck_ell_kSZ.npy")

        # apply filter
        pix_area = 4*np.pi/npix
        ipix_area = 1./pix_area
        mp = hp.alm2map(hp.almxfl(hp.map2alm(mp*msk, iter=3), fl_ksz), nside) #*ipix_area
        #hp.mollview(cmb_fltr_masked)
        #plt.show()
    """
    
    # initialize projection class
    # original size
    GP = GnomonicProj(rot=rot, coord=coord, xsize=xsize, ysize=ysize, reso=resCutoutArcmin)
    # TESTING!!!!!!!!!!!!!!
    """
    size_gp = 2*(((xsize-1)//2)//15)+1
    pix_size_arcmin = xsize/size_gp*resCutoutArcmin
    print("new res = ", pix_size_arcmin)
    GP = GnomonicProj(rot=rot, coord=coord, xsize=size_gp, ysize=size_gp, reso=pix_size_arcmin)
    dDeg_gp = size_gp*(pix_size_arcmin/60.)
    shape, wcs = enmap.geometry(np.array([[-0.5*dDeg_gp,-0.5*dDeg_gp],[0.5*dDeg_gp,0.5*dDeg_gp]])*utils.degree, res=pix_size_arcmin*utils.arcmin, proj=projCutout)
    small_mp_gp = enmap.zeros(shape, wcs)
    """

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
    posmap = cutoutMap.posmap() # pixel positions (centers)
    lats = posmap[0]/utils.degree # declination in degrees
    lons = posmap[1]/utils.degree # right ascension in degrees
    dlat = lats[1, 0]-lats[0, 0]
    dlon = lons[0, 1]-lons[0, 0]    
    pixmap = cutoutMap.pixmap()
    pixy = posmap[0]
    pixx = posmap[1]
    
    # create empty array for the apertures
    T_APs = np.zeros(len(RA))
    for i in range(len(RA)):
        if i % 100 == 0: print("i = ", i)
        
        # query radius in a map
        # faster way of doing things (no control over resolution) agrees for larger cutouts but not for smaller?
        """
        ipix_inner = hp.query_disc(nside, vec[i], radius, nest=nest) # radians
        ipix_outer = hp.query_disc(nside, vec[i], radius*np.sqrt(2.), nest=nest) # radians
        ipix_outer = ipix_outer[np.in1d(ipix_outer, ipix_inner, invert=True)]
        flux_inner = np.sum(mp[ipix_inner]*msk[ipix_inner])/np.sum(msk[ipix_inner])
        flux_outer = np.sum(mp[ipix_outer]*msk[ipix_outer])/np.sum(msk[ipix_outer])
        T_AP_query = flux_inner-flux_outer
        T_APs[i] = T_AP_query
        continue
        """
        
        # mirror-turned because of pixell (preferred: interpolates and changes resolution)
        """
        #t = time.time()
        interp_vals = hp.get_interp_val(mp, theta=L[i]+lons, phi=B[i]+lats, nest=nest, lonlat=True)
        small_mp[:, :] = interp_vals[:, ::-1]
        #print("time map interp = ", time.time()-t) # 0.006758451461791992
        #t = time.time()
        rot = (L[i], B[i], 0.) # psi is rotation along the line of sight
        small_msk = GP.projmap(map=msk, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        #print("time mask GP = ", time.time()-t) # 0.0014510154724121094
        cutoutMap[:, :] += small_mp[:, :]
        """

        # reprojecting with pixell
        
        rot = None
        #rot = "gal, equ"
        if rot == "gal, equ":
            if RA[i] > 180.:
                ra, dec = RA[i], DEC[i]
            else:
                ra, dec = RA[i]-360., DEC[i]
        else:
            if L[i] > 180.:
                ra, dec = L[i], B[i]
            else:
                ra, dec = L[i]-360., B[i] # Doesn't seem to make a difference
        ra, dec = ra*utils.degree, dec*utils.degree
        shape, wcs = enmap.geometry(shape=shape, res=resCutoutArcmin*utils.arcmin, pos=(dec, ra), deg=False, proj=projCutout) # docs: DEC, RA of projected to; 
        # bubas
        if i == 0:
            alm = reproject.enmap_from_healpix_buba_p1(mp, shape, wcs, ncomp=1, unit=1., lmax=5000, rot=rot) # spherical harmonics
        t = time.time()
        small_mp = reproject.enmap_from_healpix_buba_p2(alm, shape, wcs, ncomp=1, unit=1., lmax=5000, rot=rot)[0] # spherical harmonics
        #print("time map = ", time.time()-t) # 0.26482295989990234
        #small_mp = reproject.enmap_from_healpix(mp, shape, wcs, ncomp=1, unit=1., lmax=5000, rot=rot)[0] # spherical harmonics
        #small_mp = reproject.enmap_from_healpix_interp(mp, shape, wcs, interpolate=True, rot=rot) 
        #t = time.time()
        small_msk = reproject.enmap_from_healpix_interp(msk, shape, wcs, interpolate=False, rot=rot) # can toggle interpolate (False is rombs and True is same as above)
        #print("time mask = ", time.time()-t) # 0.01005697250366211
        #small_mp[:, :] = small_mp[:, ::-1]
        small_mp = small_mp
        cutoutMap += small_mp
        

        # get gnomonic projection (with interpolation, but weird edges)
        """
        rot = (L[i], B[i], 0.) # psi is rotation along the line of sight
        small_mp_gp[:, :] = GP.projmap(map=mp, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        small_msk_gp = GP.projmap(map=msk, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        
        # corresponding true coordinates on the big healpy map (last two zeros are RA and DEC)
        ipos = rotfuncs.recenter(posmap[::-1], [0, 0, 0, 0])[::-1]

        # Here, I use bilinear interpolation
        small_mp[:, :] = small_mp_gp.at(ipos, prefilter=True, mask_nan=False, order=3)
        assert small_mp.shape == cutoutMap.shape # x and y are switched!!!!
        cutoutMap += small_mp
        # TESTING
        small_msk = small_mp*0+1
        small_msk[small_mp == 0.] = 0.
        """

        """
        # clean version of gnomonic projection  (no interpolation, so everything is a romb)
        rot = (L[i], B[i], 0.) # psi is rotation along the line of sight
        small_mp[:, :] = GP.projmap(map=mp, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        small_msk = GP.projmap(map=msk, vec2pix_func=vec2pix_func, rot=rot, coord=coord)
        """
        
        # record aperture
        if want_MF:
            T_APs[i] = calc_T_MF(small_mp, fmap=fmap, mask=small_msk, power=power, test=False, apod_pix=int(10*0.05/resCutoutArcmin)) # used to be 20
            #T_APs[i] = calc_T_MF(small_mp, fmap=fmap, mask=None, power=power, test=True, apod_pix=0)
        else:
            flux_inner = np.sum((small_mp*small_msk)[inner])/np.sum(small_msk[inner])
            flux_outer = np.sum((small_mp*small_msk)[outer])/np.sum(small_msk[outer])
            T_APs[i] = flux_inner - flux_outer
            #T_APs[i] = np.mean(small_mp[inner]) - np.mean(small_mp[outer])
        
        if plot:
            #print(np.mean(mp[ipix_inner]), np.mean(small_mp[inner])) # enable query disk
            #print(np.mean(mp[ipix_outer]), np.mean(small_mp[outer])) # enable query disk
            #print("T_AP query = ", T_AP_query, len(ipix_inner), len(ipix_outer)) # enable query disk
            print("T_AP inter = ", T_APs[i], np.sum(inner), np.sum(outer))
            
            plt.figure(1)
            plt.imshow(small_mp)
            fig = plt.gcf()
            ax = fig.gca()
            circle = plt.Circle((xsize*0.5, xsize*0.5), Theta/resCutoutArcmin, fill=False, color='red')
            ax.add_patch(circle)
            circle = plt.Circle((xsize*0.5, xsize*0.5), Theta/resCutoutArcmin*np.sqrt(2.), fill=False, color='yellow')
            ax.add_patch(circle)
            plt.colorbar()
            #plt.savefig(f"figs/cutout_obj{i:04d}_Th{Theta:.2f}_res{resCutoutArcmin:.2f}_gal{galaxy_sample}_cmb{cmb_sample}.png")
            #plt.close()

            plt.figure(2)
            plt.imshow(small_msk)
            plt.colorbar()
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
    choice = (T_APs != 0.) & (~np.isnan(T_APs))
    T_APs = T_APs[choice]
    index = index[choice]
    Z = Z[choice]
    print("percentage T_APs == 0. or nans = ", np.sum(~choice)*100./len(RA), np.sum(~choice))
    
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
    parser.add_argument('--plot', help='Want to plot every stamp', action='store_true')
    parser.add_argument('--want_random', '-rand', help='Random seed to shuffle galaxy positions (-1 does not randomize)', type=int, default=-1)
    parser.add_argument('--data_dir', help='Directory where the data is stored', default=DEFAULTS['data_dir'])
    args = vars(parser.parse_args())

    main(**args)
