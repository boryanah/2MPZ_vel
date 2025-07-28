import glob
import os

import matplotlib.pyplot as plt
import numpy as np

import pyccl as ccl
import healpy as hp
from astropy.io import fits

from utils_power import bin_mat
from tools_pairwise_momentum import get_lum
colors = ['c', 'm', 'y', 'k', 'r', 'g', 'b', 'orange', 'purple', 'brown']


# define Cosmology object
cosmo_dic = {'h': 0.6736, 'Omega_c': 0.26447, 'Omega_b': 0.04930, 'A_s': 2.083e-9, 'n_s': 0.9649, 'T_CMB': 2.7255, 'Neff': 2.0328, 'm_nu': 0.06, 'm_nu_type': 'single', 'transfer_function': 'boltzmann_class'}
cosmo = ccl.Cosmology(**cosmo_dic)

# file names
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
gal_fn = data_dir+"/sdss_data/post_catalog.dr72bright0.fits"
rand_fns = glob.glob(data_dir+"/sdss_data/random-*.dr72bright.fits")
delta_fn = data_dir+"/sdss_data/delta_dr72bright0.fits"
mask_fn = data_dir+"/sdss_data/mask_lores_dr72bright0.fits"
kcorr_fn = data_dir+"/sdss_data/kcorrect.nearest.petro.z0.10.fits"

# load catalog
hdul = fits.open(gal_fn)
RA = hdul[1].data['RA'].flatten() # 0, 360
DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 70
Z = hdul[1].data['Z'].flatten()
M = hdul[1].data['M'].flatten()
ABSM = hdul[1].data['ABSM'][:, 2].flatten() # ugrizJKY (r)
OBJECT_POSITION = hdul[1].data['OBJECT_POSITION'].flatten()
hdul = fits.open(kcorr_fn)
KCORRECT = hdul[1].data['KCORRECT']
print(ABSM[:100])
print(hdul[1].data['ABSMAG'][OBJECT_POSITION][:100])
lum = get_lum(Z, M, OBJECT_POSITION, KCORRECT)
i_sort = np.argsort(ABSM)
i_sort_rev = np.argsort(i_sort)
print(i_sort[:100])
i_sort = np.argsort(lum)[::-1]
print(i_sort[:100])
n_subs = 8
n_jump = len(RA)//n_subs
RA = RA[i_sort]
DEC = DEC[i_sort]
Z = Z[i_sort]
lum = lum[i_sort]
print("maximum redshift = ", np.max(Z))
print("number of objects = ", len(RA))



if os.path.exists(data_dir+"/sdss_data/random_comb.dr72bright.npz"):
    data = np.load(data_dir+"/sdss_data/random_comb.dr72bright.npz")
    RA_rand = data['RA']
    DEC_rand = data['DEC']
else:
    RA_rand = np.zeros(70*len(RA))
    DEC_rand = np.zeros(70*len(DEC))
    sum = 0
    for rand_fn in rand_fns:
        hdul = fits.open(rand_fn)
        RA_rand_this = hdul[1].data['RA'].flatten() # 0, 360
        DEC_rand_this = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 70
        print("number of randoms/objects = ", len(RA_rand_this)/len(RA))
        RA_rand[sum:sum + len(RA_rand_this)] = RA_rand_this
        DEC_rand[sum:sum + len(DEC_rand_this)] = DEC_rand_this
        sum += len(DEC_rand_this)
    DEC_rand = DEC_rand[:sum]
    RA_rand = RA_rand[:sum]
    np.savez(data_dir+"/sdss_data/random_comb.dr72bright.npz", RA=RA_rand, DEC=DEC_rand)
    
# healpy params
nside = 2048
npix = hp.nside2npix(nside)
nside_lo = 256
ell_bins = np.linspace(1, 1500, 30)
z_edges = np.linspace(0., 0.5, 1001)

def get_random(RA_rand, DEC_rand):
    # 2. Compute the same thing for the randoms. Call this nr
    ipix = hp.ang2pix(nside, RA_rand, DEC_rand, lonlat=True)
    nr = np.bincount(ipix, minlength=npix)

    # 4. Compute a low-resolution version of nr. This is so you can divide by it safely when computing delta
    nr_lo = hp.ud_grade(hp.ud_grade(nr.astype(float), nside_out=nside_lo), nside_out=nside)

    # 5. Find the "good pixels", i.e. those that aren't empty in nr_lo:
    goodpix = nr_lo > 0
    return nr, nr_lo, goodpix

def get_delta(RA_this, DEC_this, nr, nr_lo, goodpix):
    # 1. Compute a map at the desired resolution that contains the number of galaxies per pixel
    ipix = hp.ang2pix(nside, RA_this, DEC_this, lonlat=True)
    nd = np.bincount(ipix, minlength=npix)
    
    # 3. Compute alpha = (number of data)/(number of random)
    alpha = len(RA_this)/len(RA_rand)

    # 6. Finally compute delta as:
    delta = np.zeros(npix)
    delta[goodpix] = (nd[goodpix]-alpha*nr[goodpix])/(alpha*nr_lo[goodpix])

    return delta

# get the random shit
nr, nr_lo, goodpix = get_random(RA_rand, DEC_rand)


if os.path.exists(mask_fn):
    os.remove(mask_fn)
mask = goodpix.astype(float)
#hp.write_map(mask_fn, mask)
fsky = np.sum(mask**2)/len(mask)

sum = 0
bias = np.zeros(len(RA))
for i in range(n_subs):
    if i == n_subs-1:
        delta = get_delta(RA[sum:], DEC[sum:], nr, nr_lo, goodpix)
        Z_this = Z[sum:]
    else:
        delta = get_delta(RA[sum:sum+n_jump], DEC[sum:sum+n_jump], nr, nr_lo, goodpix)
        Z_this = Z[sum:sum+n_jump]
    print(len(Z_this))
    
    delta_fn = data_dir+f"/sdss_data/delta_dr72bright0_{i:02d}.fits"
    if os.path.exists(delta_fn):
        os.remove(delta_fn)

    print("MAKETI")
        
    # save stuff
    #hp.write_map(delta_fn, delta)
    gal_masked = delta*mask
    cl_gal = hp.anafast(gal_masked, lmax=int(ell_bins[-1]), pol=False)/fsky

    # bin power spectrum
    ell_data = np.arange(int(ell_bins[-1])+1)
    ell_binned, cl_gal_binned = bin_mat(ell_data, cl_gal, ell_bins)
    
    # compute N(z)
    dNdz, _ = np.histogram(Z_this, bins=z_edges)
    dNdz = dNdz.astype(np.float64)
    dNdz /= np.max(dNdz)
    z = 0.5*(z_edges[1:] + z_edges[:-1])
    
    # calculate theoretical Cls
    #b = 1./ccl.background.growth_factor(cosmo, 1./(1+z))
    #b /= b[0]
    b = np.ones_like(z)
    gal = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, dNdz), bias=(z, b))
    cls_gal_th = ccl.angular_cl(cosmo, gal, gal, ell_binned)
    
    # save result
    #np.savez(data_dir+f"/sdss_data/cl_gal_{i:02d}.npz", cl_gal_data=cl_gal_binned, cl_gal_theory=cls_gal_th, ell=ell_binned)
    plt.plot(ell_binned, np.sqrt(cl_gal_binned/cls_gal_th), c=colors[i], label=f'{i:d}')
    b = np.sqrt(cl_gal_binned/cls_gal_th)[np.argmin(np.abs(ell_binned-150.))]
    if i == n_subs-1:
        bias[sum:sum+n_jump] = b
    else:
        bias[sum:] = b
    sum += n_jump
print(bias) 
np.savez(data_dir+f"/sdss_data/bias_dr72bright0.npz", bias=bias[i_sort_rev], lum=lum[i_sort_rev])
plt.legend()
plt.show()

quit()

# load galaxy delta map

#low_nside = 8#64
#low_npix = hp.nside2npix(low_nside)
#low_ipix = hp.ang2pix(low_nside, RA, DEC, lonlat=True)
#low_nmap = np.bincount(low_ipix, minlength=low_npix)
#gal_msk[low_nmap > 0.] = 1.
#low_ipix = np.arange(low_npix)
#RA, DEC = hp.pix2ang(low_nside, low_ipix, lonlat=True)
#gal_msk = np.zeros(low_npix)
#in_mask = mangle_msk.contains(RA, DEC)
#in_mask = mangle_msk.contains_numba(RA, DEC)
#print("Inside mask: %d/%d" % (np.sum(in_mask), len(in_mask)))
#gal_msk[~in_mask] = 1.
#gal_msk = hp.ud_grade(gal_msk, nside)
#nmean = np.sum(nmap*gal_msk)/np.sum(gal_msk)
factor = 1
#mangle_msk = pymangle.Mangle(data_dir+"/sdss_data/mask.dr72bright0.ply")
mangle_msk = pymangle.Mangle(data_dir+"/sdss_data/window.dr72bright0.ply")
RA, DEC = mangle_msk.genrand(len(RA)*factor)
np.savez("tf", RA=RA, DEC=DEC)
print("generated randoms")
ipix = hp.ang2pix(nside, RA, DEC, lonlat=True)
nrand = np.bincount(ipix, minlength=npix)/factor
print("generated random map")
gal_msk = (nrand > 0.).astype(np.float64)
gal_delta = np.zeros(npix)
gal_delta[nrand > 0.] = nmap[nrand > 0.]/nrand[nrand > 0.] - 1.
print("mean gal delta = ", np.sum(gal_delta*gal_msk)/np.sum(gal_msk))
hp.mollview(gal_msk)
plt.show()
hp.mollview(gal_delta)
plt.show()

hp.write_map()
