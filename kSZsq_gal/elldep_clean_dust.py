"""
Conversion factor taken from: https://irsa.ipac.caltech.edu/docs/knowledgebase/ercsc-validation-conversions.pdf. There is a typo for Table V, 545 GHz (Tthermo to MJy/sr): 57 -> 571.
"""
import os
import sys
sys.path.append('../pairwise')

import numpy as np
import healpy as hp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from utils_power import bin_mat

# names of stuff
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
galaxy_sample = "DECALS"
#T_cmb = hp.read_map(data_dir+"/cmb_data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits", verbose=True) # Kelvin
T_cmb = hp.read_map(data_dir+"/cmb_data/LGMCA/WPR2_CMB_muK.fits", verbose=True)/1.e6 # Kelvin
T_545 = hp.read_map(data_dir+"/cmb_data/dust/HFI_SkyMap_545-field-Int_2048_R3.00_full.fits") # MJy/sr
T_545 /= 571.943 # Kelvin
print("min max cmb = ", np.min(T_cmb), np.max(T_cmb))
print("min max 545 = ", np.min(T_545), np.max(T_545))

# get dust map
T_dust = T_545

# load mask
msk_fn = data_dir+"/cmb_data/HFI_Mask_Gal70.fits"
star_msk_fn = data_dir+"/cmb_data/HFI_Mask_PointSrc_all.fits"
msk = hp.read_map(msk_fn, verbose=True)
npix = len(msk)
nside = hp.npix2nside(npix)
star_msk = hp.read_map(star_msk_fn, verbose=True)
if galaxy_sample == 'WISE':
    gal_msk = hp.read_map(data_dir+"wise_data/mask_gals.fits")
elif galaxy_sample == 'DECALS':
    gal_msk = hp.read_map(data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits")
gal_msk = hp.ud_grade(gal_msk, nside)
msk *= (gal_msk*star_msk)
fsky = np.sum(msk)*1./len(msk)

# power parameters
LMIN = 0
LMAX = 3*nside+1
ell_data = np.arange(LMIN, LMAX, 1)

# load galaxy delta map
gal_delta = hp.read_map(data_dir+f"/kSZsq_gal/delta_galaxy_{galaxy_sample}.fits")
gal_masked = gal_delta*msk
T_dust_masked = T_dust*msk
T_cmb_masked = T_cmb*msk

# compute power spectrum
cl_cmb_gal = hp.anafast(T_cmb_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky
cl_dust_gal = hp.anafast(T_dust_masked, gal_masked, lmax=LMAX-1, pol=False)/fsky

# bin in ell
ell_binned = np.linspace(300, 2900, 14)
_, cl_cmb_gal_binned = bin_mat(ell_data, cl_cmb_gal, ell_binned)
ell_binned, cl_dust_gal_binned = bin_mat(ell_data, cl_dust_gal, ell_binned)

# compute alpha at each bin
alpha_binned = cl_cmb_gal_binned/(cl_dust_gal_binned-cl_cmb_gal_binned)

# interpolate and get alpha(ell)
print("alpha binned = ", alpha_binned)
alpha_binned = np.nan_to_num(alpha_binned)
print("alpha binned = ", alpha_binned)
ell_spl = np.hstack((np.linspace(ell_data[0], ell_binned[0], 10, endpoint=False), ell_binned, np.linspace(ell_binned[-1]+(ell_binned[1]-ell_binned[0]), ell_data[-1], 10)))
alpha_spl = np.hstack((np.zeros(10), alpha_binned, np.zeros(10)))
spl = CubicSpline(ell_spl, alpha_spl) # cubic
alpha_data = spl(ell_data)
np.save("alpha_binned.npy", alpha_binned)
np.save("alpha_data.npy", alpha_data)
plt.plot(ell_data, alpha_data)
plt.plot(ell_binned, alpha_binned)
plt.savefig("alpha.png")
plt.show()

# apply function to cmb and dust and get clean map
#T_cmb_alpha = hp.alm2map(hp.almxfl(hp.map2alm(T_cmb, iter=3), alpha_data), nside)
#T_dust_alpha = hp.alm2map(hp.almxfl(hp.map2alm(T_dust, iter=3), alpha_data), nside)
T_cmbmdust_alpha = hp.alm2map(hp.almxfl(hp.map2alm(T_cmb-T_dust, iter=3), alpha_data), nside)
T_clean = T_cmb + T_cmbmdust_alpha

# save map
#hp.write_map(data_dir+"/cmb_data/dust/COM_CMB_IQU-smica-nosz-elldep-nodust_2048_R3.00_full.fits", T_clean)
hp.write_map(data_dir+f"/cmb_data/dust/WPR2_CMB_elldep_nodust_K_{galaxy_sample}.fits", T_clean)

min, max = -300/1.e6, +300/1.e6
hp.mollview(T_dust, title='Tdust')
hp.mollview(T_clean, min=min, max=max, title='Tclean')
hp.mollview(T_cmb, min=min, max=max, title='TCMB')
hp.mollview(T_cmb-T_clean, title='TCMB-Tclean')
plt.show()
