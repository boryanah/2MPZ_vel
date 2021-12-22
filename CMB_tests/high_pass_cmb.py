import numpy as np
from astropy.io import fits
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
    
hdul = fits.open("../CMB_tests/cmb_data/COM_CMB_IQU-smica_2048_R3.00_full.fits")
print(hdul.info())
print(hdul[0].data)
print(hdul[1].data.dtype)
print(hdul[2].data.dtype)
#print(hdul[1].data)
print(hdul[1].data.shape)
temp = hdul[1].data['I_STOKES']

colombi1_cmap = ListedColormap(np.loadtxt("cmb_data/Planck_Parchment_RGB.txt")/255.)
colombi1_cmap.set_bad("gray") # color of missing pixels
colombi1_cmap.set_under("white") # color of background, necessary if you want to use
# this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
cmap = colombi1_cmap

nside = 2048
npix = hp.nside2npix(nside)
print(npix)
#ipix = np.arange(npix)
#x_cart, y_cart, z_cart = hp.pix2vec(nside, ipix)
prange = 0.0005 # K
# 500*10^-6 microK = 5*10^-4

# high-pass filter
ls = np.arange(3*nside)
hp_filter = np.zeros_like(ls)
ell_filter = 3000
hp_filter[ls > ell_filter] = 1
alms_temp = hp.sphtfunc.map2alm(temp, pol=False)
alms_temp_high = hp.sphtfunc.almxfl(alms_temp, hp_filter)
temp_high = hp.sphtfunc.alm2map(alms_temp_high, nside)

dpi = 300
figsize_inch = 60, 40
fig = plt.figure(figsize=figsize_inch, dpi=dpi)

hp.mollview(temp_high, xsize=figsize_inch[0]*dpi, fig=fig.number, title="", cmap=cmap, min=-prange, max=prange, nest=True)
plt.savefig("figs/cmb_high.png", dpi=dpi, bbox_inches="tight")
plt.show()
"""
(numpy.record, [('I_STOKES', '>f4'), ('Q_STOKES', '>f4'), ('U_STOKES', '>f4'), ('TMASK', '>f4'), ('PMASK', '>f4'), ('I_STOKES_INP', '>f4'), ('Q_STOKES_INP', '>f4'), ('U_STOKES_INP', '>f4'), ('TMASKINP', '>f4'), ('PMASKINP', '>f4')])

https://wiki.cosmos.esa.int/planck-legacy-archive/index.php/CMB_maps
https://pla.esac.esa.int/#maps
2018 CMB full mission SMICA Stokes I, Q and U map; I and P confidence masks; Stokes I, Q, U inpainted maps; I and P inpainting masks.
"""
