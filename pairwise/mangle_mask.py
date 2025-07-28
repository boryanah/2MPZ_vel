import pymangle
import numpy as np
from astropy.io import fits

# load catalog
data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
gal_fn = data_dir+"/sdss_data/post_catalog.dr72bright0.fits"
hdul = fits.open(gal_fn)
RA = hdul[1].data['RA'].flatten() # 0, 360
DEC = hdul[1].data['DEC'].flatten() # -90, 90 # -10, 70
catalog_post = {}
catalog_post['RA'] = RA
catalog_post['DEC'] = DEC

# %% We now turn to the geometry and mangle.
combmask = pymangle.Mangle(data_dir+"/sdss_data/lss_combmask.dr72.ply")
window = pymangle.Mangle(data_dir+"/sdss_data/window.dr72bright0.ply")
mask = pymangle.Mangle(data_dir+"/sdss_data/mask.dr72bright0.ply")

# %% Almost all galaxies should be within the window function of the respective
# sample. A few could fall off due to round-off errors.
in_window = window.contains(catalog_post['RA'], catalog_post['DEC'])
print("Inside window: %d/%d" % (np.sum(in_window), len(in_window)))

# %% Almost no galaxy should lie inside the mask.
in_mask = mask.contains(catalog_post['RA'], catalog_post['DEC'])
print("Inside mask: %d/%d" % (np.sum(in_mask), len(in_mask)))

# %% The polygon ID provided in the catalog should also be the same then the
# one we get out of the mask file for each galaxy. Small differences are
# probably due to overlapping polygons.
polyid = combmask.polyid(catalog_post['RA'], catalog_post['DEC'])
print("Consistent poly ID: %d/%d" % (np.sum(icomb == polyid), len(icomb)))

# %% Let's make a plot of the geometry.
ra_grid = np.linspace(0, 360, 500)
dec_grid = np.linspace(-20, 90, 500)
ra_grid, dec_grid = np.meshgrid(ra_grid, dec_grid)

in_window_grid = np.reshape(window.contains(
    np.ravel(ra_grid), np.ravel(dec_grid)), ra_grid.shape)

in_mask_grid = np.reshape(mask.contains(
    np.ravel(ra_grid), np.ravel(dec_grid)), ra_grid.shape)

polyid_grid = np.reshape(combmask.polyid(np.ravel(ra_grid),
                                         np.ravel(dec_grid)),
                         ra_grid.shape)
fgotmain_grid = mask_info['FGOTMAIN'][polyid_grid]
fgotmain_grid = np.where((polyid_grid != -1) & (fgotmain_grid >= 0.8),
                         fgotmain_grid, np.nan)
fgotmain_grid = np.where(in_window_grid, fgotmain_grid, np.nan)
fgotmain_grid = np.where(np.logical_not(in_mask_grid), fgotmain_grid, np.nan)

plt.figure(figsize=(7, 4))
plt.contourf(ra_grid, dec_grid, fgotmain_grid)
sel = mask_info['FGOTMAIN'][icomb] >= 0.8
random = np.arange(np.sum(sel))
np.random.shuffle(random)
cb = plt.colorbar()
cb.set_label(r"FGOTMAIN")
plt.scatter(catalog_post['RA'][sel][random][::30],
            catalog_post['DEC'][sel][random][::30], color='black', s=1,
            edgecolors='none')
plt.xlim(0, 360)
plt.ylim(-20, 75)
plt.xlabel(r"Right Ascension $\alpha / \mathrm{deg}$")
plt.ylabel(r"Declination $\delta / \mathrm{deg}$")
plt.tight_layout(pad=0.3)
plt.savefig('geometry.pdf')
plt.close()
