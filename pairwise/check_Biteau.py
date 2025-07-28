import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from pixell import utils
from astropy.coordinates import SkyCoord, match_coordinates_sky


data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/2mpz_data/"
hdul = fits.open(data_dir+"2MPZ_FULL_wspec_coma_complete.fits")
data = np.load(data_dir+"2MPZ_Biteau.npz")

hdl_radec = SkyCoord(ra=hdul[1].data['SUPRA'], dec=hdul[1].data['SUPDEC'], frame="icrs", unit="deg")
npz_radec = SkyCoord(ra=data['RA'], dec=data['DEC'], frame="icrs", unit="deg")

"""
# matching
idx, sep, dist = match_coordinates_sky(npz_radec, hdl_radec)
print(sep.unit, dist.unit)
np.save("idx.npy", idx)
np.save("sep.npy", sep.value)
np.save("dist.npy", dist.value)
"""


idx = np.load("idx.npy")
sep = np.load("sep.npy") # deg
dist = np.load("dist.npy")
sep_min = 0.006 # 0.008 # PSF, 0.006 # visually chosen

# get photo+zspec from 2MPZ
hdl_zphoto = hdul[1].data['ZPHOTO']
hdl_zspec = hdul[1].data['ZSPEC']
hdl_z = hdl_zphoto.copy()
hdl_z[hdl_zspec > 0.] = hdl_zspec[hdl_zspec > 0.]

"""
# TESTING final output
data = np.load(data_dir+"2MPZ_Biteau_radec.npz")

plt.scatter(hdl_z[idx[sep < sep_min]], data['Z'][sep < sep_min], s=1, marker='x', alpha=0.1, color='dodgerblue')
plt.plot([0., 1.], [0., 1.], color='black', ls='--')
plt.xlim([0, 0.3])
plt.ylim([0, 0.3])
plt.xlabel("2MPZ Z (photo+spec)")
plt.ylabel("Biteau Z")
plt.show()
"""

print(np.sum(sep > sep_min)/len(sep))

RA = data['RA']
DEC = data['DEC']
Z = data['Z'].copy()
print(len(idx), len(RA))
assert len(RA) == len(idx)

RA[sep < sep_min] = hdul[1].data['SUPRA'][idx[sep < sep_min]]
DEC[sep < sep_min] = hdul[1].data['SUPDEC'][idx[sep < sep_min]]
Z[sep < sep_min] = hdl_z[idx[sep < sep_min]]

want_cut = True
if want_cut:
    RA = RA[sep < sep_min]
    DEC = DEC[sep < sep_min]
    Z = Z[sep < sep_min]
    np.savez(data_dir+"2MPZ_Biteau_radec_cut.npz", Z=data['Z'][sep < sep_min], Z_hdul=Z, RA=RA, DEC=DEC, d_L=data['d_L'][sep < sep_min], M_star=data['M_star'][sep < sep_min], L=data['L'][sep < sep_min], B=data['B'][sep < sep_min])
else:
    np.savez(data_dir+"2MPZ_Biteau_radec.npz", Z=data['Z'], Z_hdul=Z, RA=RA, DEC=DEC, d_L=data['d_L'], M_star=data['M_star'], L=data['L'], B=data['B'])

bins = np.linspace(0., sep.max()+0.01, 1001)
binc = (bins[1:]+bins[:-1])*.5
hist, _ = np.histogram(sep, bins=bins)
plt.figure(1)
print(np.sum(hist), len(sep))
plt.plot(binc, hist/np.sum(hist))
#plt.show()

bins = np.linspace(dist.min(), dist.max(), 1001)
binc = (bins[1:]+bins[:-1])*.5
hist, _ = np.histogram(dist, bins=bins)
plt.figure(2)
plt.plot(binc, hist/np.sum(hist))
plt.show()
quit()


# not used
npz = np.vstack((data['RA'], data['DEC'])).T
#hdl = np.vstack((hdul[1].data['RA'], hdul[1].data['DEC'])).T/utils.degree
hdl = np.vstack((hdul[1].data['SUPRA'], hdul[1].data['SUPDEC'])).T

npz_z = data['Z']
hdl_zphoto = hdul[1].data['ZPHOTO']
hdl_zspec = hdul[1].data['ZSPEC']

hdl_z = hdl_zphoto.copy()
hdl_z[hdl_zspec > 0.] = hdl_zspec[hdl_zspec > 0.]

z_flag = hdl_zspec > 0.

d_L = data['d_L']

npz = npz[(d_L > 100.) & (d_L < 350.)]
npz_z = npz_z[(d_L > 100.) & (d_L < 350.)]

inds = [6487, 8120, 10376, 12430]
for i in range(npz.shape[0]):#inds:#range(npz.shape[0]):
    if i % 1000 == 0: print(i)
    match = np.all(np.isclose(npz[i], hdl, rtol=0., atol=0.005), axis=1)
    n_match = np.sum(match)
    if n_match > 0:
        print(i, n_match, npz[i], hdl[match], npz_z[i], hdl_z[match], z_flag[match])
        print("-----------------------")

#np.savez(data_dir+"2MPZ_Biteau.npz", Z=Z, RA=RA, DEC=DEC, d_L=distL, M_star=Mstar, L=GLON, B=GLAT) 
