import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits, ascii
from pixell import utils
from astropy.coordinates import SkyCoord, match_coordinates_sky


save_dir = "/mnt/marvin1/boryanah/2MPZ_vel/2mpz_data/"
#['Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'flag_asso', 'distL', 'r_distL', 'Mstar', 'r_Mstar', 'cNm', 'SFR', 'r_SFR', 'cNs', 'PGC', 'AllWISE', 'IDX']
key_import = ['Name', 'AllWISE', 'RAJ2000', 'DEJ2000','distL', 'Mstar', 'GLON', 'GLAT']
table = ascii.read(save_dir+"table5.mrt", include_names = key_import)
choice = table['Name'] == 'clone'
table = table[~choice]
print("dulicates = ", np.sum(choice)*100./len(choice))

RA = np.array(table['RAJ2000'], dtype=float)
DEC = np.array(table['DEJ2000'], dtype=float)
distL = np.array(table['distL'], dtype=float)
Mstar = np.array(table['Mstar'], dtype=float)
GLON = np.array(table['GLON'], dtype=float)
GLAT = np.array(table['GLAT'], dtype=float)
AllWISE = table['AllWISE']
Z = np.zeros(len(RA), dtype=float)
AllWISE[:10]

data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/2mpz_data/"
hdul = fits.open(data_dir+"2MPZ_FULL_wspec_coma_complete.fits")

quit()

data = np.load(data_dir+"2MPZ_Biteau.npz")


# match Biteau's "AllWISE" to WISE's "designation" and then matching WISE's "CNTR" to 2MPZ's "WISEID"


quit()
"""
hdl_radec = SkyCoord(ra=hdul[1].data['SUPRA'], dec=hdul[1].data['SUPDEC'], frame="icrs", unit="deg")
npz_radec = SkyCoord(ra=data['RA'], dec=data['DEC'], frame="icrs", unit="deg")

idx, sep, dist = match_coordinates_sky(npz_radec, hdl_radec)
print(sep.unit, dist.unit)
np.save("idx.npy", idx)
np.save("sep.npy", sep.value)
np.save("dist.npy", dist.value)
"""

idx = np.load("idx.npy")
sep = np.load("sep.npy") # deg
dist = np.load("dist.npy")
sep_min = 0.006 # visually chosen

# get photo+zspec from 2MPZ
hdl_zphoto = hdul[1].data['ZPHOTO']
hdl_zspec = hdul[1].data['ZSPEC']
hdl_z = hdl_zphoto.copy()
hdl_z[hdl_zspec > 0.] = hdl_zspec[hdl_zspec > 0.]


# TESTING final output
data = np.load(data_dir+"2MPZ_Biteau_radec.npz")

plt.scatter(hdl_z[idx[sep < sep_min]], data['Z'][sep < sep_min], s=1, marker='x', alpha=0.1, color='dodgerblue')
plt.plot([0., 1.], [0., 1.], color='black', ls='--')
plt.xlim([0, 0.3])
plt.ylim([0, 0.3])
plt.xlabel("2MPZ Z (photo+spec)")
plt.ylabel("Biteau Z")
plt.show()
quit()


print(np.sum(sep > sep_min)/len(sep))

RA = data['RA']
DEC = data['DEC']
Z = data['Z'].copy()
print(len(idx), len(RA))
assert len(RA) == len(idx)

RA[sep < sep_min] = hdul[1].data['SUPRA'][idx[sep < sep_min]]
DEC[sep < sep_min] = hdul[1].data['SUPDEC'][idx[sep < sep_min]]
Z[sep < sep_min] = hdl_z[idx[sep < sep_min]]

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
