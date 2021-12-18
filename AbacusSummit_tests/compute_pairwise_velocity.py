import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits

from estimator import pairwise_velocity_sky as pairwise_velocity
import plotparams
plotparams.buba()

# load mock data
hdul = fits.open("../2MPZ_tests/data/mock_1024_002_HAMonly_noMW.fits")
X = hdul[1].data['PX']
Y = hdul[1].data['PY']
Z = hdul[1].data['PZ']
VX = hdul[1].data['VX']
VY = hdul[1].data['VY']
VZ = hdul[1].data['VZ']
K_rel = hdul[1].data['K_REL']

# comoving distance to observer
P = np.vstack((X, Y, Z)).T
dist = np.sqrt(np.sum(P**2, axis=1)) # min 1 Mpc/h and max 1050 Mpc/h, comes from putting three boxes in each dimension at z = 0 next to each other

# LOS velocity, defined as hat P dot v
V_los = (X*VX + Y*VY + Z*VZ)/dist

# let's make cuts to reduce number of objects
choice = K_rel < 12.5 # original cut is 13.9, but we are currently challenged computationally
V_los = V_los[choice]
P = P[choice]
print("number of galaxies = ", np.sum(choice))

# define bins
rbins = np.linspace(0., 30., 16)
rbinc = (rbins[1:]+rbins[:-1])*.5
np.save("data/rbinc.npy", rbinc)
is_log_bin = False
periodic = False
nthread = 32 # 64 threads and 32 cores
boxsize = 0. # this is not used since we are on the sky

# change dtype
dtype = np.float32
P = P.astype(dtype)
V_los = V_los.astype(dtype)
    
# run estimator
print("starting...")
DD, PV = pairwise_velocity(P, V_los, boxsize, rbins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
print("done!")
np.save("data/DD.npy", DD)
np.save("data/PV.npy", PV)

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc, PV/100.)
plt.ylabel(r"$v_{12}(r) \ [100 {\rm km/s}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.show()


show_nz = False
if show_nz:
    # get redshift distribution (same as in paper)
    zbins = np.linspace(0, 0.4, 41)
    zbinc = (zbins[1:]+zbins[:-1])*.5
    hist_z, _ = np.histogram(hdul[1].data['Z_TRUE'], bins=zbins)

    plt.figure(figsize=(9, 7))
    plt.plot(zbinc, hist_z)
    plt.yscale('log')
    plt.show()





"""
hdul.info()
hdul[1].data
dtype=(numpy.record, [('ID', '>i8'), ('RA', '>f4'), ('DEC', '>f4'), ('Z_OBS', '>f4'), ('Z_TRUE', '>f4'), ('Z_COSMO', '>f4'), ('K_REL', '>f4'), ('K_ABS', '>f4'), ('PX', '>f4'), ('PY', '>f4'), ('PZ', '>f4'), ('VX', '>f4'), ('VY', '>f4'), ('VZ', '>f4')]))
hdul[1].data['Z_TRUE'].shape # (1285393,)
hdul[1].data['Z_TRUE'].min() # -0.0007482307
hdul[1].data['Z_TRUE'].max() # 0.38544536
hdul[1].data['K_REL'].max() # 13.999999
hdul[1].data['K_REL'].min() # 3.9206426
np.sum(hdul[1].data['K_REL'] < 13.9) # 1116309
np.sum(hdul[1].data['K_REL'] < 13.5) # 634481
hdul[1].data['K_ABS'].min() # -26.187534
hdul[1].data['K_ABS'].max() # -17.0
"""

