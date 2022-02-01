import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

galaxy_sample = "2mpz"
#galaxy_sample = "mock_2015"
#galaxy_sample = "mock_box"
#galaxy_sample = "mock_rsd_box"
#galaxy_sample = "random"

h = 1.
#h = 0.7
rbinc = np.load("data_pairwise/rbinc.npy")
DD = np.load(f"data_pairwise/{galaxy_sample:s}_DD.npy")
PV = np.load(f"data_pairwise/{galaxy_sample:s}_PV.npy")
print(DD, rbinc)

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc/h, PV/100.)
#plt.plot(rbinc, DD)
plt.ylabel(r"$v_{12}(r) \ [100 \ {\rm km/s}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.ylim([-2.5, 0.])
plt.xlim([0., 30.])
plt.show()
