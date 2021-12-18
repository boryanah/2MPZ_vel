import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

h = 0.7
rbinc = np.load("data/rbinc.npy")
DD = np.load("data/DD.npy")
PV = np.load("data/PV.npy")

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc, PV/100.)
plt.ylabel(r"$v_{12}(r) \ [100 \ {\rm km/s}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.ylim([-2.5, 0.])
plt.xlim([0., 30.])
plt.show()
