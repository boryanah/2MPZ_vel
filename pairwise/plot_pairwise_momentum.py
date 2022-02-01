import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

#galaxy_samples = ["2MPZ_ACT_BN", "2MPZ_ACT_D56", "BOSS_North_ACT_BN", "BOSS_South_ACT_D56"]
galaxy_samples = ["SDSS_L61_ACT_BN", "SDSS_L61_ACT_D56", "BOSS_North_ACT_BN", "BOSS_South_ACT_D56"]

#h = 1.
h = 0.7


plt.subplots(2, 2, figsize=(14, 9))
plt.subplots_adjust(top=0.95, right=0.95, wspace=0.3, hspace=0.3)
for i in range(len(galaxy_samples)):
    galaxy_sample = galaxy_samples[i]
    DD = np.load(f"data/{galaxy_sample:s}_DD.npy")
    PV = np.load(f"data/{galaxy_sample:s}_PV.npy")
    rbinc = np.load(f"data/{galaxy_sample:s}_rbinc.npy")
    
    plt.subplot(2, 2, i+1)
    plt.plot(rbinc/h, np.zeros(len(PV)), 'k--')

    plt.plot(rbinc/h, PV)
    plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
    if h < 1.:
        plt.xlabel(r"$r \ [{\rm Mpc}]$")
    else:
        plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
    plt.ylim([-0.4, 0.4])
    plt.xlim([0., 100.])
    plt.text(x=0.2, y=0.1, s=" ".join(galaxy_sample.split("_")), transform=plt.gca().transAxes)
plt.savefig("figs/all_pairwise.png")
plt.show()
quit()
# plot pairwise velocity
plt.figure(figsize=(9, 7))

plt.plot(rbinc/h, PV)
#plt.plot(rbinc/h, DD)
plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.ylim([-0.4, 0.4])
#plt.xlim([0., 30.])
plt.xlim([0., 100.])
plt.savefig(f"figs/{galaxy_sample:s}_pairwise.png")
plt.show()
