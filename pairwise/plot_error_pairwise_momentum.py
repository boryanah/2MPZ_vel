import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

#galaxy_sample = "2MPZ_ACT_BN" # not really 
#galaxy_sample = "2MPZ_ACT_D56" # yes signal
#galaxy_sample = "BOSS_South" # yes signal
#galaxy_sample = "BOSS_North_ACT_BN" # not really
galaxy_sample = "2MPZ"
cmb_sample = "ACT_D56"

#h = 1.
h = 0.7
rbinc = np.load(f"data/{galaxy_sample:s}_{cmb_sample:s}_rbinc.npy")

# number of jackknife samples
n_jack = 100
pre_saved = True

if pre_saved:
    PV_mean = np.load(f"data/{galaxy_sample:s}_{cmb_sample}_PV_mean.npy")
    PV_err = np.load(f"data/{galaxy_sample:s}_{cmb_sample}_PV_err.npy")
else:
    # initialize
    PV_2D = np.zeros((len(rbinc), n_jack))
    DD_2D = np.zeros((len(rbinc), n_jack))
    for i_rank in range(n_jack):
        # save arrays
        DD = np.load(f"data_jack/{galaxy_sample:s}_{cmb_sample}_DD_rank{i_rank:d}.npy")
        PV = np.load(f"data_jack/{galaxy_sample:s}_{cmb_sample}_PV_rank{i_rank:d}.npy")
        PV_2D[:, i_rank] = PV
        DD_2D[:, i_rank] = DD

    # compute errorbars
    DD_mean = np.mean(DD_2D, axis=1)
    PV_mean = np.mean(PV_2D, axis=1)    
    DD_err = np.sqrt(n_jack-1)*np.std(DD_2D, axis=1)
    PV_err = np.sqrt(n_jack-1)*np.std(PV_2D, axis=1)

# plot pairwise velocity
plt.figure(figsize=(9, 7))
plt.plot(rbinc/h, np.zeros(len(PV_mean)), 'k--')
plt.errorbar(rbinc/h, PV_mean, yerr=PV_err, capsize=4.
)
plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
if h < 1.:
    plt.xlabel(r"$r \ [{\rm Mpc}]$")
else:
    plt.xlabel(r"$r \ [{\rm Mpc}/h]$")
plt.ylim([-0.4, 0.4])
plt.xlim([0., 100.])
plt.savefig(f"figs/{galaxy_sample:s}_{cmb_sample:s}_pairwise.png")
plt.show()
