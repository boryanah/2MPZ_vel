import numpy as np
import matplotlib.pyplot as plt

from pixell import powspec

import plotparams
plotparams.buba()


cmb_samples = ["ACT_BN", "ACT_D56", "ACT_DR5_f090_45uK", "ACT_DR5_f090_65uK", "ACT_DR5_f150_45uK", "ACT_DR5_f150_65uK"]

camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
cltt = camb_theory[0, 0, :3000]
ls = np.arange(cltt.size)

plt.figure(figsize=(12, 8))
plt.plot(ls, cltt*ls*(ls+1)/(np.pi*2.), lw=3, color='k')

for cmb_sample in cmb_samples:
    binned_power = np.load(f"camb_data/{cmb_sample}_binned_power.npy")
    centers = np.load(f"camb_data/{cmb_sample}_centers.npy")

    plt.plot(centers, binned_power*centers*(centers+1.)/(np.pi*2.), label=" ".join(cmb_sample.split('_')), marker="o", ls="none")
    
plt.yscale('log')
plt.xlabel('$\\ell$')
plt.ylabel('$D_{\\ell}$')
plt.legend(ncol=2)
plt.savefig("figs/power_spectra.png")
plt.show()
