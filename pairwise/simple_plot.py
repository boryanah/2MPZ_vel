import sys

import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

root = f"{sys.argv[1]:s}"
fn = f"data/{root:s}.npy"
PV_2D = np.load(fn)
rbinc = np.load(f"data/rbinc.npy")

if "boot" in fn:
    error_type = "boot"
elif "jack" in fn:
    error_type = "jack"
else:
    error_type = ""

if error_type == 'jack':
    n_sample = PV_2D.shape[1]
    PV_cov = np.cov(PV_2D)*(n_sample-1.)
    PV_err = np.std(PV_2D, axis=1)*np.sqrt(n_sample-1.)
    PV_mean = np.mean(PV_2D, axis=1)
elif error_type == 'boot':
    n_sample = PV_2D.shape[1]
    PV_cov = np.cov(PV_2D)
    PV_err = np.std(PV_2D, axis=1)
    PV_mean = np.mean(PV_2D, axis=1)
else:
    PV_mean = PV_2D
    PV_err = np.zeros_like(rbinc)

plt.figure(figsize=(9, 7))
plt.plot(rbinc, np.zeros(len(PV_mean)), 'k--')
plt.errorbar(rbinc, PV_mean, yerr=PV_err, capsize=4.)
plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
plt.xlabel(r"$r \ [{\rm Mpc}]$")
plt.ylim([-0.35, 0.35])
plt.xlim([0., 150.])
text = " ".join(root.split("_"))
plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)
plt.show()
