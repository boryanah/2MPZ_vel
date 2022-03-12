import glob
import sys

import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

root = f"{sys.argv[1]:s}"
fns = glob.glob(f"data/{root:s}.npy")
r_max = 60.
r_min = 0.

for fn in fns:
    
    PV_2D = np.load(fn)
    rbinc = np.load(f"data/rbinc.npy")
    root = (fn.split('data/')[-1]).split('.npy')[0]
    
    if "boot" in fn:
        error_type = "boot"
        PV = np.load('data/'+root.split('_boot')[0]+'.npy')
    elif "jack" in fn:
        error_type = "jack"
        PV = np.load('data/'+root.split('_jack')[0]+'.npy')
    else:
        error_type = ""
        PV = np.load('data/'+root+'.npy')

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

    choice = (rbinc < r_max) & (rbinc > r_min)
    PV_cov_red = np.cov(PV_2D[choice, :])
    if error_type == 'jack':
        PV_cov_red *= (n_sample-1.)
    print("chi2 = ", np.dot(PV_mean[choice], np.dot(np.linalg.inv(PV_cov_red), PV_mean[choice])))
    print("chi2 diag = ", np.sum(PV_mean[choice]**2./np.diag(PV_cov_red)))
    print("dof = ", np.sum(choice))
    print("cond number = ", np.linalg.cond(PV_cov_red))
    
    plt.figure(figsize=(9, 7))
    plt.plot(rbinc, np.zeros(len(PV_mean)), 'k--')
    plt.errorbar(rbinc, PV, yerr=PV_err, capsize=4.)
    plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
    plt.xlabel(r"$r \ [{\rm Mpc}]$")
    #plt.ylim([-0.35, 0.35])
    plt.ylim([-0.1, 0.1])
    plt.xlim([0., 150.])
    text = " ".join(root.split("_"))
    plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)
    plt.savefig(f"figs/{root:s}.png")
plt.show()
