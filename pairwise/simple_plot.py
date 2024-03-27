import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.optimize import brentq

import plotparams
plotparams.buba()

c = 3.e5 # km/s
T_CMB = 2.73; T_unit = 1.e6
#T_CMB = 2.73e-6; T_unit = 1.

# make cuts
r_max = 150. # 40., 60.
r_min = 20. # 0., 10., 20.

def cov2corr(A):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A

def get_model_fn():
    """
    data = np.loadtxt("../stats/data/L43_S18_ksz_vij_iz1.dat")
    r = data[:, 2]
    V = data[:, 3]
    """
    data = np.load("../desi/data/V_pairwise_z0.32_HOD.npz")
    r = data['rsep']
    V = data['V']
    V_f = interp1d(r, V)
    fn = lambda A, r: -T_CMB/c*A*V_f(r)
    return fn

def get_chi2(A, r, D, iC):
    d = (p_fn(A, r) - D)
    chi2 = np.dot(np.dot(d, iC), d)
    return chi2

# get theory function
p_fn = get_model_fn()

root = f"{sys.argv[1]:s}"
if "-1" == root:
    files = glob.glob("data/*PV_boot.npy")
    #files = glob.glob("data/*PV_kmeans.npy")
    files.sort(key=os.path.getmtime)
    root = files[-1].split('data/')[-1]
if '.npy' in root:
    root = root.split('.npy')[0]
fns = sorted(glob.glob(f"data/{root:s}.npy"))
if len(fns) == 0:
    fns = [f"keep/{root:s}.npy"]
    loc = "keep/"
else:
    loc = "data/"
    
for fn in fns:
    PV_2D = np.load(fn)
    rbinc = np.load(f"data/rbinc.npy")
    root = (fn.split(f"{loc}")[-1]).split('.npy')[0]
    print("root = ", fn)

    # TESTING
    """
    if "boot" in fn:
        error_type = "boot"
        PV = np.load(f"{loc}/"+root.split('_boot')[0]+'.npy')
    elif "jack" in fn:
        error_type = "jack"
        PV = np.load(f"{loc}/"+root.split('_jack')[0]+'.npy')
    elif "kmeans" in fn:
        error_type = "kmeans"
        PV = np.load(f"{loc}/"+root.split('_kmeans')[0]+'.npy')
    else:
        error_type = ""
        PV = np.load(f"{loc}/"+root+'.npy')
    """
    error_type = "boot"

    if error_type == 'jack':
        n_sample = PV_2D.shape[1]
        print("n_sample = ", n_sample)
        PV_cov = np.cov(PV_2D)*(n_sample-1.)
        PV_err = np.std(PV_2D, axis=1)*np.sqrt(n_sample-1.)
        PV_mean = np.mean(PV_2D, axis=1)
    if error_type == 'kmeans':
        n_sample = PV_2D.shape[1]
        print("n_sample = ", n_sample)
        PV_cov = np.cov(PV_2D)*(n_sample-1.)
        PV_err = np.std(PV_2D, axis=1)*np.sqrt(n_sample-1.)
        PV_mean = np.mean(PV_2D, axis=1)
    elif error_type == 'boot':
        n_sample = PV_2D.shape[1]
        print("n_sample = ", n_sample)
        PV_cov = np.cov(PV_2D)
        PV_err = np.std(PV_2D, axis=1)
        PV_mean = np.mean(PV_2D, axis=1)
    else:
        PV_mean = PV_2D
        PV_err = np.zeros_like(rbinc)
    # lazy butt will need to rerun
    PV = PV_mean # TESTING
        
    if error_type not in ['boot', 'jack', 'kmeans']: continue
    choice = (rbinc < r_max) & (rbinc > r_min)
    PV_cov_reduced = PV_cov[choice[:, None] & choice[None, :]].reshape(np.sum(choice), np.sum(choice))
    iC_reduced = np.linalg.inv(PV_cov_reduced)
    if error_type == 'jack' or error_type == 'kmeans':
        PV_cov_reduced *= (n_sample-1.)
    chi2_null = np.dot(PV[choice], np.dot(iC_reduced, PV[choice]))
    dof = np.sum(choice)
    A = 1.
    res = minimize(get_chi2, A, args=(rbinc[choice], PV[choice], iC_reduced), method="CG")
    chi2_min = res['fun']
    A = res['x'][0]
    print("chi2 null = ", chi2_null)
    print("chi2 min = ", chi2_min)
    print("chi2 null (diag) = ", np.sum(PV[choice]**2./np.diag(PV_cov_reduced))) 
    #print("chi2 diag = ", np.sum((PV[choice]/PV_err_red)**2)) # almost same as above
    print("dof = ", dof)
    print("PTE", 1 - stats.chi2.cdf(chi2_null, dof))
    print("SNR", np.sqrt(chi2_null-chi2_min))
    print("cond number = ", np.linalg.cond(PV_cov_reduced))
    print("PV = ", PV)
    
    # compute chi2
    As = np.linspace((1.-0.5)*A, (1+0.5)*A, 100)
    chi2s = np.zeros_like(As)
    for i in range(len(As)):
        chi2s[i] = get_chi2(As[i], rbinc[choice], PV[choice], iC_reduced)
    plt.figure(figsize=(9, 7))
    plt.plot(As, chi2s)
    f = lambda a: get_chi2(a, rbinc[choice], PV[choice], iC_reduced) - chi2_min - 1.
    A_lo = brentq(f, (1.-0.8)*A, A)
    A_hi = brentq(f, A, (1.+0.8)*A)
    print(f"tau = {(A/T_unit):.2e} pm {(A_hi-A_lo)*0.5/T_unit:.2e}")
    print("fraction constrained", (A-A_lo)/A, (A_hi-A)/A)
    plt.gca().axhline(chi2_min+1., color='k', ls='--')
    plt.gca().axhline(chi2_min+4., color='k', ls='--')
    plt.gca().axvline(A_hi, color='k', ls='--')
    plt.gca().axvline(A_lo, color='k', ls='--')
    #plt.close()
    
    plt.figure(figsize=(10, 7))
    PV_corr = cov2corr(PV_cov)
    plt.imshow(PV_corr)
    #plt.savefig(f"figs/{root:s}_corr.png")
    #plt.close()
    
    plt.figure(figsize=(10, 7))
    plt.plot(rbinc, np.zeros(len(PV)), 'k--')
    #plt.errorbar(rbinc, PV, yerr=PV_err, capsize=4.)
    plt.errorbar(rbinc, PV, yerr=PV_err, marker='o', ls='', capsize=4.)
    #plt.plot(rbinc, PV_mean, 'k:') # TESTING
    plt.plot(rbinc, p_fn(res['x'], rbinc), 'k-')
    plt.ylabel(r"$\hat p_{\rm kSZ}(r) \ [\mu {\rm K}]$")
    plt.xlabel(r"$r \ [{\rm Mpc}]$")
    #plt.ylim([-0.35, 0.35])
    #plt.ylim([-0.1, 0.1])
    plt.xlim([0., 150.])
    text = " ".join(root.split("_"))
    #plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)
    plt.savefig(f"figs/{root:s}.png")
    print("_________________________________________________")
plt.show()
plt.close()
    
