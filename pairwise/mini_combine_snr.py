import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import plotparams
plotparams.buba()

def get_stats(PV, PV_2D):
    n_sample = PV_2D.shape[1]
    choice = (rbinc < r_max) & (rbinc > r_min)
    PV_cov_red = np.cov(PV_2D[choice, :])
    #PV_mean = np.mean(PV_2D, axis=1)
    
    if err_type == 'jack' or err_type == 'kmeans':
        PV_cov_red *= (n_sample-1.)
        
    chi2 = np.dot(PV[choice], np.dot(np.linalg.inv(PV_cov_red), PV[choice]))
    #chi2_diag = np.sum(PV[choice]**2./np.diag(PV_cov_red)) 
    #dof = np.sum(choice)
    #cond number = np.linalg.cond(PV_cov_red)
    return chi2

def get_chi2(pars):
    bias = np.array([pars[0], pars[1], 1. - pars[0] - pars[1]])
    taus = np.array([pars[2], pars[3], 1. - pars[2] - pars[3]])
    if np.any(bias) < 0.: return np.inf
    if np.any(taus) < 0.: return np.inf

    PV_all = np.zeros(len(rbinc))
    PV_2D_all = np.zeros((len(rbinc), 1000))
    for i in range(len(names)):
        for j in range(len(names)):
            fn = f"data/SDSS_{names[i]}_{names[j]}_premask_{cmb_sample}_fixedTh{Th:.2f}_cross_PV_{err_type}.npy"
            PV = np.load(''.join(fn.split(f'_{err_type}')))
            PV_2D = np.load(fn)
            PV_all += taus[i]*bias[j]*PV
            PV_2D_all += taus[i]*bias[j]*PV_2D
            
    chi2 = -get_stats(PV_all, PV_2D_all)
    print("bias 0, 1, 2 = ", bias)
    print("taus 0, 1, 2 = ", taus)
    print("chi2 = ", chi2)
    print("----------------------------")
    return chi2


names = ['L43D', 'L61D', 'L79']
err_type = "kmeans"
Th = 2.1
cmb_sample = "ACT_DR5_f150"
r_max = 150.
r_min = 0.
rbinc = np.load(f"data/rbinc.npy")

method = 'powell'
#method = 'Nelder-Mead'
p0 = np.array([0.33, 0.33, 0.33, 0.33]) 

res = minimize(get_chi2, p0, method=method)#, args=())
bias0, bias1, taus0, taus1 = res['x']
print("bias 0, 1, 2 = ", bias0, bias1, 1.-bias0-bias1)
print("taus 0, 1, 2 = ", taus0, taus1, 1.-taus0-taus1)

quit()
bias = np.array([1.3870224169854222, 1.5896371204371453, 1.9908256254306487]) # mean
#bias = np.array([1.384701451372272, 1.5810544292363633, 2.1450128725568067]) # median
taus = np.array([0.59, 1.10, 1.92])*1.e-4 # ACT DR5 f150

def get_stats(PV_2D, rbinc, r_min, r_max, err_type):
    n_sample = PV_2D.shape[1]
    choice = (rbinc < r_max) & (rbinc > r_min)
    PV_cov_red = np.cov(PV_2D[choice, :])

    PV_mean = np.mean(PV_2D, axis=1)
    PV = PV_mean # not strictly true but lazy

    if err_type == 'jack' or err_type == 'kmeans':
        PV_cov_red *= (n_sample-1.)
    print("chi2 = ", np.dot(PV[choice], np.dot(np.linalg.inv(PV_cov_red), PV[choice])))
    print("chi2 diag = ", np.sum(PV[choice]**2./np.diag(PV_cov_red))) 
    print("dof = ", np.sum(choice))
    print("cond number = ", np.linalg.cond(PV_cov_red))

# minimizer/maximizer of SNR for tau/bias
#bias = 0 to 1
#taus = 0 to 1
PV_2D_all = np.zeros((len(rbinc), 1000))
for i in range(len(names)):
    for j in range(len(names)):
        # 43D (massive), 61D (rec) - 61D, 43D
        fn = f"data/SDSS_{names[i]}_{names[j]}_premask_{cmb_sample}_fixedTh{Th:.2f}_cross_PV_{err_type}.npy"
        #root = (fn.split('data/')[-1]).split('.npy')[0]
        #PV = np.load('data/'+root.split('_boot')[0]+'.npy')
        
        PV_2D = np.load(fn)
        get_stats(PV_2D, rbinc, r_min, r_max, err_type)

        if i == j:
            PV_2D_all += taus[i]*bias[j]*PV_2D
        else:
            PV_2D_all += taus[i]*bias[j]*PV_2D
        print("_________________________________________")

get_stats(PV_2D_all, rbinc, r_min, r_max, err_type)

quit()

root = f"{sys.argv[1]:s}"
if "-1" == root:
    #files = glob.glob("data/*PV_boot.npy")
    files = glob.glob("data/*PV_kmeans.npy")
    files.sort(key=os.path.getmtime)
    root = files[-1].split('data/')[-1]
if '.npy' in root:
    root = root.split('.npy')[0]
fns = sorted(glob.glob(f"data/{root:s}.npy"))
if ("MGS" in root) or ("2MPZ" in root):
    r_max = 60. #60.
    r_min = 20.#10.
else:
    r_max = 150.
    r_min = 0.

for fn in fns:
    PV_2D = np.load(fn)
    rbinc = np.load(f"data/rbinc.npy")
    root = (fn.split('data/')[-1]).split('.npy')[0]
    print("root = ", fn)
    
    if "boot" in fn:
        error_type = "boot"
        PV = np.load('data/'+root.split('_boot')[0]+'.npy')
    elif "jack" in fn:
        error_type = "jack"
        PV = np.load('data/'+root.split('_jack')[0]+'.npy')
    elif "kmeans" in fn:
        error_type = "kmeans"
        PV = np.load('data/'+root.split('_kmeans')[0]+'.npy')
    else:
        error_type = ""
        PV = np.load('data/'+root+'.npy')

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

    if error_type in ['boot', 'jack', 'kmeans']:
        choice = (rbinc < r_max) & (rbinc > r_min)
        PV_cov_red = np.cov(PV_2D[choice, :])
        #PV_err_red = np.std(PV_2D[choice, :])
        #PV_err_red = PV_err[choice] # almost same as above
        if error_type == 'jack' or error_type == 'kmeans':
            PV_cov_red *= (n_sample-1.)
        print("chi2 = ", np.dot(PV[choice], np.dot(np.linalg.inv(PV_cov_red), PV[choice])))
        print("chi2 diag = ", np.sum(PV[choice]**2./np.diag(PV_cov_red))) 
        #print("chi2 diag = ", np.sum((PV[choice]/PV_err_red)**2)) # almost same as above
        print("dof = ", np.sum(choice))
        print("cond number = ", np.linalg.cond(PV_cov_red))
    print("PV = ", PV)
        
    plt.figure(figsize=(10, 7))
    plt.plot(rbinc, np.zeros(len(PV)), 'k--')
    plt.errorbar(rbinc, PV, yerr=PV_err, capsize=4.)
    plt.plot(rbinc, PV_mean, 'k:')
    plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
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
    
