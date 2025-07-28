import glob
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import plotparams
plotparams.buba()

def get_stats_mini(PV, PV_2D):
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


def get_stats(PV_2D, rbinc, r_min, r_max, err_type):
    n_sample = PV_2D.shape[1]
    choice = (rbinc < r_max) & (rbinc > r_min)
    PV_cov_red = np.cov(PV_2D[choice, :])

    PV_mean = np.mean(PV_2D, axis=1)
    PV = PV_mean # not strictly true but lazy

    if err_type == 'jack' or err_type == 'kmeans':
        PV_cov_red *= (n_sample-1.)
    chi2 = np.dot(PV[choice], np.dot(np.linalg.inv(PV_cov_red), PV[choice]))
    print("chi2 = ", chi2)
    print("chi2 diag = ", np.sum(PV[choice]**2./np.diag(PV_cov_red))) 
    print("dof = ", np.sum(choice))
    print("cond number = ", np.linalg.cond(PV_cov_red))
    return chi2

def get_chi2(pars):#, r_min, r_max, err_type):
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
            
    chi2 = -get_stats_mini(PV_all, PV_2D_all)#, r_min, r_max, err_type)
    #print("bias 0, 1, 2 = ", bias)
    #print("taus 0, 1, 2 = ", taus)
    #print("chi2 = ", chi2)
    #print("----------------------------")
    return chi2


names = ['L43D', 'L61D', 'L79']
err_type = "kmeans"
Th = 2.1
cmb_sample = "ACT_DR5_f150"
r_max = 150.
r_min = 0.
rbinc = np.load(f"data/rbinc.npy")

want_mini = True
if want_mini:
    method = 'powell'
    #method = 'Nelder-Mead'
    p0 = np.array([0.33, 0.33, 0.33, 0.33]) 

    res = minimize(get_chi2, p0, method=method)#, args=(r_min, r_max, err_type))
    bias0, bias1, taus0, taus1 = res['x']
    print("bias 0, 1, 2 = ", bias0, bias1, 1.-bias0-bias1)
    print("taus 0, 1, 2 = ", taus0, taus1, 1.-taus0-taus1)
    print("minimized chi2 = ", -get_chi2(res['x']))
    quit()

bias = np.array([1.3870224169854222, 1.5896371204371453, 1.9908256254306487]) # mean
#bias = np.array([1.384701451372272, 1.5810544292363633, 2.1450128725568067]) # median
taus = np.array([0.59, 1.10, 1.92])*1.e-4 # ACT DR5 f150
Ns = np.array([130577, 109911, 103159])
#bias = np.ones_like(bias)
#taus = np.ones_like(taus)

print("input taus = ", taus*Ns/np.sum(taus*Ns))
print("input bias = ", bias/np.sum(bias))

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

        PV_2D_all += taus[i]*Ns[i]*bias[j]*PV_2D
        print("_________________________________________")

get_stats(PV_2D_all, rbinc, r_min, r_max, err_type)
