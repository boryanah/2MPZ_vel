import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

#galaxy_mode = "2MPZ"
galaxy_mode = "all"

cmb_mode = "DR4"
#cmb_mode = "DR5"

save_fn = f"figs/{galaxy_mode:s}_{cmb_mode:s}_pairwise.png"

if galaxy_mode == "2MPZ":
    galaxy_samples = ["2MPZ", "2MPZ", "2MPZ", "2MPZ"]
    #Thetas1 = [7.74, 7.74, 4.89, 4.89] # 2mpz
    #Thetas2 = [7.69, 7.69, 4.85, 4.85] # 2mpz
    Thetas1 = [6., 7., 8., 9.] 
    Thetas2 = [6., 7., 8., 9.] 
elif galaxy_mode == "all":
    galaxy_samples = ["2MPZ", "SDSS_L43", "SDSS_L61", "SDSS_L79"]
    #Thetas1 = [5.83, 2.1, 2.1, 2.1] # bn
    #Thetas2 = [5.83, 2.1, 2.1, 2.1] # bn
    #Thetas1 = [5.86, 2.1, 2.1, 2.1] # d56
    #Thetas2 = [5.86, 2.1, 2.1, 2.1] # d56
    #Thetas1 = [5.75, 2.1, 2.1, 2.1] # dr5
    #Thetas2 = [5.75, 2.1, 2.1, 2.1] # dr5
    Thetas1 = [6., 2.1, 2.1, 2.1] 
    Thetas2 = [6., 2.1, 2.1, 2.1]

if cmb_mode == "DR4":
    cmb_sample1 = "ACT_BN"
    #cmb_sample1 = "ACT_D56"
    cmb_sample2 = "ACT_BN"
    #cmb_sample2 = "ACT_D56"
elif cmb_mode == "DR5":
    cmb_sample1 = "ACT_DR5_f090"
    #cmb_sample1 = "ACT_DR5_f150"
    cmb_sample2 = "ACT_DR5_f090"
    #cmb_sample2 = "ACT_DR5_f150"

error_types = ["boot", "boot", "boot", "boot"]
vary_Thetas = [False, False, False, False]
#vary_Thetas = [True, True, True, True]

plot_corr = False

rbinc = np.load(f"data/rbinc.npy")

plt.subplots(2, 2, figsize=(14, 9))
plt.subplots_adjust(top=0.95, right=0.95, wspace=0.3, hspace=0.3)
for i in range(len(galaxy_samples)):
    galaxy_sample = galaxy_samples[i]
    error_type = error_types[i]
    vary_Theta = vary_Thetas[i]
    Theta1 = Thetas1[i]
    Theta2 = Thetas2[i]
    
    vary_str = "vary" if vary_Theta else "fixed"
    
    PV_2D1 = np.load(f"data/{galaxy_sample:s}_{cmb_sample1}_{vary_str:s}Th{Theta1:.2f}_PV_{error_type:s}.npy")
    PV_2D2 = np.load(f"data/{galaxy_sample:s}_{cmb_sample2}_{vary_str:s}Th{Theta2:.2f}_PV_{error_type:s}.npy")
    assert PV_2D1.shape == PV_2D2.shape # assumption is we've chosen the same number of samples for both
    if plot_corr:
        PV_corr1 = np.corrcoef(PV_2D1)
        PV_corr2 = np.corrcoef(PV_2D2)
        assert PV_corr1.shape == PV_corr2.shape == (len(rbinc), len(rbinc))
        plt.subplots()
        plt.imshow(PV_corr1-np.eye(PV_corr1.shape[0]))
        plt.colorbar()
        plt.subplots()
        plt.imshow(PV_corr2-np.eye(PV_corr2.shape[0]))
        plt.colorbar()
        plt.show()

    n_sample = PV_2D1.shape[1]
    if error_type == 'jack':
        PV_cov1 = np.cov(PV_2D1)*(n_sample-1.)
        PV_cov2 = np.cov(PV_2D2)*(n_sample-1.)
    elif error_type == 'boot':
        PV_cov1 = np.cov(PV_2D1)
        PV_cov2 = np.cov(PV_2D2)
    PV_mean1 = np.mean(PV_2D1, axis=1)
    PV_mean2 = np.mean(PV_2D2, axis=1)
    if error_type == 'jack':
        PV_err1 = np.std(PV_2D1, axis=1)*np.sqrt(n_sample-1.)
        PV_err2 = np.std(PV_2D2, axis=1)*np.sqrt(n_sample-1.)
    elif error_type == 'boot':
        PV_err1 = np.std(PV_2D1, axis=1)
        PV_err2 = np.std(PV_2D2, axis=1)
    PV_std1 = np.sqrt(np.diag(PV_cov1))
    PV_std2 = np.sqrt(np.diag(PV_cov2))
    print("std vs err 1 = ", (PV_err1-PV_std1)*100./PV_err1)
    print("std vs err 2 = ", (PV_err2-PV_std2)*100./PV_err2)
    
    inv_Err2 = 1./PV_err1**2 + 1./PV_err2**2
    PV_mean = (PV_mean1/PV_err1**2 + PV_mean2/PV_err2**2)/inv_Err2 
    PV_err = np.sqrt(1./inv_Err2)
    
    plt.subplot(2, 2, i+1)
    plt.plot(rbinc, np.zeros(len(PV_mean)), 'k--')
    plt.errorbar(rbinc, PV_mean, yerr=PV_err, capsize=4.)
    
    plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
    plt.xlabel(r"$r \ [{\rm Mpc}]$")
    plt.ylim([-0.35, 0.35])
    plt.xlim([0., 150.])
    text = " ".join(galaxy_sample.split("_"))+", ACT ("+cmb_sample1.split('_')[-1]+", "+cmb_sample2.split('_')[-1]+rf"), $\Theta$={Theta1:.1f}'"
    plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)

plt.savefig(save_fn)
plt.show()
