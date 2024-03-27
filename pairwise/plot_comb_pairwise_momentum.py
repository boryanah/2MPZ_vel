import sys

import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

#galaxy_mode = "2MPZ"
#galaxy_mode = "2MPZ_rand"
#galaxy_mode = "2MPZ_premask"
galaxy_mode = "all"
r_max = 100. #60. # 60.
r_min = 0. # 20

if galaxy_mode == "2MPZ_premask":
    Th = float(sys.argv[1])

cmb_mode = "DR4"
#cmb_mode = "DR5"
#cmb_mode = "all"

if galaxy_mode == "2MPZ":
    galaxy_samples = ["2MPZ", "2MPZ", "2MPZ", "2MPZ"]
    Thetas = [6., 7., 8., 9.] 
    mask_str = " mask0"
elif galaxy_mode == "2MPZ_rand": # 100 looks good
    galaxy_samples = ["2MPZ_rand1", "2MPZ_rand201", "2MPZ_rand202", "2MPZ_rand300"]#203, 300
    Thetas = [6., 6., 6., 6.] 
    mask_str = " mask0"
elif galaxy_mode == "2MPZ_premask": # 100 looks good
    galaxy_samples = ["2MPZ_premask", "2MPZ_premask", "2MPZ_premask", "2MPZ_premask"]
    #galaxy_samples = ["2MPZ_mtype2_premask", "2MPZ_mtype2_premask", "2MPZ_mtype2_premask", "2MPZ_mtype2_premask"]
    #galaxy_samples = ["2MPZ_mtype5_premask", "2MPZ_mtype5_premask", "2MPZ_mtype5_premask", "2MPZ_mtype5_premask"]
    if galaxy_samples[0] == "2MPZ_premask":
        mask_str = " mask0"
    else:
        mask_str = f" mask{galaxy_samples[0].split('_')[1].split('mtype')[1]}"
    #galaxy_samples = ["2MPZ_premask_rand1", "2MPZ_premask", "2MPZ_premask", "2MPZ_premask"] # if randoms
    Thetas = [Th, Th, Th, Th]

elif galaxy_mode == "all":
    #galaxy_samples = ["2MPZ", "SDSS_L43", "SDSS_L61", "SDSS_L79"]
    galaxy_samples = ["SDSS_L43D_premask", "SDSS_L43_premask", "SDSS_L61_premask", "SDSS_L79_premask"]
    Thetas = [2.1, 2.1, 2.1, 2.1] 
    mask_str = " mask0"
    
if cmb_mode == "DR4":
    cmb_sample1 = "ACT_BN"
    #cmb_sample1 = "ACT_D56"
    #cmb_sample2 = "ACT_BN"
    cmb_sample2 = "ACT_D56"
elif cmb_mode == "DR5":
    #cmb_sample1 = "ACT_DR5_f090"
    cmb_sample1 = "ACT_DR5_f150"
    #cmb_sample2 = "ACT_DR5_f090"
    cmb_sample2 = "ACT_DR5_f150"
elif cmb_mode == "all":
    cmb_samples1 = ["ACT_D56", "ACT_BN", "ACT_DR5_f090", "ACT_DR5_f150"]
    cmb_samples2 = ["ACT_D56", "ACT_BN", "ACT_DR5_f090", "ACT_DR5_f150"]
    cmb_samples1 = ["Planck_healpix", "Planck_healpix", "Planck_healpix", "Planck_healpix"]
    cmb_samples2 = ["Planck_healpix", "Planck_healpix", "Planck_healpix", "Planck_healpix"]
    
error_types = ["boot", "boot", "boot", "boot"]
#error_types = ["jack", "jack", "jack", "jack"]
vary_Thetas = [False, False, False, False]
#vary_Thetas = [True, True, True, True]

if galaxy_mode == "2MPZ_premask":
    save_fn = f"figs/{galaxy_samples[0]:s}_{cmb_mode:s}_Th{Thetas[0]:.2f}_pairwise.png"
else:
    save_fn = f"figs/{galaxy_mode:s}_{cmb_mode:s}_pairwise.png"

plot_corr = False

rbinc = np.load(f"data/rbinc.npy")

plt.subplots(2, 2, figsize=(14, 9))
plt.subplots_adjust(top=0.95, right=0.95, wspace=0.3, hspace=0.3)
for i in range(len(galaxy_samples)):
    galaxy_sample = galaxy_samples[i]
    error_type = error_types[i]
    vary_Theta = vary_Thetas[i]
    Theta = Thetas[i]
    
    if cmb_mode == "all":
        cmb_sample1 = cmb_samples1[i]
        cmb_sample2 = cmb_samples2[i]

        
    vary_str = "vary" if vary_Theta else "fixed"

    if cmb_sample1 == cmb_sample2:
        same = True
    else:
        same = False
    
    PV_2D1 = np.load(f"data/{galaxy_sample:s}_{cmb_sample1}_{vary_str:s}Th{Theta:.2f}_PV_{error_type:s}.npy")
    print(f"data/{galaxy_sample:s}_{cmb_sample1}_{vary_str:s}Th{Theta:.2f}_PV_{error_type:s}.npy")
    print(f"data/{galaxy_sample:s}_{cmb_sample2}_{vary_str:s}Th{Theta:.2f}_PV_{error_type:s}.npy")
    PV_2D2 = np.load(f"data/{galaxy_sample:s}_{cmb_sample2}_{vary_str:s}Th{Theta:.2f}_PV_{error_type:s}.npy")
    PV_mean1 = np.load(f"data/{galaxy_sample:s}_{cmb_sample1}_{vary_str:s}Th{Theta:.2f}_PV.npy")
    PV_mean2 = np.load(f"data/{galaxy_sample:s}_{cmb_sample2}_{vary_str:s}Th{Theta:.2f}_PV.npy")
    assert PV_2D1.shape == PV_2D2.shape # assumption is we've chosen the same number of samples for both
    try:
        PV_rand1 = np.load(f"data/{galaxy_sample:s}_rand10_{cmb_sample1}_{vary_str:s}Th{Theta:.2f}_PV.npy")
        PV_rand2 = np.load(f"data/{galaxy_sample:s}_rand10_{cmb_sample2}_{vary_str:s}Th{Theta:.2f}_PV.npy")
    except:
        PV_rand1 = np.zeros_like(PV_mean1)
        PV_rand2 = np.zeros_like(PV_mean2)

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
    #PV_mean1 = np.mean(PV_2D1, axis=1)
    #PV_mean2 = np.mean(PV_2D2, axis=1)
    if error_type == 'jack':
        PV_err1 = np.std(PV_2D1, axis=1)*np.sqrt(n_sample-1.)
        PV_err2 = np.std(PV_2D2, axis=1)*np.sqrt(n_sample-1.)
    elif error_type == 'boot':
        PV_err1 = np.std(PV_2D1, axis=1)
        PV_err2 = np.std(PV_2D2, axis=1)
    PV_std1 = np.sqrt(np.diag(PV_cov1))
    PV_std2 = np.sqrt(np.diag(PV_cov2))
    print("std vs err 1 [percent] = ", ((PV_err1-PV_std1)*100./PV_err1)[0])
    print("std vs err 2 [percent] = ", ((PV_err2-PV_std2)*100./PV_err2)[0])

    choice = (rbinc < r_max) & (rbinc > r_min)
    print(f"data/{galaxy_sample:s}_{cmb_sample1}_{vary_str:s}Th{Theta:.2f}_PV")
    print("dof = ", np.sum(choice))
    PV_cov1_red = np.cov(PV_2D1[choice, :])
    PV_cov2_red = np.cov(PV_2D2[choice, :])
    if error_type == 'jack':
        PV_cov1_red *= (n_sample-1.)
        PV_cov2_red *= (n_sample-1.)
    print("chi2 = ", np.dot(PV_mean1[choice], np.dot(np.linalg.inv(PV_cov1_red), PV_mean1[choice])))
    print("chi2_rand = ", np.dot(PV_rand1[choice], np.dot(np.linalg.inv(PV_cov1_red), PV_rand1[choice])))
    #print("chi2_1 = ", np.dot(PV_mean1[choice], np.dot(np.linalg.inv(PV_cov1_red), PV_mean1[choice])))
    #print("chi2_2 = ", np.dot(PV_mean2[choice], np.dot(np.linalg.inv(PV_cov2_red), PV_mean2[choice])))
    print("cond number = ", np.linalg.cond(PV_cov1_red))#, np.linalg.cond(PV_cov2_red))
    print("------------------------------------------------")
    inv_Err2 = 1./PV_err1**2 + 1./PV_err2**2
    PV_mean = (PV_mean1/PV_err1**2 + PV_mean2/PV_err2**2)/inv_Err2
    PV_rand = (PV_rand1/PV_err1**2 + PV_rand2/PV_err2**2)/inv_Err2  # kinda made up
    PV_err = np.sqrt(1./inv_Err2)

    # I am gonna multiply by np.sqrt(2) since these are the same thing
    if same:
        PV_err *= np.sqrt(2) # because otherwise we get twice smaller error
    
    plt.subplot(2, 2, i+1)
    plt.plot(rbinc, np.zeros(len(PV_mean)), 'k--')
    plt.errorbar(rbinc, PV_rand, ls='--', color='dodgerblue', capsize=4.)
    plt.errorbar(rbinc, PV_mean, yerr=PV_err, color='dodgerblue', capsize=4.)
    
    plt.ylabel(r"$\hat p_{kSZ}(r) \ [\mu {\rm K}]$")
    plt.xlabel(r"$r \ [{\rm Mpc}]$")
    plt.ylim([-0.35, 0.35])
    plt.xlim([0., 150.])
    if same:
        text = (" ".join(galaxy_sample.split("_"))).split(" premask")[0].split(" mtype")[0]+", ACT ("+cmb_sample1.split('_')[-1]+rf"), $\Theta$={Theta:.1f}'"
    else:
        assert (("D56" in cmb_sample1) and ("BN" in cmb_sample2)) or (("D56" in cmb_sample2) and ("BN" in cmb_sample1))
        text = (" ".join(galaxy_sample.split("_"))).split(" premask")[0].split(" mtype")[0]+rf", ACT (DR4), $\Theta$={Theta:.1f}'"      
    #text = (" ".join(galaxy_sample.split("_"))).split(" premask")[0].split(" mtype")[0]+mask_str+", ACT ("+cmb_sample1.split('_')[-1]+", "+cmb_sample2.split('_')[-1]+rf"), $\Theta$={Theta:.1f}'"
    plt.text(x=0.03, y=0.1, s=text, transform=plt.gca().transAxes)

plt.savefig(save_fn)
#plt.close()
plt.show()
