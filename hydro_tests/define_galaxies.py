import numpy as np
import matplotlib.pyplot as plt

import plotparams
plotparams.buba()

fp_dm = 'fp'
data_dir = "/mnt/alan1/boryanah/MTNG/data_fp/"; pos_unit = 1.; Lbox = 500. # cMpc/h
#data_dir = "/mnt/gosling1/boryanah/TNG300/"; pos_unit = 1.e3; Lbox = 205. # cMpc/h
#snapshot = 264; z = 0.
#snapshot = 214; z = 0.5
snapshot = 237; z = 0.245
#snapshot = 99; z = 0.
if snapshot == 99:
    snap_str = ""
else:    
    snap_str = f"_{snapshot:d}"
h = 0.6744

# load subhalo fields at z = 0
if 'MTNG' in data_dir:
    SubhaloGrNr = np.load(f"{data_dir:s}/SubhaloGroupNr_{fp_dm:s}{snap_str:s}.npy")
else:
    SubhaloGrNr = np.load(f"{data_dir:s}/SubhaloGrNr_{fp_dm:s}{snap_str:s}.npy")
SubhaloStellarPhotometrics = np.load(f"{data_dir:s}/SubhaloStellarPhotometrics_{fp_dm:s}{snap_str:s}.npy") #U, B, V, K, g, r, i, z [mag]
SubhaloMstar = np.load(f"{data_dir:s}/SubhaloMassType_{fp_dm:s}{snap_str:s}.npy")[:, 4]*1.e10/h # Msun
SubhaloSFR = np.load(f"{data_dir:s}/SubhaloSFR_{fp_dm:s}{snap_str:s}.npy")
SubhalosSFR = SubhaloSFR/SubhaloMstar
SubhalosSFR[SubhaloMstar == 0.] = 0.

# load halo fields
"""
Group_M_TopHat200 = np.load(f"{data_dir:s}/Group_M_TopHat200_{fp_dm:s}{snap_str:s}.npy")*1.e10
Group_R_TopHat200 = np.load(f"{data_dir:s}/Group_R_TopHat200_{fp_dm:s}{snap_str:s}.npy")/pos_unit # comoving Mpc/h
print(np.median(Group_M_TopHat200), "top")
print(np.median(Group_R_TopHat200))
"""
Group_M_TopHat200 = np.load(f"{data_dir:s}/Group_M_Mean200_{fp_dm:s}{snap_str:s}.npy")*1.e10/h # Msun
Group_R_TopHat200 = np.load(f"{data_dir:s}/Group_R_Mean200_{fp_dm:s}{snap_str:s}.npy")/pos_unit/h # cMpc
"""
Group_M_TopHat200 = np.load(f"{data_dir:s}/Group_M_Crit200_{fp_dm:s}{snap_str:s}.npy")*1.e10
Group_R_TopHat200 = np.load(f"{data_dir:s}/Group_R_Crit200_{fp_dm:s}{snap_str:s}.npy")/pos_unit # comoving Mpc/h
print(np.median(Group_M_TopHat200), "crit")
print(np.median(Group_R_TopHat200))
"""

# choose galaxies
# version 1
"""
# how many galaxies
ngal = 0.02 # gal/(Mpc/h)^3
Ngal = int(ngal*Lbox**3)
print("number of galaxies = ", Ngal)
print("number density of galaxies = ", ngal)

gal_inds = np.argsort(SubhaloStellarPhotometrics[:, 3])[:Ngal]
print("maximum Ks and minimum Ks = ", SubhaloStellarPhotometrics[gal_inds[-1], 3], SubhaloStellarPhotometrics[gal_inds[0], 3])
#print("in our paper we do 13.9 mag")
"""
# version 2
gal_inds = np.arange(len(SubhaloMstar))
choice = SubhaloMstar > 1.e11
gal_inds = gal_inds[choice]
print("how many galaxies", len(gal_inds), len(gal_inds)/Lbox**3)

# central subhalos
_, sub_inds_cent = np.unique(SubhaloGrNr, return_index=True)
gal_inds_cent = np.intersect1d(sub_inds_cent, gal_inds)
print("percentage centrals = ", len(gal_inds_cent)*100./len(gal_inds))

# compute HOD
par_inds = SubhaloGrNr[gal_inds]
par_inds_cent = SubhaloGrNr[gal_inds_cent]
par_m200 = Group_M_TopHat200[par_inds]
par_m200_cent = Group_M_TopHat200[par_inds_cent]
par_r200 = Group_R_TopHat200[par_inds]
par_r200_cent = Group_R_TopHat200[par_inds_cent]
gal_mass = SubhaloMstar[gal_inds]
gal_ssfr = SubhalosSFR[gal_inds]

print("redshift = ", z )
print("median and mean r200 [cMpc] = ", np.median(par_r200), np.mean(par_r200)) # 0.068, 0.082
print(f"median and mean m200 [Msun] = {np.median(par_m200):.2e}, {np.mean(par_m200):.2e}, {np.min(par_m200):.2e}") # 0.068, 0.082
print("median and mean r200 [Mpc] = ", np.median(par_r200)*1./(1+z), np.mean(par_r200)*1./(1+z))
print(f"median and mean mass [Msun] = {np.median(gal_mass):.2e}, {np.mean(gal_mass):.2e}, {np.min(gal_mass):.2e}") # 0.068, 0.082

rbins = np.geomspace(0.01, par_r200.max(), 51)
rbinc = (rbins[1:] + rbins[:-1]) * .5
hist, _ = np.histogram(par_r200, bins=rbins)
plt.plot(rbinc, hist)
plt.xscale("log")
plt.xlabel("Comoving radius [Mpc]")
plt.show()

# mass bins
mbins = np.logspace(12, 15.5, 36)
mbinc = (mbins[1:]+mbins[:-1])*.5

# compute HOD
par_hist, _ = np.histogram(par_m200, bins=mbins)
par_hist_cent, _ = np.histogram(par_m200_cent, bins=mbins)
halo_hist, _ = np.histogram(Group_M_TopHat200, bins=mbins)
hod_hist = par_hist/halo_hist
hod_hist_cent = par_hist_cent/halo_hist
hod_hist_sats = hod_hist - hod_hist_cent
np.save("data/hod_cent.npy", hod_hist_cent)
np.save("data/hod_sats.npy", hod_hist_sats)
np.save("data/mbinc.npy", mbinc)
np.save("data/mbins.npy", mbins)

plt.figure(1, figsize=(9, 7))
plt.plot(mbinc, hod_hist, color="dodgerblue", ls='-', label="Total")
plt.plot(mbinc, hod_hist_cent, color="dodgerblue", ls='--', label="Centrals")
plt.xscale('log')
plt.yscale('log')
plt.ylabel(r"$\langle N_{\rm gal} \rangle$")
plt.xlabel(r"$M_{\rm halo} \ [M_\odot/h]$")
plt.legend()
#plt.show()


# test whether this is a cut in stellar mass
mstar_min = 1.e8 # Msun/h
mchoice = SubhaloMstar > mstar_min
par_m200_mcut = Group_M_TopHat200[SubhaloGrNr[mchoice]]
gal_mass_mcut = SubhaloMstar[mchoice]
gal_ssfr_mcut = SubhalosSFR[mchoice]

plt.figure(2, figsize=(9, 7))
alpha = 0.08
plt.scatter(gal_mass_mcut, gal_ssfr_mcut, s=1, color='orangered', alpha=alpha)
plt.scatter(gal_mass, gal_ssfr, s=10, color='dodgerblue', alpha=alpha)
plt.xlabel(r"$M_{\ast} \ [M_\odot/h]$")
plt.ylabel(r"${\rm sSFR}$")
plt.xscale('log')
plt.yscale('log')
plt.xlim([1.e8, 1.e11])
plt.ylim([1.e-13, 1.e-7])
plt.show()
