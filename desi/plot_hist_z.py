import fitsio
from astropy.table import Table

import numpy as np
import matplotlib.pyplot as plt


#t = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto_vac.dat.fits")); LOGM_key = 'LOGM'
#t = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto_vac_perl.dat.fits")); LOGM_key = 'LOGMSTAR'
t = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto_vac_marvin.dat.fits")); LOGM_key = 'LOGM' # best!
#t = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_lensing.dat.fits")); LOGM_key = 'LOGMSTAR'
print(t.keys()) #['TARGETID', 'Z', 'RA', 'DEC', 'LOGM']

choice = ~np.isnan(t['Z']) & (t[LOGM_key] > 11.) & (t[LOGM_key] < 15.) & (t['Z'] < 1.) & (t['Z'] > 0.0)

T = t['TARGETID']
print("unique", len(np.unique(T))/len(T))
print(np.sum(choice))
Z = t['Z'][choice]
print(np.mean(Z), np.median(Z))
quit()

plt.hist(Z, bins=100)
plt.show()
