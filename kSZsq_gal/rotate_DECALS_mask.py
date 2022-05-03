import numpy as np

import healpy as hp
import matplotlib.pyplot as plt


data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
#mask_fn = data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36.fits" # equatorial
mask_fn = data_dir+"cmb_data/HFI_Mask_PointSrc_Gal70.fits" # galactic
map_Equ = hp.read_map(mask_fn) # ring, not nested
print("masked percentage before = ", np.sum(map_Equ)*100./len(map_Equ))

#rot_eq2gal = hp.Rotator(coord="CG")
rot_eq2gal = hp.Rotator(coord="GC") # galactic to equatorial
map_Gal = rot_eq2gal.rotate_map_alms(map_Equ)
map_Gal[map_Gal > 0.5] = 1.
map_Gal[map_Gal <= 0.5] = 0.
print("masked percentage after = ", np.sum(map_Gal)*100./len(map_Gal))

#hp.write_map(data_dir+f"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits", map_Gal) # galactic
hp.write_map(data_dir+f"cmb_data/HFI_Mask_PointSrc_Gal70_equatorial.fits", map_Gal) # equatorial

hp.mollview(map_Gal)
hp.mollview(map_Equ)

plt.show()
