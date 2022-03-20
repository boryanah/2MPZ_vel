import healpy as hp
import matplotlib.pyplot as plt


data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
mask_fn = data_dir+"dels_data/Legacy_footprint_final_mask_cut_decm36.fits"
map_Equ = hp.read_map(mask_fn) # ring, not nested

rot_eq2gal = hp.Rotator(coord="CG")
map_Gal = rot_eq2gal.rotate_map_alms(map_Equ)
map_Gal[map_Gal > 0.5] = 1.
map_Gal[map_Gal <= 0.5] = 0.

hp.write_map(data_dir+f"dels_data/Legacy_footprint_final_mask_cut_decm36_galactic.fits", map_Gal)

hp.mollview(map_Gal)
hp.mollview(map_Equ)

plt.show()
