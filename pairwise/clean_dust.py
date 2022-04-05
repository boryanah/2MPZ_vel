import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
T_cmb = hp.read_map(data_dir+"/cmb_data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits", verbose=True) # Kelvin
T_217 = hp.read_map(data_dir+"/cmb_data/dust/HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits") # Kelvin
T_545 = hp.read_map(data_dir+"/cmb_data/dust/HFI_SkyMap_545-field-Int_2048_R3.00_full.fits") # MJy/sr
T_545 /= 57.1943 # Kelvin
print("min max cmb = ", np.min(T_cmb), np.max(T_cmb))
print("min max 217 = ", np.min(T_217), np.max(T_217))
print("min max 545 = ", np.min(T_545), np.max(T_545))

# construct clean map
T_dust = 1.0085*(T_545-T_217)
alpha = -0.0002
T_clean = (1+alpha)*T_cmb-alpha*T_dust
hp.write_map(data_dir+"/cmb_data/dust/COM_CMB_IQU-smica-nosz-nodust_2048_R3.00_full.fits", T_clean)

hp.mollview(T_dust)
hp.mollview(T_clean)
hp.mollview(T_cmb)
hp.mollview(T_cmb-T_clean)
plt.show()
