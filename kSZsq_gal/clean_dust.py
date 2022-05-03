"""
Conversion factor taken from: https://irsa.ipac.caltech.edu/docs/knowledgebase/ercsc-validation-conversions.pdf. There is a typo for Table V, 545 GHz (Tthermo to MJy/sr): 57 -> 571.
"""
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

data_dir = "/mnt/marvin1/boryanah/2MPZ_vel/"
#T_cmb = hp.read_map(data_dir+"/cmb_data/COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits", verbose=True) # Kelvin
T_cmb = hp.read_map(data_dir+"/cmb_data/LGMCA/WPR2_CMB_muK.fits", verbose=True)/1.e6 # Kelvin
T_217 = hp.read_map(data_dir+"/cmb_data/dust/HFI_SkyMap_217-field-IQU_2048_R3.00_full.fits") # Kelvin
T_545 = hp.read_map(data_dir+"/cmb_data/dust/HFI_SkyMap_545-field-Int_2048_R3.00_full.fits") # MJy/sr
#T_545 /= 9125.753 # Kelvin (wrong)
T_545 /= 571.943 # Kelvin (right)
print("min max cmb = ", np.min(T_cmb), np.max(T_cmb))
print("min max 217 = ", np.min(T_217), np.max(T_217))
print("min max 545 = ", np.min(T_545), np.max(T_545))

# construct clean map
T_dust = 1.0085*(T_545-T_217)
alpha = -0.0002
T_clean = (1+alpha)*T_cmb-alpha*T_dust
#hp.write_map(data_dir+"/cmb_data/dust/COM_CMB_IQU-smica-nosz-nodust_2048_R3.00_full.fits", T_clean)
hp.write_map(data_dir+"/cmb_data/dust/WPR2_CMB_nodust_K.fits", T_clean)

min, max = -300/1.e6, +300/1.e6
hp.mollview(T_217, min=min, max=max, title='T217')
hp.mollview(T_545, title='T545')
hp.mollview(T_dust, title='Tdust')
hp.mollview(T_clean, min=min, max=max, title='Tclean')
hp.mollview(T_cmb, min=min, max=max, title='TCMB')
hp.mollview(T_cmb-T_clean, title='TCMB-Tclean')
plt.show()
