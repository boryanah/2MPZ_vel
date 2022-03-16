import numpy as np

from astropy.io import ascii
from classy import Class

from utils_class import lum_dist_to_redshift

save_dir = "/mnt/marvin1/boryanah/2MPZ_vel/2mpz_data/"

# cosmological parameters
wb = 0.02225
wc = 0.1198
s8 = 0.83
ns = 0.964
h = 67.3/100.
COSMO_DICT = {'h': h,
              'omega_b': wb,
              'omega_cdm': wc,
              'A_s': 2.100549e-09, # doesn't affect calculation
              #'sigma8': s8,
              'n_s': ns}


# create instance of the class "Class"
Cosmo = Class()
Cosmo.set(COSMO_DICT)
Cosmo.compute() 

#['Name', 'RAJ2000', 'DEJ2000', 'GLON', 'GLAT', 'flag_asso', 'distL', 'r_distL', 'Mstar', 'r_Mstar', 'cNm', 'SFR', 'r_SFR', 'cNs', 'PGC', 'AllWISE', 'IDX']
key_import = ['Name', 'AllWISE', 'RAJ2000', 'DEJ2000','distL', 'Mstar', 'GLON', 'GLAT']
table = ascii.read(save_dir+"table5.mrt", include_names = key_import)
choice = table['Name'] == 'clone'
table = table[~choice]
print("dulicates = ", np.sum(choice)*100./len(choice))

RA = np.array(table['RAJ2000'], dtype=float)
DEC = np.array(table['DEJ2000'], dtype=float)
distL = np.array(table['distL'], dtype=float)
Mstar = np.array(table['Mstar'], dtype=float)
GLON = np.array(table['GLON'], dtype=float)
GLAT = np.array(table['GLAT'], dtype=float)
#AllWISE = np.array(table['AllWISE'], dtype=int)
Z = np.zeros(len(RA), dtype=float)

print("RA")
print(RA)
print(RA.dtype)
print(RA.min(), RA.max())

print("DEC")
print(DEC)
print(DEC.dtype)
print(DEC.min(), DEC.max())

print("distL")
print(distL)
print(distL.dtype)
print(distL.min(), distL.max())

print("Mstar")
print(Mstar)
print(Mstar.dtype)
print(Mstar.min(), Mstar.max())


for i in range(len(Z)):
    if i%1000 == 0: print(i, Z[i-1])
    #print(distL[i])
    Z[i] = lum_dist_to_redshift(Cosmo, distL[i])
    #print(Z[i])
    
print("Z")
print(Z)
print(Z.dtype)
print(Z.min(), Z.max())

np.savez(save_dir+"2MPZ_Biteau.npz", Z=Z, RA=RA, DEC=DEC, d_L=distL, M_star=Mstar, L=GLON, B=GLAT)
