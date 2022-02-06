# if the target size was 1.1 cMpc
z_SDSS = 0.5
target_SDSS = 1.1 # cMpc
target_SDSS *= 1./(1+z_SDSS) # Mpc

# clusters have the same proper size but change comoving size

# much smaller clusters seen by 2MPZ (maybe 30% smaller)
z_2MPZ = 0.08
factor = 0.09187562/0.14437036
# for 1.e-4 # 0.122464 # for 3.e-4 # see notes.txt (at z = 0, {3.e-4,1.e-4} vs. 1.e-2)
target_2MPZ = target_SDSS * factor # Mpc
target_2MPZ /= 1./(1+z_2MPZ) # cMpc
print("comoving Mpc target 2MPZ = ", target_2MPZ)
# 0.5 or 0.6 depending on number density
