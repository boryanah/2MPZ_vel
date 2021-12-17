from scipy.optimize import fsolve

from classy import Class

# create instance of the class "Class"
Cosmo = Class()

# pass input parameters
param_dict = {"h": 0.7, "Omega_Lambda": 0.7, "Omega_cdm": 0.25, "Omega_b": 0.05}
Cosmo.set(param_dict)

# run class
Cosmo.compute()
z = 0.5
d = Cosmo.luminosity_distance(z)
print(d) #1322.0377771533842

def func(z, d_target):
    return Cosmo.luminosity_distance(z) - d_target    
    
z0 = 0.45
z_solve = fsolve(func, args=(d), x0=z0)
print(z_solve)
