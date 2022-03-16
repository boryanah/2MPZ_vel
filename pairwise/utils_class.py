"""
Script that solves for z given the luminosity distance and Cosmology
"""
from scipy.optimize import fsolve

from classy import Class

def func(z, d_target, Cosmo):
    return Cosmo.luminosity_distance(z) - d_target    
    

def lum_dist_to_redshift(Cosmo, d_L, z0=0.):
    try:
        z_sol = fsolve(func, args=(d_L, Cosmo), x0=z0)[0]
    except:
        print("couldn't compute")
        z_sol = 0.
    return z_sol

def main():
    # pass input parameters
    param_dict = {"h": 0.7, "Omega_Lambda": 0.7, "Omega_cdm": 0.25, "Omega_b": 0.05}

    # create instance of the class "Class"
    Cosmo = Class()
    Cosmo.set(param_dict)
    Cosmo.compute()

    # test that script works
    z = 0.5
    d = Cosmo.luminosity_distance(z)
    d = 100.
    print("input = ", z, d)
    print("solution = ", lum_dist_to_redshift(Cosmo, d))
    d = 350.
    print("input = ", z, d)
    print("solution = ", lum_dist_to_redshift(Cosmo, d))

    
if __name__ == "__main__":
    main()
    
"""
from scipy.optimize import fsolve
def guess_comoving_distance(z):
    d = Cosmo.luminosity_distance(z)

    def func(z, d_target):
        return Cosmo.luminosity_distance(z) - d_target    
    
    z0 = 0.1
    z_solve = fsolve(func, args=(d), x0=z0)
    print(z_solve)
    return
"""
