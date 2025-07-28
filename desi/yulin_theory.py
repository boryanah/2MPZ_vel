import camb
from camb import model, initialpower
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
import hmf
from hmf import MassFunction
import warnings
import time
from numba import jit
import numba
warnings.filterwarnings('ignore')

# https://arxiv.org/abs/1603.03904

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.02225, omch2=0.1198)
pars.InitPower.set_params(ns=0.964)
results = camb.get_results(pars)

redshift = 0.52

cosmo = FlatLambdaCDM(H0 = 67.3, Om0 = 0.314)
rho_crit = cosmo.critical_density(redshift).value #g/cm

Msun_to_g = 2e33
Mpc_to_cm = 3.086e24 

Omega_M = 0.32
H0 = 67.3


PK = camb.get_matter_power_interpolator(pars,zmin=0, zmax=2,kmax=1e3, hubble_units=False, k_hunit=False,nonlinear=False,return_z_k=False,extrap_kmax=1e6)


def Da(z):
    a = 1/(1+z)
    H_z = results.hubble_parameter(z)
    def integrand2(x):
        return 1/(x*results.hubble_parameter(1/x-1)/H0)**3
    return 5*Omega_M/2*H_z/H0*quad(integrand2, 0, a)[0]

# growth factor
Daz = Da(redshift)
Da0 = Da(0)

# mass function for halos of interest
MF = MassFunction(z=redshift,Mmin=13,Mmax=16,dlog10m = 0.00001,hmf_model="Bhattacharya")
Mass = np.logspace(13,16,300000)
n_m = hmf.mass_function.integrate_hmf.hmf_integral_gtm(Mass, MF.dndm, mass_density=False)


# find n(m) at m
@jit(nopython=True)
def find_n_m(m):
    difference_array = np.absolute(Mass-m)
    index = difference_array.argmin()
    return n_m[index]

# number density of halos at some mass

# radius
@jit(nopython=True)
def R(M):
    return (M * Msun_to_g * 3 / (Omega_M*rho_crit*4*np.pi))**(1/3) / Mpc_to_cm #Mpc

# tophat
@jit(nopython=True)
def W(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

# yes
def sigma0_2_integrand(k,M,z):
    return (k**2) * PK.P(0,k) * W(k*R(M))**2 

# yes
def b_h(M):
    sigma02 = quad(sigma0_2_integrand, 0, np.inf,(M,0))[0]/ (2 * np.pi**2)
    return 1 + (1.686**2 - sigma02)/(sigma02*1.686*Daz/Da0)

# yes
def b_numerator_integrand(M,q):
    return M * find_n_m(M) * b_h(M)**q

# yes
@jit(nopython=True)
def b_denominator_integrand(M):
    return M * find_n_m(M)

# yes
def b(Mmin,Mmax,q):
    return quad(b_numerator_integrand, Mmin,Mmax,q)[0]/quad(b_denominator_integrand,Mmin,Mmax)[0]

pars.set_matter_power(redshifts=[redshift], kmax=10)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
s8 = np.array(results.get_sigma8())
fs8 = results.get_fsigma8()
f = fs8/s8


@jit(nopython=True)
def j0(x):
    return np.sin(x)/x

@jit(nopython=True)
def j1(x):
    return np.sin(x)/x**2 - np.cos(x)/x

def Xi_integrand(k,r):
    return k**2 * PK.P(redshift,k) * j0(k*r)

def Xi(r):
    return quad(Xi_integrand, 0, np.inf,r)[0]/ (2 * np.pi**2)

def Xi_dv_integrand(k,r):
    return k * PK.P(redshift,k) * j1(k*r)

def Xi_dv(r):
    return -f*results.hubble_parameter(redshift)/(1+redshift) * quad(Xi_dv_integrand, 0, np.inf,r)[0]/ (2 * np.pi**2)

rsep_org = np.array([  5. ,  15. ,  25. ,  35. ,  45. ,  55. ,  65. ,  75. ,  85. ,
        95. , 105. , 115. , 125. , 135. , 145. , 175. , 225. , 282.5, 355. ])
V_org = np.array([ 92.110696, 170.114842, 190.432266, 179.17866 , 156.669427,
       133.348398, 112.655852,  95.249593,  80.858212,  69.016717,
        59.218883,  51.041904,  44.197739,  38.759021,  35.038824,
        26.921789,  15.055384,   8.242138,   4.400122])

V_new = np.zeros(len(rsep_org))
for i in range(len(V_new)):
    V_new[i] = -2*b(1e13,1e16,1)*Xi_dv(rsep_org[i])/(1+b(1e13,1e16,1)**2*Xi(rsep_org[i]))

plt.plot(rsep_org,V_org,'o',label = 'Calafut21')
plt.plot(rsep_org,V_new,ls = '--',label = 'following Soergel et al.')
plt.legend(fontsize = 14)
plt.xlabel('r [Mpc]', fontsize = 14)
plt.ylabel("$\^V$ [km/s]", fontsize = 14)

rsep_new = np.array([  5 ,  15 ,  25. ,  35. ,  45. ,  54.99 ,  65. ,  75. ,  85. ,
        95. , 105. , 115. , 125. , 135. , 145. , 175. , 225. , 282.5, 355. ])

V_new = np.zeros(len(rsep_new))
for i in range(len(V_new)):
    V_new[i] = -2*b(1e13,1e16,1)*Xi_dv(rsep_new[i])/(1+b(1e13,1e16,1)**2*Xi(rsep_new[i]))

plt.plot(rsep_org,V_org,'o',label = 'Calafut21')
plt.plot(rsep_new,V_new,ls = '--',label = 'following Soergel et al.')
plt.legend(fontsize = 14)
plt.xlabel('r [Mpc]', fontsize = 14)
plt.ylabel("$\^V$ [km/s]", fontsize = 14)
