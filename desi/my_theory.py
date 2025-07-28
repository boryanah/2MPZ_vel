import warnings
import time

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

import camb
from camb import model, initialpower
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad
from scipy.interpolate import interp1d, CubicSpline

import hmf
from hmf import MassFunction
from colossus.lss import bias, peaks, mass_function
from colossus.cosmology import cosmology

from numba import jit
import numba
warnings.filterwarnings('ignore')

# https://arxiv.org/abs/1603.03904

# TODO SAME COSMOLOGY
cosmo_colossus = cosmology.setCosmology('planck18')

# parameters
Msun_to_g = 2e33
Mpc_to_cm = 3.086e24 
redshift = 0.52
#redshift = 0.32
Omega_M = 0.32
H0 = 67.3
cosmo = FlatLambdaCDM(H0 = H0, Om0 = Omega_M)
rho_crit = cosmo.critical_density(redshift).value #g/cm
h = cosmo_colossus.h
print("h = ", h)

# load HOD at z = 2.5 Mstar > 1.e11
mbinc = np.load("data/mbinc.npy")
hod_cent = np.load("data/hod_cent.npy")
hod_sats = np.load("data/hod_sats.npy")
hod_gals = hod_cent+hod_sats
choice = ~np.isnan(hod_gals) & (hod_gals > 1.e-3)
hod_gals = hod_gals[choice]
mbinc = mbinc[choice]
N_hod_logf = CubicSpline(np.log(mbinc), np.log(hod_gals))
N_hod_f = lambda m: np.exp(N_hod_logf(np.log(m)))

#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.3, ombh2=0.02225, omch2=0.1198)
pars.InitPower.set_params(ns=0.964)
results = camb.get_results(pars)
pars.set_matter_power(redshifts=[redshift], kmax=10)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)
s8 = np.array(results.get_sigma8())
fs8 = results.get_fsigma8()
f = fs8/s8

# no h units
PK = camb.get_matter_power_interpolator(pars, zmin=0, zmax=2, kmax=1e3, hubble_units=False, k_hunit=False, nonlinear=False, return_z_k=False, extrap_kmax=1e12)

def Da(z):
    a = 1/(1+z)
    H_z = results.hubble_parameter(z)
    def integrand2(x):
        return 1/(x*results.hubble_parameter(1/x-1)/H0)**3
    return 5*Omega_M/2*H_z/H0*quad(integrand2, 0, a)[0]

# growth factor (only used in bias measurement)
Daz = Da(redshift)
Da0 = Da(0.)

# define halo masses
Mass = np.logspace(12, 16, 40000) # Msun

# collossus expects h units and returns h^3 but we give it M in solar mass
# Eve says she used virial mass from Kravtsov
mdef = '200m' # better
#mdef = '200c'
#mdef = '200t' # doesn't exist
model_hmf = 'tinker08' # better
#model_hmf = 'bhattacharya11'; mdef = 'fof'
"""
dndlnm = mass_function.massFunction(Mass * h, z=redshift, mdef=mdef, model=model, q_in='M', q_out='dndlnM') # h^3
dndm = dndlnm / Mass
dndm *= h**3. # 1/Msun 1/Mpc^3
dndm_logf = CubicSpline(np.log(Mass), np.log(dndm))
dndm_f = lambda m: np.exp(dndm_logf(np.log(m)))
"""
dndm_f = lambda m: mass_function.massFunction(m * h, z=redshift, mdef=mdef, model=model_hmf, q_in='M', q_out='dndlnM')/m * h**3. # h^3

# colossus expects h units 
model_bias = 'tinker10'# better
#model_bias = 'cole89' # og Yulin script
#model_bias = 'sheth01'
#model_bias = 'bhattacharya11'
bias_f = lambda m: bias.haloBias(m * h, model=model_bias, z=redshift, mdef=mdef)


# I think this isn't right
"""
#MF = MassFunction(z=redshift,Mmin=13,Mmax=16,dlog10m = 0.00001,hmf_model="Bhattacharya")
m_min = 12
m_max = 16
Mass = np.logspace(m_min, m_max, 400000) # solar mass
MF = MassFunction(z=redshift, Mmin=m_min*h, Mmax=m_max*h, dlog10m = 0.00001, hmf_model="Bhattacharya") # this returns h^4 units for dndm
dndm = MF.dndm # h^4
dndm *= h**4 # h-unitless
#n_m = hmf.mass_function.integrate_hmf.hmf_integral_gtm(Mass, MF.dndm, mass_density=False) # Mpc^-3
"""
# it has to be dndm in the formula in that thing because n(M) depends on how big your dM's are which makes no sense so you have to divide
# imagine you have 10k obj in one bin and 10k in the next one; if you now combined both, voila, you get 20k but that makes no sense
"""
# matches abacus-ish
print(Mass)
print(n_m)
#plt.plot(Mass, dndm*2000**3)
plt.plot(Mass*h, dndlnm*(np.log(Mass[1])-np.log(Mass[0]))*2000.**3)
plt.xscale('log')
plt.yscale('log')
plt.show()
"""

# find n(m) at m
@jit(nopython=True)
def find_n_m(m):
    difference_array = np.absolute(Mass-m)
    index = difference_array.argmin()
    #return n_m[index]
    return dndm[index]

@jit(nopython=True)
def find_hod(m):
    difference_array = np.absolute(Mass-m)
    index = difference_array.argmin()
    return N_hod[index]

"""
print("find", find_n_m(1.e13))
#n_m_f = interp1d(Mass, n_m)
n_m_logf = CubicSpline(np.log(Mass), np.log(n_m))
n_m_f = lambda m: np.exp(n_m_logf(np.log(m)))
print("interp", n_m_f(1.e13))
# number density of halos at some mass
"""

# radius
@jit(nopython=True)
def R(M):
    return (M * Msun_to_g * 3 / (Omega_M*rho_crit*4*np.pi))**(1/3) / Mpc_to_cm #Mpc

# tophat
@jit(nopython=True)
def W(x):
    return 3 * (np.sin(x) - x * np.cos(x)) / x**3

def sigma0_2_integrand(k,M,z):
    return (k**2) * PK.P(0,k) * W(k*R(M))**2 

def b_h(M):
    sigma02 = quad(sigma0_2_integrand, 0, np.inf,(M,0))[0]/ (2 * np.pi**2)
    return 1 + (1.686**2 - sigma02)/(sigma02*1.686*Daz/Da0)

def b_numerator_integrand(M, q):
    #return M * find_n_m(M) * b_h(M)**q # * find_hod(M) # TESTING
    #return M * find_n_m(M) * bias.haloBias(M*h, model='tinker10', z=redshift, mdef=mdef)**q
    return M * dndm_f(M) * bias_f(M)**q * N_hod_f(M)    

#@jit(nopython=True)
def b_denominator_integrand(M):
    #return M * n_m_f(M) * N_hod_f(M)
    #return M * find_n_m(M)# * find_hod(M) # TESTING!
    return M * dndm_f(M) * N_hod_f(M)

def b(Mmin, Mmax, q):
    return quad(b_numerator_integrand, Mmin, Mmax, q)[0]/quad(b_denominator_integrand, Mmin, Mmax)[0]

@jit(nopython=True)
def j0(x):
    return np.sin(x)/x

@jit(nopython=True)
def j1(x):
    return np.sin(x)/x**2 - np.cos(x)/x

def Xi_integrand(k,r):
    return k**2 * PK.P(redshift,k) * j0(k*r)

# k integral; r is carried over
def Xi(r):
    return quad(Xi_integrand, 0, np.inf,r)[0]/ (2 * np.pi**2)

# k rather than k^2 means you are less sensitive to small scales
def Xi_dv_integrand(k, r):
    return k * PK.P(redshift, k) * j1(k*r)

# - f(z) a(z) H(z) (km/s/Mpc); k integral; r is carried over
def Xi_dv(r):
    return -f*results.hubble_parameter(redshift)/(1+redshift) * quad(Xi_dv_integrand, 0, np.inf, r)[0]/ (2 * np.pi**2)

# rsep
rsep_org = np.loadtxt("data/L79_S18_ksz_vij_iz1.dat")[:, 2] # cMpc
M_max = 1.e16 # Msun

# version 1
# all are virial masses
#V_org = np.loadtxt("data/L79_S18_ksz_vij_iz1.dat")[:, 3]; M_min = 1.66*1.e13 # Msun 
#V_org = np.loadtxt("data/L61_S18_ksz_vij_iz1.dat")[:, 3]; M_min = 1.00*1.e13 # Msun
V_org = np.loadtxt("data/L43_S18_ksz_vij_iz1.dat")[:, 3]; M_min = 0.52*1.e13 # Msun
name_str = f"_M{np.log10(M_min):.1f}-{np.log10(M_max):.1f}"
N_hod_f = lambda m: 1.
print(f"{mbinc[0]:.2e}, {M_min:.2e}")

M_min *= h
"""
# version 2
M_min = mbinc[0]
name_str = f"_HOD"
"""

V_new = np.zeros(len(rsep_org))
for i in range(len(V_new)):
    print(i)
    #print(b(M_min, M_max, 1)**2 * Xi(rsep_org[i]))
    V_new[i] = -2. * b(M_min, M_max, 1) * Xi_dv(rsep_org[i]) / (1 + b(M_min, M_max, 1)**2 * Xi(rsep_org[i]))
    #V_new[i] = -2. * b(M_min, M_max, 1) * Xi_dv(rsep_org[i]) / (1 + b(M_min, M_max, 2) * Xi(rsep_org[i]))
    print("rsep, Vorg, Vnew", rsep_org[i], V_org[i], V_new[i], np.abs(V_org[i]-V_new[i])/V_org[i]*100.)
np.savez(f"V_pairwise_z{redshift:.2f}{name_str}.npz", rsep=rsep_org, V=V_new)

plt.figure(1)
plt.plot(rsep_org, V_org, 'o', label='Calafut21')
plt.plot(rsep_org, V_new, ls='--', label='Soergel')
plt.legend(fontsize=14)
plt.xlabel('$r$ [Mpc]', fontsize=14)
plt.ylabel("$\^V$ [km/s]", fontsize = 14)
plt.show()
quit()

rsep_new = np.array([  5 ,  15 ,  25. ,  35. ,  45. ,  54.99 ,  65. ,  75. ,  85. ,
        95. , 105. , 115. , 125. , 135. , 145. , 175. , 225. , 282.5, 355. ])

V_new = np.zeros(len(rsep_new))
for i in range(len(V_new)):
    V_new[i] = -2*b(1e13,1e16,1)*Xi_dv(rsep_new[i])/(1+b(1e13,1e16,1)**2*Xi(rsep_new[i]))

plt.figure(2)
plt.plot(rsep_org,V_org,'o',label = 'Calafut21')
plt.plot(rsep_new,V_new,ls = '--',label = 'following Soergel et al.')
plt.legend(fontsize = 14)
plt.xlabel('r [Mpc]', fontsize = 14)
plt.ylabel("$\^V$ [km/s]", fontsize = 14)
