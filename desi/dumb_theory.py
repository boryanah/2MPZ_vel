import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, quadrature, romberg, fixed_quad

# where does this approximation break down r > 30 Mpc?
#chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.aanda.org/articles/aa/pdf/2015/11/aa26051-15.pdf
#chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/astro-ph/0305078.pdf

H0 = 67.36 # km/s/Mpc
Om_m = 0.3089
h = H0/100.
sig8 = 0.81

r1 = 2.33 # Mpc/h
r2 = 3.51 # Mpc/h
g1 = 1.72
g2 = 1.28
r1 /= h # Mpc
r2 /= h # Mpc

def xi(r, sig8, r1, g1, r2, g2):
    return (sig8/0.83)**2 * ((r/r1)**(-g1) + (r/r2)**(-g2))

def alpha(g):
    return 1.2 - 0.65*g

def xi_bar(r, sig8, r1, g1, r2, g2):
    xi_intgrd = lambda x: xi(x, sig8, r1, g1, r2, g2)*x**2
    res = quad(xi_intgrd, 0, r)
    #res = quadrature(xi_intgrd, 0, r)
    xi_int, xi_err = res
    #xi_int = romberg(xi_intgrd, 1.e-6, r)
    return 3.*xi_int/(r**3 * (1. + xi(r, sig8, r1, g1, r2, g2)))

def v_12(r, H0, Om_m, sig8, r1, g1, r2, g2):
    xi_mean = xi_bar(r, sig8, r1, g1, r2, g2)
    g = 0.5*(g1+g2) # γ=−(dlnξ/dlnr)|ξ=1
    v = -2./3*H0*r*Om_m**0.6*xi_mean * (1. + alpha(g)*xi_mean)
    #v = -2./3*H0*r*Om_m**0.6*xi_mean / (1. + alpha(g)*xi_mean)
    return v

r_bins = np.linspace(0, 100, 501)
r_binc = 0.5*(r_bins[1:] + r_bins[:-1]) # Mpc
print(r_bins)

V = np.zeros(len(r_binc))
for i in range(len(r_binc)):
    V[i] = v_12(r_binc[i], H0, Om_m, sig8, r1, g1, r2, g2)/100.
print(V)

plt.figure(2)
plt.plot(r_binc, xi(r_binc, sig8, r1, g1, r2, g2))

plt.figure(1)
plt.plot(r_binc, V)
plt.xlabel("r [Mpc]")
plt.ylabel("V(r) [100 km/s]")
plt.ylim([-3, 0])
plt.show()
