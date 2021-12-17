import time

import Corrfunc # only using for comparison
import numpy as np

from estimator_box import pairwise_velocity

bins = np.linspace(2.5, 24.3, 19)

#nthread = 1
nthread = 8

dtype = np.float32
#dtype = np.float64
boxsize = 1000.
print("nthread = ", nthread)
print("dtype = ", dtype)
print("boxsize = ", boxsize)

Ns = [1000, 10000, 100000]
for i in range(len(Ns)):
    N = Ns[i]
    print("number of tracers = ", N)

    X = np.random.uniform(0., boxsize, N)
    Y = np.random.uniform(0., boxsize, N)
    Z = np.random.uniform(0., boxsize, N)
    P = np.vstack((X, Y, Z)).T
    V_los = np.random.uniform(10., 20., N)

    P = P.astype(dtype)
    V_los = V_los.astype(dtype)

    DD, PV = pairwise_velocity(P, V_los, boxsize, bins, dtype=dtype, nthread=nthread)
    t1 = time.time()
    DD, PV = pairwise_velocity(P, V_los, boxsize, bins, dtype=dtype, nthread=nthread)
    time_numba = time.time()-t1
    #print(DD, PV)

    t1 = time.time()
    res = Corrfunc.theory.DD(1, nthread, bins, *P.T, periodic=True, boxsize=boxsize)
    time_corrfunc = time.time()-t1
    #print(res['npairs'])

    assert np.sum(res['npairs'] - DD) == 0.
    #print(res['npairs'] - DD)

    print(f"Corrfunc is {time_numba/time_corrfunc:.3f} times faster")
    print("------------------------------------------------")

quit()
# Lehman's implementation of pairwise velocities, which I don't fully understand though it seems super similar
bins = np.logspace(-1, 1, 4)

pos = np.array([[0., 0., 0.],
        [1., 0., 0.]])
vel = np.array([[0., 0., 0.],
        [1., 0., 0.]])

res = Corrfunc.theory.DD(1, 4, bins, *pos.T, weights1=vel.T, periodic=True, boxsize=100., verbose=True, weight_type='pairwise_vel', isa='fallback')

print(res['npairs'])
