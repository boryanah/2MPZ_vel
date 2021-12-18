import time

import Corrfunc # only using for comparison
import numpy as np

#from estimator import pairwise_velocity_box as pairwise_velocity; periodic = True # assumes LOS along z direction
from estimator import pairwise_velocity_sky as pairwise_velocity; periodic = False

# defining bins
#bins = np.linspace(2.5, 24.3, 19); is_log_bin = False
bins = np.geomspace(2.5, 24.3, 19); is_log_bin = True

#nthread = 1
nthread = 8

#dtype = np.float32 # THE ASSERTION MIGHT BREAK CAUSE OF NUMERICAL STUFF
dtype = np.float64 # THE ASSERTION SHOULD HOLD
boxsize = 1000.
print("nthread = ", nthread)
print("dtype = ", dtype)
print("boxsize = ", boxsize)


Ns = [1000, 10000, 100000]
for i in range(len(Ns)):
    N = Ns[i]
    print("number of tracers = ", N)
    n = N/boxsize**3.
    print(f"number density = {n:.5f}")
    
    X = np.random.uniform(0., boxsize, N)
    Y = np.random.uniform(0., boxsize, N)
    Z = np.random.uniform(0., boxsize, N)
    P = np.vstack((X, Y, Z)).T
    V_los = np.random.uniform(10., 20., N)

    P = P.astype(dtype)
    V_los = V_los.astype(dtype)

    DD, PV = pairwise_velocity(P, V_los, boxsize, bins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
    t1 = time.time()
    DD, PV = pairwise_velocity(P, V_los, boxsize, bins, is_log_bin=is_log_bin, dtype=dtype, nthread=nthread)
    time_numba = time.time()-t1
    #print(DD, PV)

    t1 = time.time()
    res = Corrfunc.theory.DD(1, nthread, bins, *P.T, periodic=periodic, boxsize=boxsize)
    time_corrfunc = time.time()-t1
    #print(res['npairs'])

    assert np.sum(res['npairs'] - DD) == 0.
    

    print("Corrfunc time = ", time_corrfunc)
    print(f"Corrfunc is {time_numba/time_corrfunc:.3f} times faster")
    print("------------------------------------------------")

quit()
# Lehman's implementation of pairwise velocities, which I don't fully understand though it seems super similar
bins = np.logspace(-1, 1, 4)

pos = np.array([[0., 0., 0.],
        [1., 0., 0.]])
vel = np.array([[0., 0., 0.],
        [1., 0., 0.]])

res = Corrfunc.theory.DD(1, 4, bins, *pos.T, weights1=vel.T, periodic=periodic, boxsize=100., verbose=True, weight_type='pairwise_vel', isa='fallback')

print(res['npairs'])
