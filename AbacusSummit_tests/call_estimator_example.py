import Corrfunc # only using for comparison
import numpy as np

from estimator_box import pairwise_velocity

bins = np.logspace(-1, 1, 4)

boxsize = 400.
N = 1000

X = np.random.uniform(0., boxsize, N)
Y = np.random.uniform(0., boxsize, N)
Z = np.random.uniform(0., boxsize, N)
P = np.vstack((X, Y, Z)).T
V_los = np.random.uniform(10., 20., N)

bins = np.linspace(0., 20., 21)

DD, PV = pairwise_velocity(P, V_los, boxsize, bins, dtype=np.float32, nthread=1)
print(DD, PV)

res = Corrfunc.theory.DD(1, 1, bins, *P.T, periodic=True, boxsize=boxsize)
print(res['npairs'])

# Lehman's implementation of pairwise velocities, which I don't fully understand though it seems super similar
pos = np.array([[0., 0., 0.],
        [1., 0., 0.]])
vel = np.array([[0., 0., 0.],
        [1., 0., 0.]])

res = Corrfunc.theory.DD(1, 4, bins, *pos.T, weights1=vel.T, periodic=True, boxsize=100., verbose=True, weight_type='pairwise_vel', isa='fallback')

print(res['npairs'])
