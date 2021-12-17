import numba
import numpy as np

def A_NFW(c):
    return np.log(1+c)-c/(1+c)

def f(c):
    return c/((c+1)*A_NFW(c))


@numba.njit(parallel=True,fastmath=True)
def pairwise_velocity(X, V_los, Lbox, bins, dtype=np.float32, nthread=1):
    """
    The formula is:
    v_1,2 = 2 Sum_A,B (s_A-s_B) p_AB / (Sum_A,B p_AB^2)
    p_AB = \hat r . (\hat r_A + \hat r_B)
    \vec r = \vec r_A - \vec r_B
    s_A = \hat r_A v_A
    
    Here we assume that the line-of-sight is along z and that we are in a
    box with periodic boundary conditions. Thus:
    \hat r_A = \hat r_B = 1
    Thus p_AB = 2 \hat r_z
    Here we are introducing V_los separately as an independently measured quantity
    """
    # TODO: check that the pairwise velocity thingy works
    # implement for logarithmic bins
    # implement for curved sky (i.e. LOS not z!)
    # fix nans which happen when we include self-counting and no pairs (can e.g. loop at the end)
    
    numba.set_num_threads(nthread)

    N = len(X)
    Lbox = dtype(Lbox)

    # only works for linear
    dbin = bins[1]-bins[0]

    pair_count = np.zeros(len(bins)-1, dtype=dtype)
    weight_count = np.zeros(len(bins)-1, dtype=dtype)
    norm_weight_count = np.zeros(len(bins)-1, dtype=dtype)
    
    for i in numba.prange(N):
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        s1 = V_los[i]
        
        for j in range(N):
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            s2 = V_los[j]
            
            dx = x2-x1
            dy = y2-y1
            dz = z2-z1

            if dx > Lbox/2.:
                dx -= Lbox
            if dy > Lbox/2.:
                dy -= Lbox
            if dz > Lbox/2.:
                dz -= Lbox
            if dx <= -Lbox/2.:
                dx += Lbox
            if dy <= -Lbox/2.:
                dy += Lbox
            if dz <= -Lbox/2.:
                dz += Lbox

            dist2 = dx**2 + dy**2 + dz**2
            dist = np.sqrt(dist2)

            hat_dz = dz/dist
            p12 = 2.*hat_dz
            
            weight = 2. * (s1-s2) * p12 
            norm_weight = p12**2.
            
            ind = np.int64(np.floor(dist/dbin))
            if ind < len(bins)-1:
                pair_count[ind] += 1.
                weight_count[ind] += weight 
                norm_weight_count[ind] += norm_weight

    pairwise_velocity = weight_count/norm_weight_count

    return pair_count, pairwise_velocity
