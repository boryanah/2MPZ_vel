import numba
import numpy as np

@numba.njit(parallel=True, fastmath=True)
def pairwise_velocity_sky(X, V_los, Lbox, bins, is_log_bin, dtype=np.float32, nthread=1):
    """
    The formula is:
    v_1,2 = 2 Sum_A,B (s_A-s_B) p_AB / (Sum_A,B p_AB^2)
    p_AB = \hat r . (\hat r_A + \hat r_B)
    \vec r = \vec r_A - \vec r_B
    s_A = \hat r_A v_A
    V_los is independently measured
    Does not use periodic boundary conditions
    """
    # TODO: check that the pairwise velocity thingy works
    
    numba.set_num_threads(nthread)

    N = len(X)
    Lbox = dtype(Lbox)
    zero = dtype(0.)
    one = dtype(1.)
    two = dtype(2.)

    # only works for linear
    if is_log_bin:
        dbin = dtype(bins[1]/bins[0])
    else:
        dbin = dtype(bins[1]-bins[0])
    fbin = dtype(bins[0])
    
    pair_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    norm_weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    pairwise_velocity = np.zeros(len(bins)-1, dtype=dtype)
    
    for i in numba.prange(N):
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        dist1 = np.sqrt(x1**two + y1**two + z1**two)

        if dist1 > zero:
            hx1, hy1, hz1 = x1/dist1, y1/dist1, z1/dist1
        else:
            hx1, hy1, hz1 = zero, zero, zero
        
        s1 = V_los[i]
        
        t = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(N):
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            dist2 = np.sqrt(x2**two + y2**two + z2**two)

            if dist2 > zero:
                hx2, hy2, hz2 = x2/dist2, y2/dist2, z2/dist2
            else:
                hx2, hy2, hz2 = zero, zero, zero
                
            s2 = V_los[j]
            
            dx = x2-x1
            dy = y2-y1
            dz = z2-z1

            """
            if dx > Lbox/two:
                dx -= Lbox
            if dy > Lbox/two:
                dy -= Lbox
            if dz > Lbox/two:
                dz -= Lbox
            if dx <= -Lbox/two:
                dx += Lbox
            if dy <= -Lbox/two:
                dy += Lbox
            if dz <= -Lbox/two:
                dz += Lbox
            """
            dist2 = dx**two + dy**two + dz**two
            dist = np.sqrt(dist2)
            
            
            if is_log_bin:
                ind = np.int64(np.floor(np.log(dist/fbin)/np.log(dbin)))
            else:
                ind = np.int64(np.floor((dist-fbin)/dbin))
                
            if (ind < len(bins)-1) and (ind >= 0):
                if dist > zero:
                    hx, hy, hz = dx/dist, dy/dist, dz/dist
                else:
                    hx, hy, hz = zero, zero, zero

                p12 = (hx1+hx2)*hx + (hy1+hy2)*hy + (hz1+hz2)*hz

                pair_count[t, ind] += one
                weight_count[t, ind] += two * (s1-s2) * p12
                norm_weight_count[t, ind] += p12**two

    pair_count = pair_count.sum(axis=0)
    weight_count = weight_count.sum(axis=0)
    norm_weight_count = norm_weight_count.sum(axis=0)
    
    for i in range(len(bins)-1):
        if norm_weight_count[i] != zero:
            pairwise_velocity[i] = weight_count[i]/norm_weight_count[i]
    

    return pair_count, pairwise_velocity


@numba.njit(parallel=True, fastmath=True)
def pairwise_velocity_box(X, V_los, Lbox, bins, is_log_bin, dtype=np.float32, nthread=1):
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
    Assumes periodic boundary conditions
    """
    # TODO: check that the pairwise velocity thingy works
    
    numba.set_num_threads(nthread)

    N = len(X)
    Lbox = dtype(Lbox)
    zero = dtype(0.)
    one = dtype(1.)
    two = dtype(2.)

    # only works for linear
    if is_log_bin:
        dbin = dtype(bins[1]/bins[0])
    else:
        dbin = dtype(bins[1]-bins[0])
    fbin = dtype(bins[0])
    
    pair_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    norm_weight_count = np.zeros((nthread, len(bins)-1), dtype=dtype)
    pairwise_velocity = np.zeros(len(bins)-1, dtype=dtype)
    
    for i in numba.prange(N):
        x1, y1, z1 = X[i][0], X[i][1], X[i][2]
        s1 = V_los[i]

        t = numba.np.ufunc.parallel._get_thread_id()
        
        for j in range(N):
            x2, y2, z2 = X[j][0], X[j][1], X[j][2]
            s2 = V_los[j]
            
            dx = x2-x1
            dy = y2-y1
            dz = z2-z1

            if dx > Lbox/two:
                dx -= Lbox
            if dy > Lbox/two:
                dy -= Lbox
            if dz > Lbox/two:
                dz -= Lbox
            if dx <= -Lbox/two:
                dx += Lbox
            if dy <= -Lbox/two:
                dy += Lbox
            if dz <= -Lbox/two:
                dz += Lbox

            dist2 = dx**tow + dy**two + dz**two
            dist = np.sqrt(dist2)

            if is_log_bin:
                ind = np.int64(np.floor(np.log(dist/fbin)/np.log(dbin)))
            else:
                ind = np.int64(np.floor((dist-fbin)/dbin))
                
            if (ind < len(bins)-1) and (ind >= 0):
                if dist > zero:
                    hat_dz = dz/dist
                else:
                    hat_dz = zero

                p12 = two * hat_dz
                pair_count[t, ind] += one
                weight_count[t, ind] += two * (s1-s2) * p12
                norm_weight_count[t, ind] += p12**two

    pair_count = pair_count.sum(axis=0)
    weight_count = weight_count.sum(axis=0)
    norm_weight_count = norm_weight_count.sum(axis=0)
    
    for i in range(len(bins)-1):
        if norm_weight_count[i] != zero:
            pairwise_velocity[i] = weight_count[i]/norm_weight_count[i]
    
    return pair_count, pairwise_velocity
