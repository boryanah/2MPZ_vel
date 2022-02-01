import numpy as np
from astropy.io import ascii

# downsample (maybe for each bin, remove randomly galaxies or weighted by their masses)

Nz = np.load("data/mock_2015_Nz.npy")
#zbins = np.load("data/mock_2015_zbins.npy")
cbins = np.load("data/mock_2015_cbins.npy")
print("target galaxies = ", np.sum(Nz))

sim_name = "AbacusSummit_base_c000_ph006"
redshift = 0.1
rsd = ""; N_gal = 91053734 # cheating a bit....
#rsd = "_rsd"; N_gal = 91053734 # cheating a bit....
mock_dir = f"/global/cscratch1/sd/boryanah/AbacusHOD_scratch/mocks_2MPZ_box/{sim_name:s}/z{redshift:.3f}/galaxies{rsd:s}/"
n_chunks = 10

pos_gal_all = np.zeros((N_gal, 3), dtype=np.float32)
vel_gal_all = np.zeros((N_gal, 3), dtype=np.float32)
sum = 0
for i in range(n_chunks):
    print("i = ", i)
    
    # filename
    fn_gal = f"{mock_dir:s}/LRGs_{i+1:d}.dat"

    # load the galaxies
    gals_arr = ascii.read(fn_gal)
    pos_gal = np.vstack((gals_arr['x'], gals_arr['y'], gals_arr['z'])).T
    vel_gal = np.vstack((gals_arr['vx'], gals_arr['vy'], gals_arr['vz'])).T
    
    pos_gal_all[sum : sum + pos_gal.shape[0]] = pos_gal
    vel_gal_all[sum : sum + pos_gal.shape[0]] = vel_gal
    sum += pos_gal.shape[0]
#assert sum == N_gal
pos_gal_all = pos_gal_all[:sum]
vel_gal_all = vel_gal_all[:sum]
print(pos_gal_all.shape)
print(pos_gal_all[:, 0].max())

# distance to observer
dist_gal_all = np.sqrt(np.sum(pos_gal_all**2, axis=1))

pos_new_all = np.zeros((2*np.sum(Nz), 3), dtype=np.float32)
vel_new_all = np.zeros((2*np.sum(Nz), 3), dtype=np.float32)
sum = 0
for i in range(len(cbins)-1):
    print("i = ", i)
    cchoice = (dist_gal_all > cbins[i]) & (dist_gal_all <= cbins[i+1])
    vel_gal = vel_gal_all[cchoice]
    pos_gal = pos_gal_all[cchoice]
    
    # want this number of mock galaxies
    N_target = Nz[i]
    
    # how many do we currently have?
    N_gal = np.sum(cchoice)
    down_factor = N_gal/N_target
    print("need to downsample by a factor of", down_factor)
    
    # draw a random number for each galaxy between 0 and 1
    p_gal = np.random.rand(N_gal)
    dchoice = (p_gal < 1./down_factor)
    pos_new_all[sum: sum+np.sum(dchoice)] = pos_gal[dchoice]
    vel_new_all[sum: sum+np.sum(dchoice)] = vel_gal[dchoice]
    
    sum += np.sum(dchoice)
print("expected, actual", np.sum(Nz), sum)
pos_new_all = pos_new_all[:sum]
vel_new_all = vel_new_all[:sum]
print(pos_new_all[:, 0].max())
np.save(f"{mock_dir:s}/pos_gal{rsd:s}_down.npy", pos_new_all)
np.save(f"{mock_dir:s}/vel_gal{rsd:s}_down.npy", vel_new_all)
