import glob
import numpy as np
import matplotlib.pyplot as plt

fns = sorted(glob.glob("data/*premask*PV_boot.npy"))
rbinc = np.load("data/rbinc.npy")
rmax = 80.
print(rmax)
imax = np.where(rbinc <= rmax)[0]
print(imax)

for i in range(len(fns)):
    PV_2D = np.load(fns[i])
    PV_2D = PV_2D[rbinc <= rmax, :]
    cov = np.cov(PV_2D)
    corr = np.corrcoef(PV_2D)
    name = fns[i].split('/')[-1]
    name = name.split('.npy')[0]
    plt.imshow(corr-np.eye(corr.shape[0]))
    plt.colorbar()
    plt.savefig(f"figs/cov/{name}.png")
    plt.close()
    print("file, cond num = ", name, '\t', np.linalg.cond(cov))
    print("----------------------------------")
