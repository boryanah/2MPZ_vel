import os
import glob

fns = glob.glob("data/*{mask_type}*")
mask_type = "mtype2"
#2MPZ_{mask_type}_premask_ACT_BN_fixedTh6.00_delta_Ts.npy
key = 'Th5.0'
key = '{mask_type}'

for i in range(len(fns)):
    old_fn = fns[i]
    half1 = old_fn.split(key)[0]
    half2 = "_".join((old_fn.split(key)[-1]).split('_')[1:])
    new_fn = half1+mask_type+"_"+half2
    print(old_fn, new_fn)
    os.rename(old_fn, new_fn)
