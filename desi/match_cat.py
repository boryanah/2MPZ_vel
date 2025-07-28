from astropy.table import Table,vstack,join
import warnings
import os
import numpy as np
import fitsio
import sys

#fname_fastspecfit_bright = "/global/cfs/cdirs/desi/spectro/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-bright.fits"
fname_fastspecfit_bright = "/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/IronPhysProp_v1.1.fits"
#fname_fastspecfit_dark = "/global/cfs/cdirs/desi/spectro/fastspecfit/iron/v2.1/catalogs/fastspec-iron-main-dark.fits"

tab_fastspecfit_bright = Table(fitsio.read(fname_fastspecfit_bright, columns=['LOGM', 'TARGETID', 'RA', 'DEC', 'Z']))
print(len(tab_fastspecfit_bright))
#print(tab_fastspecfit_bright.keys()); quit()
#,columns=["TARGETID","LOGMSTAR","ABSMAG01_SDSS_R","KCORR01_SDSS_R"]))
#tab_fastspecfit_dark = Table(fitsio.read(fname_fastspecfit_dark,columns=["TARGETID","LOGMSTAR","ABSMAG01_SDSS_R","KCORR01_SDSS_R"]))

# print(tab_fastspecfit_bright.colnames)
# print(tab_fastspecfit_dark.colnames)

tab = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto.dat.fits", columns=['TARGETID', 'RA', 'DEC', 'Z']))
print(tab.keys())
print(len(tab))

from match_searchsorted import match

new_tab = {}
ptr = match(np.asarray(tab['TARGETID']).astype(np.int64), np.asarray(tab_fastspecfit_bright['TARGETID']).astype(np.int64))
LOGM = tab_fastspecfit_bright['LOGM'][ptr[ptr > -1]]
RA = tab_fastspecfit_bright['RA'][ptr[ptr > -1]]
DEC = tab_fastspecfit_bright['DEC'][ptr[ptr > -1]]
Z = tab_fastspecfit_bright['Z'][ptr[ptr > -1]]
TARGETID = tab_fastspecfit_bright['TARGETID'][ptr[ptr > -1]]
print("matches", np.sum(ptr > -1), np.sum(ptr > -1)/len(ptr))
#[ptr > -1] ptr[ptr > -1]
print(RA[:10], tab['RA'][ptr > -1][:10])
print(DEC[:10], tab['DEC'][ptr > -1][:10])
print(Z[:10], tab['Z'][ptr > -1][:10])
new_tab = Table(new_tab)
new_tab['RA'] = RA
new_tab['DEC'] = DEC
new_tab['Z'] = Z
new_tab['TARGETID'] = TARGETID
new_tab['LOGM'] = LOGM
new_tab.write("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto_vac_marvin.dat.fits", overwrite=False)

quit()

def match_catalogue(tab,galaxy_type):
    if galaxy_type in ["BGS","BGS_BRIGHT"]:
        tab_fastspecfit = tab_fastspecfit_bright
    elif galaxy_type in ["LRG","ELG"]:
        tab_fastspecfit = tab_fastspecfit_dark
    else:
        raise ValueError(f"galaxy_type {galaxy_type} not recognized")
    
    len_tab = len(tab)
    joined_tab = join(tab,tab_fastspecfit,keys="TARGETID",join_type="left")
    len_joined_tab = len(joined_tab)
    assert len_tab == len_joined_tab, f"len_tab {len_tab} != len_joined_tab {len_joined_tab}"
    return joined_tab

def run_match_catalogue(version):
    from glob import glob
    fpath = "/global/cfs/cdirs/desicollab/science/c3/DESI-Lensing/desi_catalogues/" + version + "/"
    files = glob(fpath+'*.fits')

    for fpath_load in files:
        dirpath = os.path.dirname(fpath_load)+'/'
        filename = os.path.basename(fpath_load)
        if('.ran.fits' in filename):
            continue
        print("Reading "+fpath_load)
        galcat_data = Table.read(fpath_load)
        galaxy_type = filename.split("_")[0]

        galcat_data = match_catalogue(galcat_data,galaxy_type)
        print("Writing "+fpath_load)
        galcat_data.write(fpath_load,overwrite=True)

if(__name__=="__main__"):
    #version = sys.argv[1]
    #run_match_catalogue(version)
    tab = Table(fitsio.read("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto.dat.fits", columns=['TARGETID', 'RA', 'DEC', 'Z']))
    print(tab.keys())
    print(len(tab))
    galcat_data = match_catalogue(tab, "BGS")
    print(len(galcat_data))
    galcat_data.write("/mnt/marvin1/boryanah/2MPZ_vel/bgs_data/BGS_BRIGHT_full_noveto_vac.dat.fits", overwrite=False)
