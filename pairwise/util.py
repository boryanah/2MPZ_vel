import numpy as np
import matplotlib.pyplot as plt
import numba
from scipy import interpolate
from scipy import stats
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq

from pixell import enmap, reproject, enplot, curvedsky, utils
from pixell import powspec

import rotfuncs


np.seterr(divide='ignore', invalid='ignore')

def eshow(x, fn, **kwargs): 
    ''' Define a function to help us plot the maps neatly '''
    plots = enplot.get_plots(x, **kwargs)
    #enplot.show(plots, method = "python")
    enplot.write("figs/"+fn, plots)

def cutoutGeometry(projCutout='cea', rApMaxArcmin=6., resCutoutArcmin=0.25, test=False):
    '''Create enmap for the cutouts to be extracted.
    Returns a null enmap object with the right shape and wcs.
    '''

    # choose postage stamp size to fit the largest ring
    dArcmin = np.ceil(2. * rApMaxArcmin * np.sqrt(2.))
   
    nx = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    dxDeg = (2. * nx + 1.) * resCutoutArcmin / 60.
    ny = np.floor((dArcmin / resCutoutArcmin - 1.) / 2.) + 1.
    dyDeg = (2. * ny + 1.) * resCutoutArcmin / 60.

    # define geometry of small square maps to be extracted
    shape, wcs = enmap.geometry(np.array([[-0.5*dxDeg,-0.5*dyDeg],[0.5*dxDeg,0.5*dyDeg]])*utils.degree, res=resCutoutArcmin*utils.arcmin, proj=projCutout)
    cutoutMap = enmap.zeros(shape, wcs)
    
    if test:
        print("cutout sides are dx, dy =", dxDeg*60., ",", dyDeg*60. , "arcmin")
        print("cutout pixel dimensions are", shape)
        print("hence a cutout resolution of", dxDeg*60./shape[0], ",", dyDeg*60./shape[1], "arcmin per pixel")
        print("(requested", resCutoutArcmin, "arcmin per pixel)")

    return cutoutMap

def extractStamp(cmbMap, ra, dec, rApMaxArcmin, resCutoutArcmin, projCutout, pathTestFig='figs/', test=False, cmbMask=None, order=1):
    """Extracts a small CEA or CAR map around the given position, with the given angular size and resolution.
    ra, dec in degrees.
    Does it for the map, the mask and the hit count.
    order > 1 is too slow
    """
    # enmap 
    stampMap = cutoutGeometry(rApMaxArcmin=rApMaxArcmin, resCutoutArcmin=resCutoutArcmin, projCutout=projCutout)
    stampMask = stampMap.copy()
    
    # coordinates of the square map (between -1 and 1 deg); output map position [{dec,ra},ny,nx]
    opos = stampMap.posmap()

    # coordinate of the center of the square map we want to extract
    sourcecoord = np.array([ra, dec])*utils.degree  # convert from degrees to radians

    # corresponding true coordinates on the big healpy map
    ipos = rotfuncs.recenter(opos[::-1], [0, 0, sourcecoord[0], sourcecoord[1]])[::-1]

    # Here, I use bilinear interpolation
    stampMap[:, :] = cmbMap.at(ipos, prefilter=True, mask_nan=False, order=order)
    if cmbMask is not None:
        stampMask[:, :] = cmbMask.at(ipos, prefilter=True, mask_nan=False, order=order)
    
        # re-threshold the mask map, to keep 0 and 1 only
        stampMask[:, :] = 1.*(stampMask[:, :]>0.5)
    
    if test:
        print("Extracted cutouts around ra=", ra, "dec=", dec)
        print("- min, mean, max =", np.min(stampMap), np.mean(stampMap), np.max(stampMap))
        # don't save if empty
        if np.min(stampMap) + np.mean(stampMap) + np.max(stampMap) == 0.: return opos, stampMap

        plots = enplot.plot(enmap.upgrade(stampMap, 5), grid=True)
        enplot.write(pathTestFig+"/stampmap_ra"+str(np.round(ra, 2))+"_dec"+str(np.round(dec, 2)), plots)

    if cmbMask is not None:
        return opos, stampMap, stampMask
    return opos, stampMap

@numba.guvectorize(['float64[:],float64[:],float64[:],float64[:],'
                    'float64[:],float64[:]'],
                   '(n),(n),(),()->(),()', target='parallel')
def get_tzav_and_w_nb(dT, z, zj, sigma_z, res1, res2):
    '''Launched by get_tzav to compute formula in parallel '''
    for i in range(dT.shape[0]):
        res1 += dT[i] * np.exp(-(zj[0]-z[i])**2.0/(2.0*sigma_z**2))
        res2 += np.exp(-(zj[0]-z[i])**2/(2.0*sigma_z**2))

def get_tzav_and_w_nb_dumb(dT, z, zj, sigma_z, res1, res2):
    '''Launched by get_tzav to compute formula (equiv. but slower) '''
    for j in range(len(zj)):
        for i in range(dT.shape[0]):
            res1[j] += dT[i] * np.exp(-(zj[j]-z[i])**2.0/(2.0*sigma_z**2))
            res2[j] += np.exp(-(zj[j]-z[i])**2/(2.0*sigma_z**2))

def get_tzav(dTs, zs, sigma_z):
    '''Computes the dT dependency to redshift.
    dTs: Temperature decrements
    zs: redshifts
    sigma_z: width of the gaussian to smooth out the moving window.'''
    #To test, run test_get_tzav_nb
    #   Create empty arrays to be used by numba in get_tzav_and_w_nb'''
    res1 = np.zeros(dTs.shape[0])
    res2 = np.zeros(dTs.shape[0])
    get_tzav_and_w_nb(dTs, zs, zs, sigma_z, res1, res2)
    return res1/res2


def get_tzav_fast(dTs, zs, sigma_z):
    '''Subsample and interpolate Tzav to make it fast.
    dTs: entire list of dT decrements
    zs: entire list of redshifts
    sigma_z: width of the gaussian kernel we want to apply.
    '''
    N_samples_in_sigmaz = 15  # in one width of sigmaz use Nsamples
    zmin, zmax = zs.min(), zs.max()
    delta_z = zmax - zmin

    # evaluates Tzav N times
    N_samples = int(round(delta_z/sigma_z)) * N_samples_in_sigmaz
    z_subsampled = np.linspace(zmin, zmax, N_samples)

    #now compute tzav as we usually do.
    res1 = np.zeros(z_subsampled.shape[0])
    res2 = np.zeros(z_subsampled.shape[0])
    get_tzav_and_w_nb(dTs, zs, z_subsampled, sigma_z, res1, res2)
    tzav_subsampled = res1/res2
    #interpolate
    f = interpolate.interp1d(z_subsampled, tzav_subsampled, kind='cubic')
    tzav_fast = f(zs)
    return tzav_fast

def calc_T_AP(imap, rad_arcmin, test=False, mask=None):
    modrmap = imap.modrmap()
    radius = rad_arcmin*utils.arcmin
    inner = modrmap < radius
    outer = (modrmap >= radius) & (modrmap < np.sqrt(2.)*radius)
    if mask is None:
        flux_inner = imap[inner].mean()
        flux_outer = imap[outer].mean()
    else:
        if (np.sum(mask[inner]) == 0.) or (np.sum(mask[outer]) == 0.):
            return 0.
        else:
            flux_inner = np.sum(imap[inner]*mask[inner])/np.sum(mask[inner])
            flux_outer = np.sum(imap[outer]*mask[outer])/np.sum(mask[outer])
    flux_diff = flux_inner-flux_outer
    return flux_diff


def compute_power(imap, modlmap=None, apod_pix=20, mask=None, test=False):
    if modlmap is None:
        # fourier magnitude mode map
        modlmap = imap.modlmap()
    
    # apodize and take fft
    taper = enmap.apod(imap*0+1, apod_pix) # I pass in an array of ones the same shape,wcs as imap
    if mask is not None:
        taper *= mask
    w2 = np.mean(taper**2.)
    kmap = enmap.fft(imap*taper, normalize="phys")

    # compute power in the map, bin, 
    power = np.real(kmap * np.conj(kmap))
    bin_edges = np.arange(0, 10000, 100)
    centers = (bin_edges[1:] + bin_edges[:-1])/2.
    binned_power = bin(power, modlmap, bin_edges)/w2

    if test:
        camb_theory = powspec.read_spectrum("camb_data/camb_theory.dat", scale=True) # scaled by 2pi/l/(l+1) to get C_ell
        cltt = camb_theory[0, 0, :3000]
        ls = np.arange(cltt.size)
    
        plt.plot(ls, cltt*ls*(ls+1)/(np.pi*2.), lw=3, color='k') # D_ell = C_ell l (l+1)/2pi (muK^2)
        plt.plot(centers, binned_power*centers*(centers+1.)/(np.pi*2.), marker="o", ls="none")
        plt.yscale('log')
        plt.xlabel('$\\ell$')
        plt.ylabel('$D_{\\ell}$')
        plt.show()

    return binned_power, centers

# This is a simple binning function that finds the mean in annular bins defined by bin_edges
def bin(data, modlmap, bin_edges):
    digitized = np.digitize(modlmap.flatten(), bin_edges, right=True)
    n_modes_bin = np.bincount(digitized)[1:-1]
    assert digitized.shape == (data.flatten()).shape, "Shape mismatch"
    power_bin = np.bincount(digitized, data.flatten())[1:-1]
    power = power_bin / n_modes_bin
    power = np.nan_to_num(power)
    return power

def gaussian_filter(modlmap, ell_0=1800, ell_sigma=500):
    # corresponds to 180./ell_0*60. arcmin
    ells = np.arange(0, 20000, 1)
    filt = np.exp(-(ells-ell_0)**2./2./ell_sigma**2.)
    filt_map = interpolate.interp1d(ells, filt, bounds_error=False, fill_value=0)(modlmap)
    return filt_map

def tophat_filter(modrmap, rad_arcmin):
    emap = modrmap*0
    
    radius = rad_arcmin*utils.arcmin
    inner = modrmap < radius
    outer = (modrmap >= radius) & (modrmap < np.sqrt(2.)*radius)
    emap[inner] = 1.
    emap[outer] = -1.
    filt_map = enmap.fft(emap, normalize='phys')
    return filt_map

def kSZ_filter(modrmap):
    # load theoretical kSZ power spectrum
    ell_ksz, cl_ksz = np.loadtxt("camb_data/cl_ksz_bat.dat", unpack=True)
    cl_ksz /= (ell_ksz*(ell_ksz+1)/(2.*np.pi))

    # get ell distance map
    modlmap = modrmap.modlmap()
    
    # translate into 2D fourier map
    emap = interpolate.interp1d(ell_ksz, cl_ksz, bounds_error=False, fill_value=0)(modlmap)
    
    # fourier transform
    filt_map = modlmap.copy()
    filt_map[:, :] = emap[:, :]
    return filt_map

def GNFW_filter(modrmap, theta_500=4., c_500=1.156, alpha=1.062, beta=5.4807, gamma=0.3292):
    """
    From David's paper. theta_500 = 1.8 # arcmin, 2.e13 Msun/h; theta_500 = 2.1 # arcmin, 2.e14 Msun/h
    """
    # arrray of angular sizes in real space
    thetas = np.linspace(modrmap.min(), modrmap.max(), 1001) # radians
    thetas = 0.5*(thetas[1:] + thetas[:-1])
    
    # translate to radians
    theta_500 *= utils.arcmin

    # compute filter in real space
    u_kSZs = np.zeros_like(thetas)
    for i in range(len(thetas)):
        u_kSZs[i] = u_kSZ(thetas[i], theta_500, c_500, alpha, beta, gamma)
    
    # translate into 2D map
    emap = interpolate.interp1d(thetas, u_kSZs, bounds_error=False, fill_value=None)(modrmap)
    emap = np.nan_to_num(emap)
    
    # fourier transform
    umap = modrmap.copy()
    umap[:, :] = emap[:, :]
    filt_map = enmap.fft(umap, normalize='phys')
    return filt_map
    
def calc_T_MF(imap, fmap=None, mask=None, power=None, test=False, apod_pix=20):
    # what do with mask??

    # make sure not empty
    assert fmap is not None
    assert power is not None

    # apodize and take fft
    taper = enmap.apod(imap*0+1., apod_pix) # pass an array of ones; same shape, wcs as imap
    #imap_tapered = (imap*taper) # for some reason with ACT this gives me highest SNR (apod = 20) than apod = 0 or below version (better than apod = 0?)
    # this seems the most physical version to me
    mean_imap = np.mean(imap)
    imap = imap/mean_imap - 1.
    imap_tapered = (imap*taper + 1)*mean_imap
    kmap = enmap.fft(imap_tapered, normalize="phys", nthread=1) # nthread important when running in parallel

    # flattened inverse power spectrum
    inv_C_power = (1./power)
    inv_C_power = np.nan_to_num(inv_C_power)
    inv_C_power[power == 0.] = 0.
    
    if test:        
        modlmap = kmap.modlmap()
        power = np.nan_to_num(power)
        fmap[(modlmap < 100) & (modlmap > 3000)] = 0.

        map_kSZ = enmap.ifft(kmap * (fmap*inv_C_power), normalize="phys").real
        #map_fltr = (fmap*inv_C_power).real # kSZ filter
        map_fltr = enmap.ifft((fmap*inv_C_power), normalize="phys").real # GNFW filter
        
        print("filtered = ", map_kSZ)
        print("fitlering = ", map_fltr)
        print("power = ", power.min(), power.max(), np.sum(np.isnan(power)))
        print("inverse power = ", np.sum(np.isnan(inv_C_power)), np.min(inv_C_power), np.max(inv_C_power))
        print("C_ell ell^2", power*modlmap**2)
        
        plt.figure(1); plt.title("Power (Fourier)")
        plt.imshow(np.fft.fftshift(power*modlmap**2))

        plt.figure(2); plt.title("L distance (Fourier)")
        plt.imshow(np.fft.fftshift(modlmap))

        
        plt.figure(3); plt.title("Filtering (real)")
        #plt.imshow(np.fft.fftshift((map_fltr)))
        plt.imshow(map_fltr) # GNFW filter
        
        
        plt.figure(4); plt.title("Original map (real, tapered)")
        plt.imshow(imap*taper)
        
        plt.figure(5); plt.title("Filtered map (real, tapered)")
        plt.imshow(map_kSZ)

        plt.show()

    # flatten all Fourier maps
    fmap = fmap.flatten()
    kmap = kmap.flatten()
    inv_C_power = inv_C_power.flatten()
    
    # compute filter C^-1 map / filter C^-1 filter
    #a = np.dot(fmap, np.dot(inv_C_power, kmap))
    #norm = np.dot(fmap, np.dot(inv_C_power, fmap))
    a = np.sum(fmap*inv_C_power*kmap)
    norm = np.sum(fmap*inv_C_power*fmap)
    assert np.sum(np.isnan(a)) + np.sum(np.isnan(norm)) == 0
    a /= norm
    #print("a = ", a)
    a = np.real(a)

    return a


##############################################################################################
###############             David's functions from 1604.01382            #####################
##############################################################################################
from scipy.integrate import quad

def p_M(x, c_500):
    return np.log(1.+c_500*x) - c_500*x*(1.+c_500*x)

def p_p(x, c_500, alpha, beta, gamma):
    p_p = (x*c_500)**gamma*(1.+(x*c_500)**alpha)**((beta-gamma)/alpha)
    p_p = 1./p_p
    return p_p

def dp_pdx(x, c_500, alpha, beta, gamma):
    der = ((c_500*x)**(-gamma)*(1 + (c_500*x)**alpha)**(-(alpha + beta - gamma)/alpha)*(-(c_500*x)**alpha*beta - gamma))/x
    return der

def p_n(x, c_500, alpha, beta, gamma):
    return -x**2*dp_pdx(x, c_500, alpha, beta, gamma)/p_M(x, c_500)

def integrand_num(x_z, x, c_500, alpha, beta, gamma):
    return p_n(np.sqrt(x_z**2+x**2), c_500, alpha, beta, gamma)

def integrand_den(x_r, c_500, alpha, beta, gamma):
    return x_r**2*p_n(x_r, c_500, alpha, beta, gamma)

def g_kSZ(x, c_500, alpha, beta, gamma):
    num, _ = quad(integrand_num, -np.inf, np.inf, args=(x, c_500, alpha, beta, gamma))
    den, _ = quad(integrand_den, 0., 1., args=(c_500, alpha, beta, gamma))
    return num/den

def u_kSZ(theta, theta_500, c_500, alpha, beta, gamma):
    return g_kSZ(theta/theta_500, c_500, alpha, beta, gamma)/(4.*np.pi*theta_500**2)

if __name__ == "__main__":
    # constants from paper (ask theta_500 and c_500)
    theta_500 = 1.8 # arcmin, 2.e13 Msun/h
    #theta_500 = 2.1 # arcmin, 2.e14 Msun/h
    alpha = 1.062
    beta = 5.4807
    gamma = 0.3292
    c_500 = 1.156

    # bins array in arcmin
    thetas = np.linspace(0., 10., 101)
    thetas = 0.5*(thetas[1:] + thetas[:-1])

    # compute u_kSZ
    u_kSZs = np.zeros_like(thetas)
    for i in range(len(thetas)):
        u_kSZs[i] = u_kSZ(thetas[i], theta_500, c_500, alpha, beta, gamma)

    plt.figure(1)
    plt.plot(thetas, u_kSZs)
    plt.xscale('log')
    plt.yscale('log')

    # fourier
    N_dim = len(thetas)
    u_karr = np.fft.fft(u_kSZs)
    karr = np.fft.fftfreq(N_dim, d=10./(2*np.pi*N_dim))
    
    plt.figure(2)
    plt.plot(karr[karr > 0], np.real(u_karr[karr > 0]))
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

