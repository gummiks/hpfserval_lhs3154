from __future__ import print_function
from collections import OrderedDict
import os
import copy
import serval_config
from serval_config import flag, sflag, spline_cv
import numpy as np
from serval import interp, lam2wave, fitspec
from calcspec import barshift, redshift, calcspec, dopshift
import astropy.constants as aconst
import matplotlib.pyplot as plt
import sspectrum
from wstat import wmean, wrms, rms
import read_spec
import cspline as spl
import scipy.signal
import logger
from astropy.convolution import Gaussian1DKernel, convolve
from matplotlib.widgets import SpanSelector
from spectrum_widgets import print_mask
import pandas as pd
import scipy.ndimage.filters
import astropy.io.fits
import filepath
import re
import serval_plotting
import astropy.stats
import scipy.interpolate
import h5py
from scipy.optimize import curve_fit
import airtovac
import resource
import utils
import spec_help
CP = [(0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
      (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
      (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
      (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
      (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
      (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
      (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
      (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
      (0.8, 0.7254901960784313, 0.4549019607843137),
      (0.39215686274509803, 0.7098039215686275, 0.803921568627451)]
c = 299792.4580

def nans(*args, **kwargs):
   return np.nan * np.empty(*args, **kwargs)

def bin_rvs_by_order(rv,e_rv,orders=[3,4,5,6,14,15,16,17,18],plot=True,bjd=None,ax=None,sub_mean=True):
    """
    Bin rv array (28 columns * X number of epochs) errors by orders.
    
    INPUT:
        rv
        e_rv
    
    """
    RV = np.zeros(rv.shape[0])
    E_RV = np.zeros(rv.shape[0])
    for epoch in range(rv.shape[0]):
        RV[epoch], E_RV[epoch] = weighted_average(rv[epoch,orders],e_rv[epoch,orders])
        if sub_mean:
            RV -= np.nanmean(RV)
    if plot:
        nbjd, nRV, e_nRV = serval_plotting.bin_rvs_by_track(bjd,RV,E_RV)
        serval_plotting.plot_RVs(bjd,tRV=RV,tRV_err=E_RV,nbjd=nbjd,nRV=nRV,nRV_err=e_nRV,ax=ax)
    return RV, E_RV

def loop(func,x,**kwargs):
    """
    Loop over x if x is an array
    """
    if np.size(x)>1:
        return np.array([func(i,**kwargs) for i in x])
    else:
        return func(x,**kwargs)

def weighted_average(x,e):
    """
    Calculate weigted average

    INPUT:
        x
        e

    OUTPUT:
        xx: weighted average of x
        ee: associated error
    """
    xx, ww = np.average(x,weights=e**(-2.),returned=True)
    return xx, np.sqrt(1./ww)

def _get_telluric_mask_file(maskfile=serval_config.path_to_tellmask_file,convolve_gauss_width=None):
    """
    Get the telluric mask file for serval

    INPUT:
        The path to the telluric mask file

    OUTPUT:
        mask: telluric mask with wavelengths, and 0, and 1s as columns

    EXAMPLE:
        serval_help.get_telluric_mask()
    """
    inst = serval_config.inst
    mask = np.genfromtxt(maskfile, dtype=None)
    #if 'telluric_mask_atlas_short.dat' in maskfile:
    #    lcorr = 0.000009  # Guillems mask needs this shift of 2.7 km/s
    #    mask[:,0] = airtovac(mask[:,0]) * (1-lcorr)

    #if inst=='HPF':
    #    m2 = np.genfromtxt(serval_config.custom_tellurics_path)
    #    df1 = pd.DataFrame(m1,columns=('wavelength','mask'))
    #    df2 = pd.DataFrame(m2,columns=('wavelength','mask'))
    #    dfc = pd.concat([df1,df2])
    #    dfc.sort_values('wavelength')
    #    mask = dfc.values
    #else:
    #    mask = m1
    #if convolve_gauss_width is not None:
    #    # Doesn't seem to work
    #    print("convolving")
    #    g = Gaussian1DKernel(convolve_gauss_width)
    #    mask[:,0] = convolve(mask[:,0],g,boundary='extend')
    return mask

def _get_sky_mask_file(maskfile=serval_config.path_to_skymask_file):
    """
    Get the sky mask file for serval

    INPUT:
        The path to the sky mask file

    OUTPUT:
        mask: sky mask with wavelengths, and 0, and 1s as columns

    EXAMPLE:
        serval_help._get_sky_mask_file()
    """
    inst = serval_config.inst
    mask = np.genfromtxt(maskfile, dtype=None)
    return mask

def _get_stellar_mask_file(maskfile=serval_config.path_to_stellarmask_file):
    """
    Get the stellar mask file for serval

    INPUT:
        The path to the stellar mask file

    OUTPUT:
        mask: stellar mask with wavelengths, and 0, and 1s as columns

    EXAMPLE:
        serval_help._get_sky_mask_file()
    """
    inst = serval_config.inst
    mask = np.genfromtxt(maskfile, dtype=None)
    return mask

def save_rv_results_as_hdf5(bjd,rv,e_rv,RV,e_RV,orders,target,template_filename='',filename='0_RESULTS/rv_results.hdf5',overwrite=True):
    '''
    Saves RV results as hdf5

    INPUT:
        bjd
        rv
        e_rv
        RV
        e_RV

    OUTPUT:
        saves file as filename

    '''
    try:
        hf = h5py.File(filename,'w')
    except Exception as e:
        if overwrite:
            print('Overwriting')
            hh= h5py.File(filename,'r')
            hh.close()
            hh =h5py.File(filename,'w')
        else:
            print('Skipping, file exist')
            return None

    dt = h5py.special_dtype(vlen=str)
    
    # Flux and ww
    hf.create_dataset('bjd',data=bjd)
    hf.create_dataset('rv',data=rv)
    hf.create_dataset('e_rv',data=e_rv)
    hf.create_dataset('RV',data=RV)
    hf.create_dataset('e_RV',data=e_RV)

    nbjd, nRV, e_nRV = serval_plotting.bin_rvs_by_track(bjd,RV,e_RV)
    hf.create_dataset('nbjd',data=nbjd)
    hf.create_dataset('nRV',data=nRV)
    hf.create_dataset('e_nRV',data=e_nRV)
    hf.create_dataset('sigma_binned',data=np.nanstd(nRV))
    hf.create_dataset('orders',data=orders)
    hf.create_dataset('target',data=target,dtype=dt)
    hf.create_dataset('template_filename',data=template_filename,dtype=dt)

    header_string = 'RV Results for {}\n'.format(target)
    header_string += 'orders = {}\n'.format(orders)
    header_string += 'Binned sigma = {:0.3f} m/s\n'.format(np.nanstd(nRV))
    header_string += 'with template = {}'.format(template_filename)
    print(header_string)
    hf.create_dataset('description',data=header_string,dtype=dt)
    
    hf.close()
    print('Saved to file: {}'.format(filename))

def read_rv_results_from_hdf5(filename='0_RESULTS/rv_results.hdf5'):
    '''
    Reads a hdf5 file that contains rvs

    Returns a hdf5 object. Remember to close.
    '''
    print('Reading file: {}'.format(filename))
    hf = h5py.File(filename,'r')
    print('This file has the following keys')
    print(hf.keys())
    #print(hf['description'])
    return hf

def read_master_results_hdf5(filename):
    """
    """
    print('Reading file: {}'.format(filename))
    hf = h5py.File(filename,'r')
    print()
    print('This file has the following keys')
    print(hf.keys())
    print()
    print(hf['general/description'])
    return hf

def save_master_results_as_hdf5(sps,spt,rv,e_rv,RV,e_RV,orders,vgrid,chi2map,dLW,e_dLW,dLWo,e_dLWo,crx,e_crx,ln_order_center,resfactor,
                         filename='RESULTS/master_results.hdf5',
                         verbose=True,return_hf=False,overwrite=True):
    '''
    Saves results as hdf5. Has three groups:
        general/
        template/
        rv/

    INPUT:
        sps - should be a SSpectrumSet object of all of the spectra including template
        spt - template class
        rv  - full matrix of RVs 28 by num epochs. In m/s
        e_rv- same but errors
        RV  - 1D RVs averaged over all orders in m/s
        e_RV- 1D RV errors

    OUTPUT:
        hf
        
    EXAMPLE:
        serval_help.read_master_results_hdf5('RESULTS/master_results.hdf5')
    '''
    try:
        hf = h5py.File(filename,'w')
    except Exception as e:
        if overwrite:
            print('Overwriting')
            hh= h5py.File(filename,'r')
            hh.close()
            hh =h5py.File(filename,'w')
        else:
            print('Skipping, file exist')
            return None
    dt = h5py.special_dtype(vlen=str)

    ##############################################################
    # Groups
    h_tp = hf.create_group('template')
    h_ge = hf.create_group('general')
    h_rv = hf.create_group('rv')
    
    ###########################################################
    # TEMPLATE
    h_tp.create_dataset('ff',data=spt.ff)
    h_tp.create_dataset('ww',data=spt.ww)
    h_tp.create_dataset('tstack_f',data=spt.tstack_f)
    h_tp.create_dataset('tstack_f_flat',data=spt.tstack_f_flat)
    h_tp.create_dataset('tstack_w',data=spt.tstack_w)
    h_tp.create_dataset('tstack_w_flat',data=spt.tstack_w_flat)
    h_tp.create_dataset('tstack_ind',data=spt.tstack_ind)
    h_tp.create_dataset('tstack_ind0',data=spt.tstack_ind0)
    h_tp.create_dataset('tstack_tellind',data=spt.tstack_tellind)
    h_tp.create_dataset('tstack_starind',data=spt.tstack_starind)
    #print('SPT header',spt.S.header)
    h_tp.create_dataset('header',data=str(spt.S.header))
    
    ###########################################################
    # RVs
    h_rv.create_dataset('berv',data=sps.df.berv.values)
    h_rv.create_dataset('bjd',data=sps.df.bjd.values)
    h_rv.create_dataset('sn5',data=sps.df.sn5.values)
    h_rv.create_dataset('spectrum_basenames',data=sps.df.basename.values,dtype=dt)
    h_rv.create_dataset('sn18',data=sps.df.sn18.values)
    h_rv.create_dataset('flag',data=sps.df.flag.values)
    h_rv.create_dataset('exptime',data=sps.df.exptime.values)
    h_rv.create_dataset('rv',data=rv)
    h_rv.create_dataset('e_rv',data=e_rv)
    h_rv.create_dataset('RV',data=RV)
    h_rv.create_dataset('e_RV',data=e_RV)
    h_rv.create_dataset('path_flat',data=sps.df.path_flat.values,dtype=dt)
    h_rv.create_dataset('istemplate',data=sps.df.istemplate.values)
    h_rv.create_dataset('gitcommit',data=sps.df.gitcommit.values,dtype=dt)
    h_rv.create_dataset('vgrid',data=vgrid)
    h_rv.create_dataset('chi2map',data=chi2map)
    h_rv.create_dataset('dLW',data=dLW)
    h_rv.create_dataset('e_dLW',data=e_dLW)
    h_rv.create_dataset('dLWo',data=dLWo)
    h_rv.create_dataset('e_dLWo',data=e_dLWo)
    h_rv.create_dataset('crx',data=crx)
    h_rv.create_dataset('e_crx',data=e_crx)
    h_rv.create_dataset('ln_order_center',data=ln_order_center)
    h_rv.create_dataset('resfactor',data=resfactor)

    nbjd, nRV, e_nRV = serval_plotting.bin_rvs_by_track(sps.df.bjd.values,RV,e_RV)
    h_rv.create_dataset('nbjd',data=nbjd)
    h_rv.create_dataset('nRV',data=nRV)
    h_rv.create_dataset('e_nRV',data=e_nRV)
    h_rv.create_dataset('sigma_binned',data=np.nanstd(nRV))

    ##############################################################
    # General
    h_ge.create_dataset('sky_subtracted',data=spt.S.sky_subtract)
    h_ge.create_dataset('sky_subtract_factor',data=spt.S.SKY_SCALING_FACTOR)
    h_ge.create_dataset('orders',data=orders)
    h_ge.create_dataset('target',data=spt.obj,dtype=dt)
    h_ge.create_dataset('numspectra',data=len(sps.splist))
    h_ge.create_dataset('path_telluric_mask',data=spt.S.path_telluric_mask,dtype=dt)
    h_ge.create_dataset('path_sky_mask',data=spt.S.path_sky_mask,dtype=dt)
    h_ge.create_dataset('path_stellar_mask',data=spt.S.path_stellar_mask,dtype=dt)
    h_ge.create_dataset('ra_deg',data=spt.target.ra)
    h_ge.create_dataset('de_deg',data=spt.target.dec)
    h_ge.create_dataset('plx',data=spt.target.px)
    h_ge.create_dataset('rv_systematic',data=spt.target.rv)
    h_ge.create_dataset('pmra',data=spt.target.pmra)
    h_ge.create_dataset('pmde',data=spt.target.pmdec)
    unique_flats  = pd.Series(sps.df.path_flat.values).unique()
    h_ge.create_dataset('path_flat_unique',data=unique_flats,dtype=dt)
    header_string = 'RV Results for {}\n'.format(spt.obj)
    header_string += 'Orders = {}\n'.format(orders)
    header_string += 'Num spectra = {}\n'.format(len(sps.splist))
    header_string += 'Binned sigma = {:0.3f} m/s\n'.format(np.nanstd(nRV))
    header_string += 'Num uniqueflats = {}\n'.format(len(unique_flats))
    for i, fl in enumerate(unique_flats):
        header_string += 'Flat{} = {}\n'.format(i,fl)
    header_string += 'Telluric mask = {}\n'.format(spt.S.path_telluric_mask)
    header_string += 'Sky mask = {}\n'.format(spt.S.path_sky_mask)
    header_string += 'Stellar mask = {}\n'.format(spt.S.path_stellar_mask)
    header_string += 'Sky subtracted {} by {}\n'.format(spt.S.sky_subtract,spt.S.SKY_SCALING_FACTOR)
    h_ge.create_dataset('description',data=header_string,dtype=dt)
    if verbose: print(header_string)

    ##############################################################
    # Flats ?
    ##############################################################
    if verbose: print('Saved to file: {}'.format(filename))
    if return_hf:
        return hf
    else:
        hf.close()
        return None


def save_template_as_hdf5(spt,sps,filename='TEMPLATES/20180716_gj699_all_orders_flat_abcd.hdf5'):
    '''
    Saves an spt template file to a hdf5 
    '''
    hf = h5py.File(filename,'a')
    dt = h5py.special_dtype(vlen=str)
    
    # Flux and ww
    hf.create_dataset('ff',data=spt.ff)
    hf.create_dataset('ww',data=spt.ww)
    
    # stacks
    hf.create_dataset('tstack_f',data=spt.tstack_f)
    hf.create_dataset('tstack_w',data=spt.tstack_w)
    hf.create_dataset('tstack_ind',data=spt.tstack_ind)
    hf.create_dataset('tstack_ind0',data=spt.tstack_ind0)
    hf.create_dataset('tstack_tellind',data=spt.tstack_tellind)
    hf.create_dataset('tstack_starind',data=spt.tstack_starind)
    hf.create_dataset('header',data=spt.S.header)
    
    # Spall
    hf.create_dataset('berv',data=sps.df.berv.values)
    hf.create_dataset('bjd',data=sps.df.bjd.values)
    hf.create_dataset('basename',data=sps.df.basename.values,dtype=dt)
    hf.create_dataset('sn5',data=sps.df.sn5.values)
    hf.create_dataset('sn18',data=sps.df.sn18.values)
    hf.create_dataset('flag',data=sps.df.flag.values)
    hf.create_dataset('exptime',data=sps.df.exptime.values)
    
    hf.close()
    print('Saved to file: {}'.format(filename))
    
def read_template_from_hdf5(filename='TEMPLATES/20180716_gj699_all_orders_flat_abcd.hdf5',method='pandas'):
    '''
    Read 

    Returns a hdf5 objet. Remember to close.
    '''
    print('Reading file: {}'.format(filename))
    hf = h5py.File(filename,'r')
    return hf

def telluric_mask_interpolate(xx,maskfile=serval_config.path_to_tellmask_file):
    """
    Interpolate over telluric mask for a given (finer spacing) xx array

    INPUT:
        xx: wavelengths to intepolate over

    OUTPUT:
        yy: telluric_mask interpolated

    EXAMPLE:
        mask = serval_help._get_telluric_mask_file()
        xx = np.linspace(mask[0,0],mask[-1,0],40000)
        yy = serval_help.telluric_mask_interpolate(xx)
        plt.plot(mask[:,0],mask[:,1])
        plt.plot(xx,yy)
    """
    mask = _get_telluric_mask_file(maskfile)
    tellmask = interp(lam2wave(mask[:,0],wlog=False), mask[:,1])
    return tellmask(xx)

def mask_tellurics_sky(w,f,return_badmask=False):
    """
    Mask tellurics and sky from wavelengths and flux.

    INPUT:
        w
        f
        return_badmask: return mask where it is TRUE if it is a telluric/sky
    
    EXAMPLE:
        wm, fm, m = mask_tellurics_sky(w,f)
    """
    m_sky  = sky_mask_interpolate(w)>0.01
    m_tell = telluric_mask_interpolate(w)>0.01
    m = m_sky | m_tell
    if return_badmask:
        return w[~m],f[~m], m
    else:
        return w[~m],f[~m]

def bpmap_interpolate(xx,x,bpmap):
    """
    Interpolate bad pixel map in pixel space

    INPUT:
        xx - the points to evaluate the bpmap at (should have len = serval.config_osize
        x -  the x points to define the interpolation, pixels from 0 to 2048 for HPF 
        bpmap - the bad pixel map to interpolate over should have len=2048 for HPF

    OUTPUT:
        Interpolated bad pixel map

    EXAMPLE:
        ww, kk, ff, ee, bb = spt.interpolate_and_baryshift_order(o=17)
        x = np.arange(2048)
        xx = np.linspace(0,2048,serval_config.osize)
        plt.plot(x,spt.S.bpmap[17,:])
        plt.plot(xx,bb,'k.')
    """
    interpolator = scipy.interpolate.interp1d(x,bpmap)
    return interpolator(xx)

def sky_mask_interpolate(xx,maskfile=serval_config.path_to_skymask_file):
    """
    Interpolate over sky mask for a given (finer spacing) xx array

    INPUT:
        xx: wavelengths to intepolate over

    OUTPUT:
        yy: sky_mask interpolated

    EXAMPLE:
        mask = serval_help._get_sky_mask_file()
        xx = np.linspace(mask[0,0],mask[-1,0],40000)
        yy = serval_help.sky_mask_interpolate(xx)
        plt.plot(mask[:,0],mask[:,1])
        plt.plot(xx,yy)
    """
    mask = _get_sky_mask_file(maskfile)
    skymask = interp(lam2wave(mask[:,0],wlog=False), mask[:,1])
    return skymask(xx)

def stellar_mask_interpolate(xx,maskfile=serval_config.path_to_stellarmask_file):
    """
    Interpolate over sky mask for a given (finer spacing) xx array

    INPUT:
        xx: wavelengths to intepolate over

    OUTPUT:
        yy: sky_mask interpolated

    EXAMPLE:
        mask = serval_help._get_stellar_mask_file()
        xx = np.linspace(mask[0,0],mask[-1,0],40000)
        yy = serval_help.stellar_mask_interpolate(xx)
        plt.plot(mask[:,0],mask[:,1])
        plt.plot(xx,yy)
    """
    mask = _get_stellar_mask_file(maskfile)
    stellarmask = interp(lam2wave(mask[:,0],wlog=False), mask[:,1])
    return stellarmask(xx)

def median_sigma_clip(y,window=49,sigma_upper=15,sigma=15):
    yy = np.copy(y)
    _y = y-scipy.signal.medfilt(yy,window)
    data = astropy.stats.sigma_clipping.sigma_clip(_y,sigma_upper=sigma_upper,sigma=sigma)
    yy[data.mask] = np.nan
    return yy, data.mask

def read_spectra(files,targetname,inst,read_data=False,verbose=True,jd_utcs=None,spi=None):
    """
    Read all spectra, and return a list of spectra, and the highest signal to noise spectrum
    
    INPUT:
        files: .tar files for HARPS spectra
        targetname: SIMBAD resolvable target name
        read_data: if true, then also read in the data (slow if looping over many files)
        
    OUTPUT:
        splist: list of all spectra (assumes all good)
        spi: number of highest SNR spectrum
        SN55Best: SNR
    
    EXAMPLE:
        dir_or_inputlist = "/Users/gks/Dropbox/mypylib/notebooks/GIT/HARPS/gj699/"
        files = sorted(glob.glob(dir_or_inputlist+os.sep+pat))
        target = Targ('GJ699',cvs="test.csv")
        splist, spi, snbest = read_spectra(files,target,inst="HARPS"):

    NOTES:
    """
    log = logger.logger()
    splist = []
    if spi is not None:
        overwrite_spi = True
        print('Overwriting SPI={}'.format(spi))
    else:
        overwrite_spi = False
    SN55best = 0.
    for n,filename in enumerate(files):   # scanning fitsheader
        if verbose: log.info(n)
        if jd_utcs is None: 
            jd_utc = None
        else:
            jd_utc = jd_utcs[n]
        sp = sspectrum.SSpectrum(filename, targetname=targetname,read_data=read_data,inst=inst,jd_utc=jd_utc)
        splist.append(sp)
        #sp.S.sa = sp.target.sa / 365.25 * (sp.S.bjd-splist[0].S.bjd)
        if sp.S.sn55 < serval_config.snmin: sp.S.flag |= sflag.lowSN
        if sp.S.sn55 > serval_config.snmax: 
            if serval_config.simulation_mode == False:
                sp.S.flag |= sflag.hiSN
        if not sp.S.flag:
            if SN55best < sp.S.sn55 < serval_config.snmax:
                SN55best = sp.S.sn55
                if overwrite_spi is False:
                    spi = n
        if spi is None and overwrite_spi is False:
            print('Warning spi is None, setting as 0')
            spi = 0
    spall = sspectrum.SSpectrumSet(splist)
    spok = sspectrum.SSpectrumSet(np.delete(splist,spi))
    spt = splist[spi]
    return spall, spok, spt, spi


def polyscalecomp2refspecflux_for_order(sp,spt,o,plot=True,use_python_version=False,interp_badpm=True,ax=None,vtfix=False,deg=3,verbose=True):
    """
    A function to polyscale a comparison spectrum flux to the flux of a reference spectrum. RV shifts and baryshifts. Very similar to calculate_pre_rv_for_order

    INPUT:
    sp  - comparison spectrum, this spectrum will be scaled
    spt - reference spectrum. Flux will be scaled to this value
    o - order
    plot - if true, plot a comparison plot

    OUTPUT:
    w1 - wavelength of reference spectrum baryshifted
    f1 - unchanged flux of reference spectrum
    w2 - wavelength of comparison spectrum baryshifted and rv shifted
    f2*poly - scaled flux of comparison spectrum
    _rv - rv in m/s
    _e_rv - error on rv in m/s

    EXAMPLE:
    """
    if use_python_version:
        print("USING PYTHON VERSION")
    # Interpolate & resample template and baryshift
    ww,kk,ff,ee,bb = spt.interpolate_and_baryshift_order(o=o,interp_badpm=interp_badpm)
    # Interpolate telluric mask for comparison spectrum
    sp.S.bpmap[o][telluric_mask_interpolate(sp.S.w[o])>0.01] = flag.atm
    # Interpolate telluric mask for comparison spectrum
    sp.S.bpmap[o][sky_mask_interpolate(sp.S.w[o])>0.01] = flag.sky
    # Interpolate telluric mask for template
    tellind = telluric_mask_interpolate(barshift(ww, -sp.S.berv)) > 0.01
    # Interpolate telluric mask for template
    skyind = sky_mask_interpolate(barshift(ww, -sp.S.berv)) > 0.01
    # Only use indices where there are no tellurics
    pind, = np.where((bb == 0) & (tellind == 0) & (skyind == 0))
    #print('lenpind=',len(pind))
    # Baryshift comparison spectrum
    w2 = barshift(sp.S.w[o],sp.S.berv)
    # Define bad-pixelmap (currently only taking out tellurics)
    b2 = sp.S.bpmap[o] 
    f2 = sp.S.f[o]
    # Only use indices without tellurics and that are finite
    idx = np.where( (b2==0) & (np.isfinite(w2)) & (np.isfinite(f2)))
    # Calculate knots on good values
    k2 = spline_cv(w2[idx],f2[idx])
    #par,fmod,keep,stat,ssr = serval_help.calculate_pre_rv_for_order(w2,f2,k2,ww,ff,ee,bb,pind,0.)
    par,fmod,keep,stat,ssr = fitspec(w2, f2, k2, ww, ff, e_y=ee,
                                     v=0., vfix=vtfix, keep=pind, deg=deg,chi2map=True,plot=False,use_python_version=use_python_version)
    poly = calcspec(w2, *par.params, retpoly=True)
    _rv = par.params[0]*1000.
    _e_rv = par.perror[0] * stat['std'] * 1000
    print("RV = {:0.4f} +/- {:0.4f} m/s ".format(_rv,_e_rv))
    # Get reference spectrum in the same frame and number of points in the array as the reference spectrum
    # could also just use ww, and ff, which are the 4x resampled versions of this
    w1,f1,e1,s1 = spt.get_order(o,baryshifted=True)
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        #ax.plot(w1,f1,label='Reference spectrum')
        ax.plot(ww,ff,label='Reference spectrum,ww,ff - ref')
        #ax.plot(ww,ff,label='Reference spectrum')
        #ax.plot(ww,fmod,label='Reference bspline')
        ax.plot(w2,f2*poly,label='Comparison spectrum*poly - transit')
        ax.plot(w2[b2==flag.atm],(f2*poly)[b2==flag.atm],label='Comp: Tellurics',color='firebrick',marker='o',lw=0,markersize=5)
        ax.plot(w2[b2==flag.sky],(f2*poly)[b2==flag.sky],label='Comp: sky',color='purple',marker='D',lw=0,markersize=5)
        ax.legend(loc='upper left',fontsize=12)
        ax.grid(alpha=0.5,lw=0.3)
        title = "Comparison template*{:0.3f} o = {}, RV = {:0.4f} +/- {:0.4f} m/s ".format(np.mean(poly),o,_rv,_e_rv)
        ax.set_title(title,fontsize=12)
        ax.set_xlabel("Wavelength [A]")
        ax.set_ylabel("Flux")
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.fill_between(barshift(spt.telluric_mask[:,0],spt.S.berv),spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="k")
        ax.fill_between(barshift(spt.sky_mask[:,0],spt.S.berv),spt.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
        ax.set_xlim(xlim)
        ax.tick_params(axis='both',labelsize=10,pad=5)
        ax.minorticks_on()
    return w1,f1,w2,f2*poly,_rv,_e_rv

def calculate_pre_rv_for_order(sp,spt,o,plot=True,use_python_version=False,vtfix=False,deg=3,verbose=True,return_res_dict=False,interp_badpm=True,ax=None,bx=None,cx=None):
    """
    INPUT:
    - sp: reference spectrum
    - spt: template
    - o: order
    
    OUTPUT:
    - par:
    - fmod:
    - keep:
    - stat:
    - ssr:
    
    NOTES:
    - Does tellurics properly
    
    EXAMPLE:
        sp = splist[1]
        spt = splist[36]
        par, fmod, keep, stat, ssr = calculate_pre_rv_for_order(sp,spt,o=12)
    """
    if use_python_version:
        print("USING PYTHON VERSION")
    # Read in data if not present in class
    #if sp.S.w is None: sp.S.read_data()
    #if spt.S.w is None: spt.S.read_data()
        
    # Interpolate & resample template and baryshift
    ww,kk,ff,ee,bb = spt.interpolate_and_baryshift_order(o=o,interp_badpm=interp_badpm)
    # Interpolate telluric mask for comparison spectrum
    sp.S.bpmap[o][telluric_mask_interpolate(sp.S.w[o])>0.01] = flag.atm
    # Interpolate telluric mask for comparison spectrum
    sp.S.bpmap[o][sky_mask_interpolate(sp.S.w[o])>0.01] = flag.sky
    # Interpolate telluric mask for template
    tellind = telluric_mask_interpolate(barshift(ww, -sp.S.berv)) > 0.01
    # Interpolate telluric mask for template
    skyind = sky_mask_interpolate(barshift(ww, -sp.S.berv)) > 0.01
    # Only use indices where there are no tellurics
    pind, = np.where((bb == 0) & (tellind == 0) & (skyind == 0))
    #print('lenpind=',len(pind))
    # Baryshift comparison spectrum
    w2 = barshift(sp.S.w[o],sp.S.berv)
    # Define bad-pixelmap (currently only taking out tellurics)
    b2 = sp.S.bpmap[o] 
    f2 = sp.S.f[o]
    # Only use indices without tellurics and that are finite
    idx = np.where( (b2==0) & (np.isfinite(w2)) & (np.isfinite(f2)))
    # Calculate knots on good values
    k2 = spline_cv(w2[idx],f2[idx])
    #par,fmod,keep,stat,ssr = serval_help.calculate_pre_rv_for_order(w2,f2,k2,ww,ff,ee,bb,pind,0.)
    par,fmod,keep,stat,ssr = fitspec(w2, f2, k2, ww, ff, e_y=ee,
                                     v=0., vfix=vtfix, keep=pind, deg=deg,chi2map=True,plot=False,use_python_version=use_python_version)
    _rv = par.params[0]*1000.
    _e_rv = par.perror[0] * stat['std'] * 1000
    #if verbose:
    #    #print("Calculated RV={:.5f}km/s, expected RV=km/s".format(par.params[0]))
    #    print(_rv,"+/-",_e_rv)    
    #    print("len(pind)",len(pind))
    #    print("np.sum(tellind)",np.sum(tellind==0))
    #    print("np.sum(b2)",np.sum(b2==0))
    #    #print("np.sum(skyind)",np.sum(skyind==0))
    #    print("np.sum(bb==0)",np.sum(bb==0))
    #    print(np.where(b2!=0))
    #    print(len(w2))
    #    print(len(b2))
    if plot:
        if ax is None:
            fig = plt.figure()
            ax = [fig.add_subplot(121),fig.add_subplot(122)]
            #fig, ax = plt.subplots(ncols=2,figsize=(15,7))
        if bx is None:
            bx = ax[1].twinx()
        if cx is None:
            cx = ax[1].twiny()
        # SSR
        ax[0].plot(par.ssr[0],par.ssr[1])
        ax[0].set_xlabel("Velocity [km/s]")
        ax[0].set_ylabel("SSR (chi^2)")
        ax[0].minorticks_on()
        ax[0].grid(alpha=0.5,lw=0.3)
        title = "o = {}, RV = {:0.4f} +/- {:0.4f} m/s ".format(o,_rv,_e_rv)
        ax[0].set_title(title,fontsize=12)

        # Also plot pixel positions on x axis
        cx.plot(np.arange(len(f2))[b2==0],f2[b2==0],label="Comparison: shifted (w2[b2==0],f2[b2==0])",lw=0)#dummy plot

        ax[1].plot(w2[b2==0],f2[b2==0],label="Comparison: shifted (w2[b2==0],f2[b2==0])")
        #ax.flat[1].plot(w2,f2,label="Comparison: shifted (w2,f2)")
        ax[1].plot(ww,ff,color="orange",label="Template: Interpolated spectrum with masked pix",alpha=0.7)
        ax[1].plot(ww[bb==0],ff[bb==0],color="green",label="Template: Interpolated spectrum without masked pix",alpha=0.8)
        ax[1].plot(ww,fmod,label="ww,fmod",alpha=0.5,color='purple')

        bx.plot(ww[keep],ff[keep]-fmod[keep],label="Residuals (ww,ff-fmod)",alpha=0.5,color="red")
        bx.legend(loc='upper right',fontsize=12)
        bx.margins(y=0.8)
        ylim = bx.get_ylim()
        bx.set_ylim(ylim[0]*0.5,ylim[1]*4.)
        ax[1].legend(loc="upper left",fontsize=12)
        ax[1].set_xlabel("Wavelength [A]")
        ax[1].set_ylabel("Flux")
        xlim = ax[1].get_xlim()
        ylim = ax[1].get_ylim()
        ax[1].fill_between(barshift(spt.telluric_mask[:,0],spt.S.berv),spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="k")
        ax[1].fill_between(barshift(spt.sky_mask[:,0],spt.S.berv),spt.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
        ax[1].set_xlim(xlim)
        ax[1].tick_params(axis='both',labelsize=10,pad=5)
        bx.tick_params(axis='both',labelsize=10,pad=5)
        cx.tick_params(axis='both',labelsize=10,pad=5)
        ax[1].minorticks_on()
        bx.minorticks_on()
        cx.minorticks_on()

    if return_res_dict:
        res = {"v":par.ssr[0],
               "chi": par.ssr[1],
                "w2": w2,
               "f2": f2,
               "ww[o]": ww,
               "ff[o]": ff,
               "ww_b":ww[bb==0],
               "ff_b":ff[bb==0],
               "ww_res":ww[keep],
               "ff_res":ff[keep]-fmod[keep]
               }
        return res
    return par, fmod, keep, stat, ssr

def calculate_pre_RVs(splist,spt,orders=None,verbose=True,use_python_version=False,plot=False,deg=3,interp_badpm=False):
    """
    Calculate pre-RVs
    
    INPUT:
    - splist: 
    - spt: template spectrum, after running interpolate_and_baryshift_orders()
    
    OUTPUT:
    - rv: 
    - e_rv:
    - RV:
    - e_RV:
    
    EXAMPLE:
        # Choosing highest SN spectrum as template
        spt_filename = files[spi]
        spt = sspectrum.SSpectrum(spt_filename)
    """
    nspec = len(splist)
    rv, e_rv = nans((2, nspec, serval_config.iomax))
    RV, e_RV = nans((2, nspec))
    bjds = nans(nspec)
    msksky = serval_config.msksky # bunch of 0s
    if orders is None: orders = serval_config.orders
    for i,sp in enumerate(splist):
        #################
        # Loop over files
        #################
        sp.S.read_data()
        if sp.S.flag: 
            print("\nNot using flagged spectrum:", i, sp.filename, 'flag:', sp.S.flag)
            pass
        for o in orders:
            ##################
            # Loop over orders
            ##################
            # Calculate RVs
            par, fmod, keep, stat, ssr = calculate_pre_rv_for_order(sp,spt,o=o,
                                                                    use_python_version=use_python_version,
                                                                    plot=plot,
                                                                    verbose=verbose,
                                                                    deg=deg,
                                                                    interp_badpm=interp_badpm)
            # Save RVs
            rv[i,o] = rvo = par.params[0]*1000.
            e_rv[i,o] = par.perror[0] * stat['std'] * 1000 

            #print("self.S.bpmap[o,:]==0 {}/{}".format(np.sum(self.S.bpmap[o,:]==0),len(self.S.bpmap[o,:])))
            if verbose: 
                print("%s-%02u %s rv=%7.2fm/s STD=%.2f SNR=%5.1f #it=%s keep=%s/%s bp=%s/%s" % 
                                  (i+1, o, sp.S.timeid, rvo, stat['std'], stat['snr'], par.niter,len(keep),len(fmod),np.sum(sp.S.bpmap[o,:]==0),len(sp.S.bpmap[o,:])))
        ind = e_rv[i] > 0.
        #RV[i], e_RV[i] = wsem(rv[i,ind], e=e_rv[i,ind])  # weight
        RV[i], e_RV[i] = weighted_average(rv[i,ind], e=e_rv[i,ind])  # weight
        #RV[i], e_RV[i] = np.nanmean(rv[i,ind]), np.nanstd(e_rv[i,ind])  # weight
        #RV[i], e_RV[i] = np.nanmean(rv[i,ind]), 1.  # weight
        #RV[i], e_RV[i] = np.nanmedian(rv[i,ind]), 1.  # weight
        bjds[i] = sp.S.bjd
        #if verbose: print('%s/%s'%(i+1,nspec), sp.S.bjd, sp.S.timeid, ' preRV =', RV[i], e_RV[i])

    return bjds, rv, e_rv, RV, e_RV

def generate_template(spt,sps,RV,orders=None,verbose=True,template_filename="",plot=False,inst="HPF",
                      plot_baryshifted=False,select_tellurics=True,estimate_optimal_knots=True,vref=0.):
    """
    Generate master template by coadding spectra
    
    INPUT:
    - spt: highest signal to noise spectrum (sspectrum.SSpectrum)
    - sps: a spslist 
    - RV: pre-RVs
    
    OUTPUT:
    - spt: final templat
    
    EXAMPLE:
    """
    print(inst)
    log = logger.logger()
    nk = int(serval_config.osize/4 * serval_config.ofac)
    npix = len(spt.S.w[0,:])
    splist = sps.splist
    #splist = sps.splist[0:2]
    #print('HACK, ONLY USING FIRST 2 SPECTRA FOR GENERATING TEMPLATE')
    ntset = len(splist)
    wmod = np.zeros((ntset,npix)) # stores the coadded wavelengths
    mod = np.zeros((ntset,npix))  # stores the coadded flux
    emod = np.zeros((ntset,npix)) # error on the coadded flux
    bmod = np.zeros((ntset,npix), dtype=int)
    if orders is None: 
        orders = serval_config.orders
    print("Using orders:",orders)
    
    # =============================
    # =============================
    # Loop over orders

    spt.tstack_w       = np.zeros((len(serval_config.orders),ntset,npix))
    spt.tstack_f       = np.zeros((len(serval_config.orders),ntset,npix))
    spt.tstack_w_flat  = np.ones((len(serval_config.orders),ntset,npix))*np.nan#OrderedDict()
    spt.tstack_f_flat  = np.ones((len(serval_config.orders),ntset,npix))*np.nan#OrderedDict()
    spt.tstack_ind     = np.zeros((len(serval_config.orders),ntset,npix),dtype=bool)
    spt.tstack_tellind = np.zeros((len(serval_config.orders),ntset,npix),dtype=bool)
    spt.tstack_ind0    = np.zeros((len(serval_config.orders),ntset,npix),dtype=bool)
    spt.tstack_starind = np.zeros((len(serval_config.orders),ntset,npix),dtype=bool)
    spt.template       = True

    for o in orders:
        # =============================
        # Loop over spectra
        #for i,sp in enumerate(splist):
        for i,sp in enumerate(splist):
            if sp.S.flag:
                print(i,"Bad spectrum! flag=1 Skipping!")
            if not sp.S.flag:
                # GENERATING TEMPLATE
                print("{}-{}: ".format(o,i),end=" ")
                sp = sp.S.get_data(pfits=2, orders=o)

                if inst=="HPF":
                    bmod[i] = sp.bpmap 
                elif inst=="HARPS":
                    bmod[i] = sp.bpmap | serval_config.msksky[o]
                else:
                    print("inst = HPF or HARPS")
                    sys.exit()

                # telluric mask
                bmod[i][telluric_mask_interpolate(sp.w)>0.01] |= flag.atm
                # skymasking
                bmod[i][sky_mask_interpolate(sp.w)>0.01] |= flag.sky

                # NOTE 20201021 VREF
                w2 = redshift(sp.w, vo=sp.berv, ve=vref-RV[i]/1000.)
                #w2 = redshift(sp.w, vo=sp.berv, ve=-RV[i]/1000.)
                #################################################################
                # ENABLE THIS TO ACTUALLY MASK THE STELLAR LINES IN THE TEMPLATE
                #bmod[i][stellar_mask_interpolate(w2)>0.01] |= flag.star
                #################################################################
                #tellind = telluric_mask_interpolate(barshift(ww, -sp.S.berv)) > 0.01
                #pind, = np.where((bpmod == 0) & (tellind == 0))
                i0 = np.searchsorted(w2, spt.ww[o].min()) - 1   # w2 must be oversized
                ie = np.searchsorted(w2, spt.ww[o].max())
                pind, = np.where(bmod[i][i0:ie] == 0)
                bmod[i][:i0] = flag.out #outside masking
                bmod[i][ie:] = flag.out #outside masking
                if not len(pind):
                    print('no valid points in n=%s, o=%s, RV=%s; skipping order' % (i, o, RV[i]))
                    continue
                # fmod has all of the data, i.e., would need to do 
                # has the same length as w2[i0:ie], 1948
                # keep has length e.g., 1780
                # Keep also masks the sigma clipped values
                par, fmod, keep, stat = fitspec(spt.ww[o], spt.ff[o], spt.kk[o], 
                                                w2[i0:ie], sp.f[i0:ie], sp.e[i0:ie], v=0, vfix=True, 
                                                keep=pind, v_step=False, clip=serval_config.kapsig,
                                                nclip=serval_config.nclip, deg=serval_config.deg)
                poly = calcspec(w2, *par.params, retpoly=True)
                # ATTENTION: devide mod[i] with polynomial
                wmod[i] = w2
                mod[i] = sp.f / poly   # be careful if  poly<0
                emod[i] = sp.e / poly  # is this correct ? 
        ind = (bmod&(flag.nan+flag.neg+flag.out)) == 0 # not valid
        tellind = (bmod&(flag.atm+flag.sky)) > 0
        starind = (bmod&flag.star) > 0
        ind *= emod > 0.0
        we = 0*mod
        
        # Weights
        we[ind] = 1. / emod[ind]**2
        # Why are these 10x lower, does this just mean that we are not perfectly masking out the tellurics ?
        # Just making them less important ?
        # Maybe we only do that for the template, but we don't use them in the RVs ?
        telluric_downweight = serval_config.telluric_downweight_factor
        we[tellind] = telluric_downweight / (ntset * (emod[tellind]**2 + np.median(emod[ind])**2))
        we[starind] = telluric_downweight / (ntset * (emod[starind]**2 + np.median(emod[ind])**2))
        ind0 = ind*1

        #CLIPPING
        niter = serval_config.niter
        for it in range(niter+1):   # clip 5 sigma outliers
            ind2 = spt.ww[o] < wmod[ind].max()  # but then the complement for ind2 in ff is from spt!!!
                                           # maybe extrapolation better
            # B-spline fit for co-adding
            ymod, smod = spl._ucbspl_fit(wmod[ind], mod[ind], we[ind], K=nk, lam=serval_config.pspllam,
                                         sigk=True, reta=True)

            # yfit is the main 
            yfit = smod(spt.ww[o][ind2])
            wko = smod.xk()   # the knot positions
            fko = smod()      # the knot values
            eko = smod.sigk   # the error estimates for knot values
            dko = smod.dk()   # ~second derivative at knots

            ######################
            # RESIDUALS
            # normalised residuals
            # including telluric and original values gives conservative limits
            res = (mod[ind]-ymod) / emod[ind]
            # original values gives conservative limits
            #res = (mod[ind]-ymod) * np.sqrt(we[ind])
            #sig = std(res)
            sig = np.std(res[~tellind[ind]])
            #print("sig={} end={}".format(res,sig))
            # iqr(res, sig=True) untested, will be slower (in normal mode) than std but fewer iterations
            #gplot(wmod[ind], res,', %s lt 3, %s lt 3, %s lt 2, %s lt 2' %(-sig,sig,-ckappa[0]*sig, ckappa[1]*sig))

            if np.isnan(sig):
                msg ='nan err_values in coadding. This may happen when data have gaps e.g. due masking or bad pixel flaging. Try the -pspline option'
                print(msg)

            G, kkk = spl._cbspline_Bk(wmod[ind], int(nk/5))
            chik = np.zeros(int(nk/5) + 3)   # chi2 per knot
            normk = np.zeros(int(nk/5) + 3)  # normalising factor to compute local chi2_red
            for i,kki in enumerate(kkk):
                normk[kki:kki+4] += G[i]
                chik[kki:kki+4] += res[i]**2 * G[i]

            vara = spl.ucbspl(chik/normk, wmod[ind].min(), wmod[ind].max())
            sig = np.sqrt(vara(wmod[ind]))
            #print("sig= {}".format(sig))

            okmap = np.array([True] * len(res))
            if serval_config.ckappa[0]: okmap *= res > -serval_config.ckappa[0]*sig
            if serval_config.ckappa[1]: okmap *= res < serval_config.ckappa[1]*sig
            okmap[tellind[ind]] = True # Oh my god. Do not reject the tellurics based on emod. That likely gives gaps and then nans.

            print("SIG=%.5f (number of clipped points = %d)" % (np.nanmedian(sig), np.sum(~okmap)))
            if it < niter:
                ind[ind] *=  okmap

        if estimate_optimal_knots:
            basename = template_filename.split('.')[0]
            savename_knotplot = basename + '_knots_o{}.png'.format(o)
            print('#######################################')
            print('ESTIMATING OPTIMAL KNOTS')
            print('#######################################')
            print('Current K value: {}'.format(nk))
            # BIC to get optimal knot spacing (smoothing)
            chired = []
            BIC = []
            #K = np.logspace(np.log10(10), np.log10(serval_config.ntpix*serval_config.knotoptmult), dtype=int)
            K = np.logspace(np.log10(100), np.log10(serval_config.ntpix*serval_config.knotoptmult),10, dtype=int)
            for i,Ki in enumerate(K):
               ymod, smod = spl._ucbspl_fit(wmod[ind], mod[ind], we[ind], K=Ki, lam=serval_config.pspllam, sigk=True, reta=True)
               #smod, ymod = spl.ucbspl_fit(wmod[ind], mod[ind], we[ind], K=Ki, lam=pspllam, mu=mu, e_mu=e_mu, e_yk=True, retfit=True)
               chi = ((mod[ind] - ymod)**2*we[ind]).sum()
               chired += [ chi / (we[ind].size-Ki)]
               BIC += [ chi + np.log(we[ind].size)*Ki]
               print('K={:10.1f}, chi={:10.2f}, chired={:10.4f}, BIC={:10.1f}'.format(Ki,chi,chired[i],BIC[i]))

            Koptimal = K[np.argmin(BIC)]
            ymod, smod = spl._ucbspl_fit(wmod[ind], mod[ind], we[ind], K=Koptimal, lam=serval_config.pspllam, sigk=True, reta=True)
            #smod, ymod = spl.ucbspl_fit(wmod[ind], mod[ind], we[ind], K=Ko, lam=pspllam, mu=mu, e_mu=e_mu, e_yk=True, retfit=True)
            print('Optimal K value: {}'.format(Koptimal))
            fig, ax = plt.subplots()
            ax.plot(K,BIC)
            ax.plot(Koptimal,min(BIC),marker='o',color='red')
            ax.vlines(nk,min(BIC),max(BIC))
            ax.set_xlabel('K')
            ax.set_ylabel('BIC')
            fig.savefig(savename_knotplot,dpi=600) ; plt.close()
            print('Saved to {}'.format(savename_knotplot))

        # estimate the number of valid points for each knot
        edges = 0.5 * (wko[1:]+wko[:-1])
        edges = np.hstack((edges[0]+2*(wko[0]-edges[0]), edges, edges[-1]+2*(wko[-1]-edges[-1])))
        bko,_ = np.histogram(wmod[ind], bins=edges, weights=(bmod[ind]==0)*1.0)


        '''estimate S/N for each spectrum and then combine all S/N'''
        sn = []
        yymod = mod * 0
        yymod[ind] = ymod
        #for i,sp in enumerate(spoklist[tset]):
        for i,sp in enumerate(splist):
            sp = sp.S
            #sp.read_data()
            if serval_config.simulation_mode:
                # NOTE: should sp.flag be > 0 ? 
                spt.S.header['HIERARCH COADD FILE %03i' % (i+1)] = (sp.timeid, 'rv = %0.5f km/s' % (-RV[i]/1000.))
                iind = (i, ind[i])   # a short-cut for the indexing
                signal = wmean(mod[iind], 1./emod[iind]**2)  # the signal
                noise = wrms(mod[iind]-yymod[iind], emod[iind])   # the noise
                sn.append(signal/noise)
            else:
                if sp.sn55 < serval_config.snmax:# and sp.flag is False:
                    # NOTE: should sp.flag be > 0 ? 
                    spt.S.header['HIERARCH COADD FILE %03i' % (i+1)] = (sp.timeid, 'rv = %0.5f km/s' % (-RV[i]/1000.))
                    iind = (i, ind[i])   # a short-cut for the indexing
                    signal = wmean(mod[iind], 1./emod[iind]**2)  # the signal
                    noise = wrms(mod[iind]-yymod[iind], emod[iind])   # the noise
                    sn.append(signal/noise)
                else:
                    print("NOT USING SPECTRUM FOR TEMPLATE:",i)
        #print("SN:",sn)
        sn = np.nansum(np.array(sn)**2)**0.5
        log.info('Combined spectrum: S/N: {:0.5f}'.format(sn))
        print('order',o,float("%.3f" % sn))
        spt.S.header['HIERARCH SERVAL COADD SN%03i' % o] = (float("%.3f" % sn), 'signal-to-noise estimate')

        #################
        # Applying the final template fit
        #################
        if estimate_optimal_knots:
            print('#######################################')
            print('Setting Optimal K={} as template knot value'.format(Koptimal))
            print('#######################################')
            # Recalculating using new smod from Koptimal
            yfit = smod(spt.ww[o][ind2])
            #wko = smod.xk()   # the knot positions
            #fko = smod()      # the knot values
            #eko = smod.sigk   # the error estimates for knot values
            #dko = smod.dk()   # ~second derivative at knots
        #################
        # Applying the final template fit and assign to spt attributes
        #################
        spt.ff[o][ind2] = yfit
        spt.wk[o] = wko
        spt.fk[o] = fko
        spt.ek[o] = eko
        spt.bk[o] = bko
        # =============================
        # Finished this order for all files
        # =============================

        # Save template stack
        spt.tstack_w[o]       = wmod
        spt.tstack_f[o]       = mod
        spt.tstack_ind[o]     = ind
        spt.tstack_tellind[o] = tellind
        spt.tstack_ind0[o]    = ind0
        spt.tstack_starind[o] = starind
        
        # add flattened
        wmin = spt.ww[o][0]
        wmax = spt.ww[o][-1]
        
        for i in range(len(splist)):
            m = (tellind[i]==False)&(starind[i]==False)&(wmod[i]>wmin)&(wmod[i]<wmax)
            x = wmod[i][m]
            y = mod[i][m]
            #y = mod[i][(tellind[i]==False)&(starind[i]==False)]
            #m = (wmod[i]> wmin) & (wmod[i] < wmax)
            #xx = x[m]
            yy = scipy.interpolate.interp1d(spt.ww[o],spt.ff[o],fill_value='extrapolate')(x)
            #print(np.sum((tellind[i]==False)&(starind[i]==False)&(x>wmin)&(x<wmax)))
            spt.tstack_w_flat[o][i][m] = x
            spt.tstack_f_flat[o][i][m] = y/yy
    
    if template_filename=='':
        print('Skipping saving template hdf5')
    else:
        fp = filepath.FilePath(template_filename)
        utils.make_dir(fp.directory)
        print("Saving template to {}".format(template_filename))
        read_spec.write_template(template_filename, spt.ff, spt.ww, spt.S.header, hdrref='', clobber=1)
        #fp.extension = 'hdf5'
        #print("Saving template to {}".format(fp._fullpath))
        #save_template_as_hdf5(spt,sps,filename=fp._fullpath)

    if plot:
        if len(orders)>10:
            print("Trying to plot too many orders. Break")
            sys.exit()
        for o in orders:
            fig, ax = plt.subplots(figsize=(12,8))
            wmod = spt.tstack_w[o]     
            mod  = spt.tstack_f[o]      
            ind  = spt.tstack_ind[o]     
            tellind = spt.tstack_tellind[o] 
            ind0 = spt.tstack_ind0[o]    

            if plot_baryshifted:
                print('Plotting baryshifted')
                ax.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",color="green",markersize=10,label="clipped: wmod[ind<ind0],mod[ind<ind0]")
                ax.plot(wmod[ind],mod[ind],label="data: wmod[ind],mod[ind]",marker="o",lw=1,markersize=3,color="black",alpha=0.3)
                ax.plot(wmod[tellind],mod[tellind],label="atm: wmod[tellind],mod[tellind]",marker="o",lw=0,markersize=3,color="firebrick")
                ax.plot(spt.wk[o],spt.fk[o],label="Knots: (wk[o],fk[o])",alpha=0.2,marker="o",markersize=2,color="chocolate")
                ax.plot(spt.ww[o],spt.ff[o],label="Template: (ww[o],ff[o])")
            #else:
            #    print('Plotting in telluric frame')
            #    ax.plot(barshift(wmod[ind<ind0],-spt.S.berv),mod[ind<ind0],lw=0,marker="D",color="green",markersize=10,label="clipped: wmod[ind<ind0],mod[ind<ind0]")
            #    ax.plot(barshift(wmod[ind],-spt.S.berv),mod[ind],label="data: wmod[ind],mod[ind]",marker="o",lw=1,markersize=3,color="black",alpha=0.3)
            #    ax.plot(barshift(wmod[tellind],-spt.S.berv),mod[tellind],label="atm: wmod[tellind],mod[tellind]",marker="o",lw=0,markersize=3,color="firebrick")
            #    ax.plot(barshift(spt.wk[o],-spt.S.berv),spt.fk[o],label="Knots: (wk[o],fk[o])",alpha=0.2,marker="o",markersize=2,color="chocolate")
            #    ax.plot(barshift(spt.ww[o],-spt.S.berv),spt.ff[o],label="Template: (ww[o],ff[o])")

            cx = ax.twiny()
            #x2 = np.arange(serval_config.ptmin,serval_config.ptmax)
            x2 = np.arange(0,serval_config.npix)
            cx.plot(x2,x2,lw=0)#dummy plot
            cx.minorticks_on()

            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.set_xlim(xlim)

            if plot_baryshifted:
                #ax.fill_between(barshift(spt.telluric_mask[:,0],spt.S.berv),spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="navy")
                #ax.fill_between(barshift(spt.sky_mask[:,0],spt.S.berv),spt.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
                ax.fill_between(redshift(spt.telluric_mask[:,0],vo=spt.S.berv,ve=vref),spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="navy")
                ax.fill_between(redshift(spt.sky_mask[:,0],vo=spt.S.berv,ve=vref),spt.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
            #else:
            #    ax.fill_between(spt.telluric_mask[:,0],spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="k")
            #    ax.fill_between(spt.sky_mask[:,0],spt.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
            ax.legend(bbox_to_anchor=(1,1),fontsize=8)
            ax.grid(lw=0.5,alpha=0.5)
            ax.minorticks_on()
            ax.set_title("Template o= {:0.5f}".format(o))
            fig.subplots_adjust(right=0.7)

        if select_tellurics:
            print("MASK SELECTING MODE")
            def onselect(xmin, xmax):
                print_mask(xmin,xmax)
            span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                rectprops=dict(alpha=0.5, facecolor='red'))
            return spt, span
    return spt



def calculate_rvs_from_final_template(spoklist,spt,rvguess=0.,vref=0.,orders=None,verb=True,
                                      ax=None,plot=False,inst="HPF",calculate_activity=True,
                                      plot_CaIRT=True,savedir_CaIRT='plots_ca_irt/',savedir_template_rvs=''):
    """
    Calculate full RVs from final template

    INPUT:
        spoklist: a list of sspectrum objects
        spt: template. Be sure to run spt.intepolate_and_baryshift_orders(), and generate_spectrum()

    OUTPUT:
        bjd
        rv
        snr
        e_rv
        RV
        e_RV

    EXAMPLE:
            spt.interpolate_and_baryshift_orders()
            ST = serval_help.generate_template(spt,splist,RV,plot=False,template_filename="template_full_test5.fits")#,plot=False)
            bjd, rv, snr, e_rv, RV, e_RV = serval_help.calculate_rvs_from_final_template(splist,spt,inst="HARPS")
        
    NOTES:
    """
    #print(inst)
    log = logger.logger()
    # Setting up arrays
    nspec = len(spoklist)
    if orders is None: 
        orders = serval_config.orders
    nord = serval_config.iomax

    results = dict((sp.S.timeid,['']*nord) for sp in spoklist)
    table = nans((7,nspec))
    bjd, RV, e_RV, rvm, rvmerr, RVc, e_RVc = table   # get pointer to the columns
    vabsvec = np.zeros(nspec)
    rv, e_rv = nans((2, nspec, nord))
    crx, e_crx = nans((2,nspec))   # get pointer to the columns
    tcrx = np.rec.fromarrays(nans((5,nspec)), names='crx,e_crx,a,e_a,l_v' )   # get pointer to the columns
    xo = nans((nspec,nord))
    snr = nans((nspec,nord))
    rchi2 = nans((nspec,nord))
    chi2map = [None] * nord
    chi2map = nans((nspec,nord, int(np.ceil((serval_config.v_hi-serval_config.v_lo)/ serval_config.v_step))))
    print(chi2map.shape)
    crx, e_crx = nans((2,nspec))   # get pointer to the columns
    tcrx = np.rec.fromarrays(nans((5,nspec)), names='crx,e_crx,a,e_a,l_v' )   # get pointer to the columns
    dLWo, e_dLWo = nans((2, nspec, nord)) # differential width change
    dLW, e_dLW = nans((2, nspec)) # differential width change
    resfactor = nans((nspec, nord)) # differential width change
    if calculate_activity:
        irt1 = np.ones((nspec,2))*np.nan
        irt1a = np.ones((nspec,2))*np.nan
        irt1b = np.ones((nspec,2))*np.nan
        irt2 = np.ones((nspec,2))*np.nan
        irt2a = np.ones((nspec,2))*np.nan
        irt2b = np.ones((nspec,2))*np.nan
        irt3 = np.ones((nspec,2))*np.nan
        irt3a = np.ones((nspec,2))*np.nan
        irt3b = np.ones((nspec,2))*np.nan
        irt_ind1_v = np.ones(nspec)*np.nan
        irt_ind1_e = np.ones(nspec)*np.nan
        irt_ind2_v = np.ones(nspec)*np.nan
        irt_ind2_e = np.ones(nspec)*np.nan
        irt_ind3_v = np.ones(nspec)*np.nan
        irt_ind3_e = np.ones(nspec)*np.nan

    # Loop over spectra
    for i,sp in enumerate(spoklist):
        log.info("i={}".format(i))
        fmod = sp.S.w * np.nan
        bjd[i]= sp.S.bjd
        # Loop over orders
        for o in orders:
            if plot:
                savename = '{}/{}_template_rv_o{}_{}_{}.png'.format(savedir_template_rvs,sp.S.obj,str(o).zfill(2),str(i).zfill(4),sp.date_str)
                utils.make_dir(savedir_template_rvs)
            else:
                savename = ''
            _rvo, _e_RV, _snr, _rchi2, _vgrid, _chi2map, _f2mod, _res, _par, _dLW, _e_dLW, _std = calculate_rv_for_order_from_final_template(sp,spt,o,rvguess=rvguess,
                                                                                                                                       verb=verb,vref=vref,plot=plot,ax=ax,
                                                                                                                                       inst=inst,save_res_dict=False,
                                                                                                                                       calculate_activity=calculate_activity,
                                                                                                                                       savename=savename)
            fmod[o] = _f2mod                                                                    
            #print(sp.S.timeid,o,_par,results)
            results[sp.S.timeid][o] = _par
            rv[i,o] = _rvo  
            snr[i,o] = _snr# stat['snr']
            rchi2[i,o] = _rchi2#stat['std']
            vgrid = _vgrid
            chi2map[i,o] = _chi2map # doesn't work
            e_rv[i,o] = _e_RV
            dLWo[i,o] = _dLW
            e_dLWo[i,o] = _e_dLW
            resfactor[i,o] = _std

        # Only use valid values (i.e., just the orders we are looking at, otherwise nan)
        ind, = np.where(np.isfinite(e_rv[i])) # do not use the failed and last order
        # Weigted RV and errors
        #RV[i], e_RV[i] = wsem(rv[i,ind], e=e_rv[i,ind])
        RV[i], e_RV[i] = weighted_average(rv[i,ind], e=e_rv[i,ind])
        #RV[i], e_RV[i] = np.nanmean(rv[i,ind]), np.nanstd(e_rv[i,ind])  # weight
        #RV[i], e_RV[i] = np.nanmean(rv[i,ind]), 1.  # weight
        #RV[i], e_RV[i] = np.nanmedian(rv[i,ind]), 1.  # weight
        #print("RV = {} +/- {} m/s".format(RV[i],e_RV[i]))

        ########################################
        # Chromatic index - CRX
        ########################################
        print('CALCULATING CRX')
        def func(x, a, b): return a + b*x #np.polynomial.polyval(x,a)
        x = np.mean(np.log(spt.S.w), axis=1)  # ln(lambda)
        xc = np.mean(x[ind])   # only to center the trend fit
        # fit trend with curve_fit to get parameter error
        pval, cov = curve_fit(func, x[ind]-xc, rv[i][ind], np.array([0.0, 0.0]), e_rv[i][ind])
        # pval[0]: a
        # pval[1]: b - the slope, what we are interested in (CRX)
        perr = np.sqrt(np.diag(cov))
        l_v = np.exp(-(pval[0]-RV[i])/pval[1]+xc)
        crx[i], e_crx[i], xo[i] = pval[1], perr[1], x
        tcrx[i] = crx[i], e_crx[i], pval[0], perr[0], l_v
        print('CRX={:10.3f}+-{:10.3f}, a={:10.3f}+-{:10.3f}, l_v={:10.3f}'.format(crx[i], e_crx[i], pval[0], perr[0], l_v))
        ########################################
        # End CRX
        ########################################

        ########################################
        # START - Ca IRT
        ########################################
        vabs = vref + RV[i]/1000.
        print('Vref={}km/s, Vabs={}km/s'.format(vref,vabs))
        vabsvec[i] = vabs

        savename_ca_irt = os.path.join(savedir_CaIRT,'{}_{:04d}_ca_irt.png'.format(spt.S.obj,i))
        if plot_CaIRT:
            print('Plotting Ca IRT plot: {}/{}'.format(i,len(spoklist)))
            title_ca_irt = '{}, {}'.format(spt.S.obj,os.path.basename(spoklist[i].filename))
        irt1[i],irt1a[i],irt1b[i],irt_ind1_v[i],irt_ind1_e[i],irt2[i],irt2a[i],irt2b[i],irt_ind2_v[i],irt_ind2_e[i],irt3[i],irt3a[i],irt3b[i],irt_ind3_v[i],irt_ind3_e[i] = get_ca_irt_indices_for_hpf_spectrum(sp,vabs,verbose=True,plot=plot_CaIRT,savename=savename_ca_irt,title=title_ca_irt)
        #plot_ca=False
        #_o = 3
        #irt1[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1',plot=plot_ca)
        #irt1a[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1a',plot=plot_ca)
        #irt1b[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1b',plot=plot_ca)
        #irt_ind1_v[i], irt_ind1_e[i] = get_line_index(irt1[i],irt1a[i],irt1b[i])
            #print('i={:2.0f},irt1={:4.2f},irt1a={:4.2f},irt1b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(i,irt1[i,0],irt1a[i,0],irt1b[i,0],irt_ind1_v[i],irt_ind1_e[i]))

        #_o = 4
        #irt2[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2',plot=plot_ca)
        #irt2a[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2a',plot=plot_ca)
        #irt2b[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2b',plot=plot_ca)
        #irt_ind2_v[i], irt_ind2_e[i] = get_line_index(irt2[i],irt2a[i],irt2b[i])
            #print('i={:2.0f},irt2={:4.2f},irt2a={:4.2f},irt2b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(i,irt2[i,0],irt2a[i,0],irt2b[i,0],irt_ind2_v[i],irt_ind2_e[i]))

        #_o = 5
        #irt3[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3',plot=plot_ca)
        #irt3a[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3a',plot=plot_ca)
        #irt3b[i] = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3b',plot=plot_ca)
        #irt_ind3_v[i], irt_ind3_e[i] = get_line_index(irt3[i],irt3a[i],irt3b[i])
        #print('i={:2.0f},irt3={:4.2f},irt3a={:4.2f},irt3b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(i,irt3[i,0],irt3a[i,0],irt3b[i,0],irt_ind3_v[i],irt_ind3_e[i]))

        ########################################
        # END - Ca IRT
        ########################################
        ind, = np.where(np.isfinite(e_dLWo[i]))
        dLW[i], e_dLW[i] = weighted_average(dLWo[i,ind], e=e_dLWo[i,ind])
    np.savetxt(os.path.join(savedir_CaIRT,'..','absolute_rvs.dat'), np.vstack([bjd,vabsvec,e_RV]).T)
    results_activity = {'dLW': dLW,
                        'e_dLW': e_dLW,
                        'dLWo': dLWo,
                        'e_dLWo': e_dLWo,
                        'crx': crx,
                        'e_crx': e_crx,
                        'xo': xo,
                        'irt1':  irt1,
                        'irt1a': irt1a,
                        'irt1b': irt1b,
                        'irt2':  irt2,
                        'irt2a': irt2a,
                        'irt2b': irt2b,
                        'irt3':  irt3,
                        'irt3a': irt3a,
                        'irt3b': irt3b,
                        'irt_ind1_v': irt_ind1_v,
                        'irt_ind1_e': irt_ind1_e,
                        'irt_ind2_v': irt_ind2_v,
                        'irt_ind2_e': irt_ind2_e,
                        'irt_ind3_v': irt_ind3_v,
                        'irt_ind3_e': irt_ind3_e,
                        'resfactor': resfactor
                        } #this is ln(lambda) wavelength center of order
    return bjd, rv, snr, e_rv, RV, e_RV, vgrid, chi2map, results_activity

def calculate_rv_for_order_from_final_template(sp,spt,o,rvguess=0.,verb=False,vref=0.,plot=True,ax=None,bx=None,cx=None,
                                               inst="HPF",save_res_dict=False,plot_in_pixelspace=False,calculate_activity=True,nbin=5,
                                               savename=''):
    """
    Calculate rv for one order using sp and spt
    
    INPUT:
    
    OUTPUT:
    
    EXAMPLE:
        sp = splist[1]
        rvo, e_RV, snr, rchi2, vgrid, chi2map, fmod, res = calculate_rv_for_order_from_final_template(sp,spt,o=5)
    
    NOTES:

    TODO:
        Actually feed the bad pixel map to the actual sp class ? 
        Should feed this into what is actually saved in the hdf5 file. That is done in the 'generate template()' function
    """
    if rvguess==0.:
        rvguess = vref
    sp.S.read_data()
    bjd = sp.S.bjd
    fmod = copy.deepcopy(sp.S.w[o]) * np.nan
    w2 = copy.deepcopy(sp.S.w[o])
    x2 = np.arange(w2.size)
    f2 = copy.deepcopy(sp.S.f[o])
    e2 = copy.deepcopy(sp.S.e[o])
    b2 = copy.deepcopy(sp.S.bpmap[o])
    pmin = serval_config.pmin
    pmax = serval_config.pmax
    b2[:pmin] = flag.out # outside
    b2[pmax:] = flag.out # outside
    # Here we flag b2 which is in pixel units. We flag both the tellurics in the reference spectrum, and also the corresponding tellurics from the template 
    # The template (ww, ff) is a smooth function with no masking
    b2[telluric_mask_interpolate(w2)>0.01]  = flag.atm # flag the tellurics in the epoch spectrum, 2048
    b2[sky_mask_interpolate(w2)>0.01]       = flag.sky # flag the sky in the epoch spectrum, 2048
    # barshift of sp.S.berv brings the wavelength to the barycenter
    # then barshifting by -spt.S.berv brings the wavelengths to the reference frame when the template was taken.

    #######################
    # NOTE 20201021 - was normally used
    if serval_config.flag_template_tellurics:
        print('##############')
        print('FLAGGING TEMPLATE TELLURICS AS WELL')
        print('NOT IMPLEMENTED')
        #b2[(telluric_mask_interpolate(barshift(w2, -spt.S.berv+sp.S.berv+(vref-rvguess)))>0.01)!=0] = flag.badT # Flag the tellurics in the template ref frame
        #b2[(sky_mask_interpolate(barshift(w2, -spt.S.berv+sp.S.berv+(vref-rvguess)))>0.01)!=0]      = flag.badT # flag the sky in the template
    #######################
    #_w2 = redshift(w2, vo=sp.S.berv, ve=-rvguess)
    #b2[stellar_mask_interpolate(_w2)>0.01] = flag.star

    # Before shifting
    #wmod = barshift(w2, np.nan_to_num(sp.S.berv))
    #ww = copy.deepcopy(spt.ww[o])

    # After shifting
    wmod = redshift(w2,vo=sp.S.berv,ve=vref)
    ww = copy.deepcopy(spt.ww[o])
    #ww = redshift(ww,vo=0.,ve=vref) #ww is already barshifted

    b2[stellar_mask_interpolate(wmod)>0.01] = flag.star

    # We will only use points where there are no issues
    pind = x2[b2==0]                                                                            

    ff = copy.deepcopy(spt.ff[o]) # ff is 0 in spt before generating template
    #kk = copy.deepcopy(spt.kk[o])
    #print('Recalculating knots from ww and ff')
    kk = serval_config.spline_cv(ww,ff)

    par, f2mod, keep, stat, chi2mapo = fitspec(ww,ff,kk, wmod, f2, e2,
                                               v=rvguess-vref, clip=serval_config.kapsig, nclip=serval_config.nclip, 
                                               deg=serval_config.deg,
                                               keep=pind, indmod=np.s_[pmin:pmax],
                                               chi2map=True)
    if calculate_activity:
        ###########################################################
        ###########################################################
        # CALCULATE dLW START
        ###########################################################
        '''
        We need the model at the observation and oversampled since we 
        need the second derivative including the polynomial
        '''
        print('Calculating dLW.............',end=' ')
        i0 = np.searchsorted(dopshift(ww,par.params[0]), ww[0])
        i1 = np.searchsorted(dopshift(ww,par.params[0]), ww[-1]) - 1
        ftmod_tmp = 0*ww
        ftmod_tmp[i0:i1] = calcspec(ww[i0:i1], *par.params) #calcspec does not work when w < wtmin
        ftmod_tmp[0:i0] = ftmod_tmp[i0]
        ftmod_tmp[i1:] = ftmod_tmp[i1-1]
        '''estimate differential changes in line width ("FWHM"). 
        Correlate the residuals with the second derivative'''
        kkk = scipy.interpolate.splrep(ww, ftmod_tmp)  # oversampled grid
        dy = scipy.interpolate.splev(wmod, kkk, der=1)
        ddy = scipy.interpolate.splev(wmod, kkk, der=2)
        #if not def_wlog:
        dy *= wmod
        ddy *= wmod**2.
        # what is this v for ?
        v = -c * np.dot(1./e2[keep]**2.*dy[keep], (f2-f2mod)[keep]) / np.dot(1./e2[keep]**2*dy[keep], dy[keep])
        # Equation 25, this is dLW = dsig
        dsig = c**2 * np.dot(1./e2[keep]**2.*ddy[keep], (f2-f2mod)[keep]) / np.dot(1./e2[keep]**2*ddy[keep], ddy[keep])
        # Equation 26
        e_dsig = c**2 * np.sqrt(1. / np.dot(1./e2[keep]**2., ddy[keep]**2.))
        # some scaling factor to scale the error by ?
        drchi = rms(((f2-f2mod) - dsig/c**2*ddy)[keep] / e2[keep])
        #if np.isnan(dsig) and not safemode: pause()
        dLW = dsig * 1000       # convert from (km/s) to m/s km/s
        e_dLW = e_dsig * 1000 * drchi
        print('dLW={:10.3f}+-{:10.3f}, drchi={:10.3f}'.format(dLW,e_dLW,drchi))
        #print par.params[1],par.params[0], v, dsig*1000, e_dsig
        ###########################################################
        # END Calculate dLW
        ###########################################################
        ###########################################################
    else:
        dLW = np.nan
        e_dLW = np.nan
       
    fmod = f2mod
    if par.perror is None: 
        par.perror = [0.,0.,0.,0.]
    rvo = par.params[0] * 1000.
    snr = stat['snr']
    rchi2 = stat['std']
    vgrid = chi2mapo[0]
    chi2map = chi2mapo[1]
    e_RV =par.perror[0] * stat['std'] * 1000 # NOTE 20190120, par.perror[0] *1000.
    res = np.nan * f2
    res[pmin:pmax] = (f2[pmin:pmax]-f2mod[pmin:pmax]) / e2[pmin:pmax]
    clipped = np.sort(list(set(pind).difference(set(keep)))) 

    # Setting clipped flag
    b2[clipped.astype(int)] = flag.clip

    if verb: print("{:2.0f} RV={:7.2f}+/-{:7.2f}m/s(+/-{:7.2f}m/s) std={:7.4f} snr={:7.4f} it={:2.0f} nkeep={:4.0f} nclip={:4.0f}".format(o, rvo,
        par.perror[0]*1000., par.perror[0]*1000*stat['std'],stat['std'],stat['snr'], par.niter, np.size(keep),len(clipped)))

    if plot:
        if ax is None and bx is None and cx is None:
            fig, (ax,bx,cx) = plt.subplots(nrows=3,figsize=(12,9),sharex=True,dpi=600)
        else:
            fig = ax.figure

        #if plot_in_pixelspace:
        #    x2 = np.arange(serval_config.npix)
        #    ax.plot(x2,f2)
        #    ax.plot(x2[b2!=0],f2[b2!=0],'r.',label='x2,f2')
        #    ax.plot(x2,fmod,label="Template: interpolated, poly-scaled (wmod,fmod)")
        #    bx.plot(x2,res,label="Normalized residuals ((f2-fmod)/e2)",color='gray')
        #    bx.plot(x2[b2!=0],res[b2!=0],label="badmask",color='red',marker='o',lw=0,markersize=5)
        #else:
        #ax.flat[1].plot(ww[b2==0],ff[b2==0],color="green",label="Template: Interpolated, no tellurics (ww[b2],ff[b2])",alpha=0.8)
        
        # Top plot
        ax.plot(wmod,fmod,label="Template (scaled)",zorder=12,lw=0.5)
        ax.plot(wmod[b2==0],f2[b2==0],marker='o',color='black',lw=0,label="Non-masked",markersize=2,alpha=0.3,zorder=0)
        ax.plot(wmod[b2!=0],f2[b2!=0],label="Masked",lw=0,color='red',marker='o',markersize=1,zorder=10,alpha=0.5)
        ax.plot(wmod[b2==flag.atm],f2[b2==flag.atm],label="Tellurics",lw=0,color='blue',marker='o',markersize=1,zorder=10,alpha=0.5)
        ax.plot(wmod[b2==flag.sky],f2[b2==flag.sky],label="Sky",lw=0,color='orange',marker='o',markersize=1,zorder=10,alpha=0.5)
        ax.plot(wmod[b2==flag.star],f2[b2==flag.star],label="Starmask",lw=0,color='brown',marker='o',markersize=3,zorder=11,alpha=0.5)
        ax.plot(wmod[b2==flag.clip],f2[b2==flag.clip],label="Clipped",lw=0,color='green',marker='D',markersize=7,zorder=11,alpha=0.6)

        # Residual plot - with tellurics
        bx.plot(wmod[b2==0],res[b2==0],marker='o',color='black',lw=0,label="Non-masked",markersize=2,alpha=0.3,zorder=0)
        bx.plot(wmod[b2!=0],res[b2!=0],label="Masked",lw=0,color='red',marker='o',markersize=1,zorder=10,alpha=0.5)
        bx.plot(wmod[b2==flag.atm],res[b2==flag.atm],label="Tellurics",lw=0,color='blue',marker='o',markersize=1,zorder=10,alpha=0.5)
        bx.plot(wmod[b2==flag.sky],res[b2==flag.sky],label="Sky",lw=0,color='orange',marker='o',markersize=1,zorder=10,alpha=0.5)
        bx.plot(wmod[b2==flag.star],res[b2==flag.star],label="Starmask",lw=0,color='brown',marker='o',markersize=3,zorder=11,alpha=0.5)
        bx.plot(wmod[b2==flag.clip],res[b2==flag.clip],label="Clipped",lw=0,color='green',marker='D',markersize=7,zorder=11,alpha=0.6)
        bx.legend(loc='upper right',fontsize=6)

        # Residual plot - without tellurics
        cx.plot(wmod[b2==0],res[b2==0],marker='o',markersize=2,alpha=0.15,label="Norm.Res",lw=0.5,color='black')
        df_bin = utils.bin_data(wmod[b2==0],res[b2==0],nbin)
        cx.plot(df_bin.x,df_bin.y,marker='.',alpha=0.5,markersize=10,lw=1,color='green',label='NBin={}'.format(nbin))
        df_bin = utils.bin_data(wmod[b2==0],res[b2==0],nbin*4)
        cx.plot(df_bin.x,df_bin.y,marker='.',alpha=0.8,markersize=10,lw=1,color='red',label='NBin={}'.format(nbin*4))
        cx.axhline(0,color='black')
        _std = np.nanstd(res[b2==0])
        cx.set_ylim(-5*_std,5.*_std)

        xlim = ax.get_xlim()
        ax.set_xlim(*xlim)
        for _xx in [ax,bx,cx]:
            utils.ax_apply_settings(_xx)

        _ax = ax.twinx()
        _bx = bx.twinx()
        _cx = cx.twinx()
        for _xx in [_ax,_bx,_cx]:
            _xx.set_xlim(*xlim)
            _xx.axes.get_yaxis().set_visible(False)
            _xx.set_ylim(0.9,1.0)
            #_xx.fill_between(barshift(spt.telluric_mask[:,0],sp.S.berv),spt.telluric_mask[:,1],alpha=0.1,label="Telluric mask",color=CP[0],lw=0)
            #_xx.fill_between(barshift(spt.sky_mask[:,0],sp.S.berv),spt.sky_mask[:,1],alpha=0.1,label="Sky mask",color="orange",lw=0)
            _xx.fill_between(redshift(spt.telluric_mask[:,0],vo=sp.S.berv,ve=vref),spt.telluric_mask[:,1],alpha=0.1,label="Telluric mask",color=CP[0],lw=0)
            _xx.fill_between(redshift(spt.sky_mask[:,0],vo=sp.S.berv,ve=vref),spt.sky_mask[:,1],alpha=0.1,label="Sky mask",color="orange",lw=0)
        ax.set_ylabel("Counts",fontsize=16)
        bx.set_ylabel("Residuals [w masked]",fontsize=16)
        cx.set_xlabel("Wavelength [A]",fontsize=16)
        cx.set_ylabel("Residuals [w/o masked]",fontsize=16)
        cx.legend(loc='upper right',fontsize=6)
        filename = sp.basename
        sn18 = sp.S.sn18
        TITLE = '{}, {}, o={}, SN18={:0.1f}, 1/SN18={:0.3f}ppt, STD={:0.3f}ppt, Berv={:0.3f}km/s'.format(sp.S.obj,filename,o,sn18,1000./sn18,_std,sp.S.berv)
        TITLE += '\nRes-Scale={:7.4f}, NKeep={:4.0f}/{}, NClip={:4.0f}'.format(stat['std'],np.size(keep),len(res),len(clipped))
        TITLE += '\nRV={:7.2f}+/-{:7.2f}m/s(+/-{:7.2f}m/s), dLW={:10.3f}+-{:10.3f}'.format(rvo,par.perror[0]*1000., par.perror[0]*1000*stat['std'],dLW,e_dLW)
        ax.set_title(TITLE)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.005,wspace=0.005,right=0.95,top=0.90)
        if savename != '':
            fig.savefig(savename,dpi=600) ; plt.close()
            print('Saved to {}'.format(savename))
        if save_res_dict:
            res_save = {"v":par.ssr[0],
                        "chi":par.ssr[1],
                        "w2": w2,    
                        "f2": f2,    
                        "res": res,  
                        "b2":b2,     
                        "e2":e2,
                        "pind":pind,
                        "fmod":fmod,
                        "f2mod":f2mod,
                        "wmod":wmod,
                        "ww":ww,
                        "ff":ff,
                        "kk":kk,
                        "res":res,
                        "indmod": np.s_[pmin:pmax],
                        }            
            utils.pickle_dump("temp_order10_shelp.pickle",res_save)
    return rvo, e_RV, snr, rchi2, vgrid, chi2map, fmod, res, par, dLW, e_dLW, stat['std']




def hpfspeclongbasename(filename):
    fp = filepath.FilePath(filename)
    fp.add_prefix(fp.directory.split('/')[-1]+"_")
    fp.add_prefix(fp.directory.split('/')[-2]+"_")
    return fp.basename

def extract_datetime_from_array(array):
    return [re.findall(r'\d{8}T\d{6}',name)[0] if len(re.findall(r'\d{8}T\d{6}',name))>0 else "" for name in array]

def extract_reference_dataset_frames_from_array(filenames,testfile='base'):
    """
    Function to get GJ699 reference dataset fitsfiles.
    Uses the dates in the ../DATA/gj699_test_dataset.csv as a reference
    """
    if testfile == 'base':
        df_ref = pd.read_csv('../DATA/gj699_test_dataset.csv')
    else:
        df_ref = pd.read_csv('../DATA/gj699_test_dataset_expanded.csv')
    df_comp = pd.DataFrame(zip(filenames,extract_datetime_from_array(filenames)),columns=['fitsfiles','dates'])
    return pd.merge(df_ref,df_comp).fitsfiles.values


def calculate_snr_from_template(x,y,xtemp,ytemp,pmin=100,pmax=-100):
    """
    Calculate a SNR using MAD comparing a single observation to a template
    
    INPUT:
        x     - epoch x value
        y     - epoch y value
        xtemp - x value from the template
        ytemp - y value from the template
    
    OUTPUT:
        SNR - signal to noise ratio
        
    EXAMPLE:
        calculate_snr_from_template(x=ST.tstack_w[o][i],
                                y=ST.tstack_f[o][i],
                                xtemp=ST.ww[o],
                                ytemp=ST.ff[o])
    """
    f = scipy.interpolate.interp1d(xtemp,ytemp,kind='cubic')(x[pmin:pmax])
    y = y[pmin:pmax]
    sigma = astropy.stats.mad_std(f-y)
    f_mean = np.nanmedian(f)
    return f_mean/sigma


def get_rv_label(rv,e_rv):
    return r"RV: $\sigma$={:4.2f}m/s, Med(errorbar)={:4.2f}m/s".format(np.std(rv),np.median(e_rv))

def vac2airshift(wave,rv=0.):
    """
    Shift wavelength to airwavelength for a given bulk RV shift.
    
    INPUT:
        rv is in km/s
        
    OUTPUT:
        Airwavelengths
    """
    wave = redshift(wave,ve=rv)
    wave = utils.vac2air(wave,P=760.,T=20.)
    return wave

    
def interpolate_nans(w,f,verbose = True,plot=False):
    """
    Linearly interpolate over bad regions in HPF data    

    EXAMPLE:
        o = 17
        interpolate_nans(np.arange(len(spall.splist[1].S.f[o])),spall.splist[1].S.f[o])
    """
    _w = np.copy(w)
    _f = np.copy(f)
    ind_good = np.isfinite(_f)
    ind_bad  = np.isnan(_f)
    ind_bad[0:4]= False
    ind_bad[-4:]= False
    if np.sum(ind_bad)>0:
        if verbose:
            print('Interpolating NaNs: #good={:4.1f} #nans={:4.1f}'.format(np.sum(ind_good),np.sum(ind_bad)))
        _f[ind_bad] = scipy.interpolate.interp1d(_w[ind_good],_f[ind_good],kind='linear')(_w[ind_bad])
        if plot:
            fig, ax = plt.subplots()
            ax.plot(w,f,color='black')
            ax.plot(_w[ind_bad],_f[ind_bad],color='red')
    else:
        print('#nans = 0. Skipping!')
    return _w, _f


def get_observation_number_from_filename(filename):
    """
    Get observation number from HPF header
    """
    return int(re.findall(r'/\d{8}/\d{4}/Slope-\d{8}T\d{6}',filename)[0].split(os.sep)[2])

def get_observation_number_from_header(filename):
    """
    Get observation number from HPF header
    """
    header = str(astropy.io.fits.getval(filename,"HISTORY"))
    return int(re.findall(r'/\d{6}\n\d{2}/\d{4}/Slope-\d{8}T\d{6}',header)[0].split(os.sep)[2])


def get_line_index(l, r1, r2):
    """
    Calculate line index from line center (l), and indices from offset regions r1 and r2
    
    Equation 27 in SERVAL paper
    
    EXAMPLE:
        vref = spt.target.rv#-110.51#12.4
        plot = False

        o = 3
        ind1_v = []
        ind1_e = []
        for sp in spall.splist:
            irt1 = get_absolute_index(sp.S.w[o],sp.S.f[o],sp.S.e[o],vref,sp.S.berv,line_name='CaIRT1',plot=plot)
            irt1a = get_absolute_index(sp.S.w[o],sp.S.f[o],sp.S.e[o],vref,sp.S.berv,line_name='CaIRT1a',plot=plot)
            irt1b = get_absolute_index(sp.S.w[o],sp.S.f[o],sp.S.e[o],vref,sp.S.berv,line_name='CaIRT1b',plot=plot)
            v, e = get_line_index(irt1,irt1a,irt1b)
            ind1_v.append(v)
            ind1_e.append(e)
    """
    if np.isnan(r1[0]) or np.isnan(r2[0]):
        return np.nan, np.nan
    s = l[0] / (r1[0]+r2[0]) * 2
    e = s * np.sqrt((l[1]/l[0])**2 + (r1[1]**2+r2[1]**2)/(r1[0]+r2[0])**2)
    return s, e

def get_line_wcen_dv1_dv2(line_name,verbose=False,return_vacuum=True):
    """
    Get line center and the offset in km/s
    
    INPUT:
        line_name - name of the line (e.g., CaIRT1, CaIRT1a)
    
    OUTPUT:
        wcen - line center in Angstrom in vacuum or air wavelengths
        dv1 - left region in km/s
        dv2 - right region in km/s
        
    EXAMPLE:
        get_line_wcen_dv1_dv2('CaIRT1')
    """
    lines = {'CaIRT1': (8498.02, -15., 15.),      # SERVAL definition
             'CaIRT1a': (8492, -40, 40),          # SERVAL definition
             'CaIRT1b': (8504, -40, 40),          # SERVAL definition
             'CaIRT2': (8542.09, -15., 15.),      # SERVAL definition
             #'CaIRT2a': (8542.09, -300., -200.), # SERVAL definition
             'CaIRT2a': (8542.09, 250., 350.),    # SERVAL definition
             'CaIRT2b': (8542.09, 250., 350.),    # 250, 350),      # My definition, +50 due telluric
             'CaIRT3': (8662.14, -15., 15.),      # NIST + my definition
             #'CaIRT3a': (8662.14, -250., -150.),  #-300, -200),    # NIST + my definition
             'CaIRT3a': (8662.14, 200., 300.),    #200, 300),      # NIST + my definition
             'CaIRT3b': (8662.14, 200., 300.),    #200, 300),      # NIST + my definition
             ####
             #'CaIRT3a': (8662.14, -300., -200.),  #-300, -200),    # NIST + my definition, OLD
         }
    wcen = lines[line_name][0]
    dv1 = lines[line_name][1]
    dv2 = lines[line_name][2]
    if verbose: 
        print('Using line: {}, wcen={}, dv1={}, dv2={}'.format(line_name,wcen,dv1,dv2))
    if return_vacuum:
        if verbose: print('Returning vacuum wavelength')
        wcen = airtovac.airtovac(wcen)
    return wcen, dv1, dv2

def shift_wavelength(wcen,vref,berv,dv):
    """
    Simple doppler shift of wavelength (used for line indices)
    
    NOTES:
        used by get_relative_index() and get_absolute_index()
    """
    c = 299792.4580   # [km/s] speed of light
    return wcen+((vref-berv+dv)/c)*wcen

def get_relative_index(w,f,fmod,v,berv,wcen=None,dv1=None,dv2=None,line_name='',plot=False,verbose=False):
    """
    Line index relative to template. Should mostly be looked at for inactive stars
    
    INPUT:
        w - wavelength in A in vacuum wavelengths (rest frame of Earth)
        f - flux
        fmod - the flux of the template (on the same grid as w and f)
        v - vref in km/s for the target 12.4km/s for AD Leo
        berv - barycentric velocity in km/s
        wcen - center of band in air wavelengths 
        dv1 - offset in km/s (start of band)
        dv2 - offset in km/s (end of band)
        
    OUTPUT:
        I - index value
        e_I - error on index
    
    EXAMPLE:
    
    NOTES:
        UNTESTED
    """
    if line_name!='':
        wcen, dv1, dv2 = get_line_wcen_dv1_dv2(line_name,return_vacuum=True)        
    w1 = shift_wavelength(wcen,v,berv,dv1)
    w2 = shift_wavelength(wcen,v,berv,dv2)
    m = (w1 < w) & (w < w2)
    if plot:
        fig, ax = plt.subplots(dpi=600)
        ax.plot(w,f,color='black')
        ax.plot(w[m],f[m],color='red',alpha=0.5)
        ax.plot(w,fmod,color='blue')
        ax.axvline(shift_wavelength(wcen,v,berv,0.),0,1,color='orange',linestyle='--',alpha=0.5)
    # relative index is not a good idea for variable lines
    if np.isnan(fmod[m]).all(): return np.nan, np.nan
    if np.isnan(fmod[m]).any(): print('Nans in line index')
    res = f[m] - fmod[m]
    I = np.mean(res/fmod[m])
    e_I = rms(np.sqrt(f[m])/fmod[m]) / np.sqrt(m.sum()) # only for photon noise
    return I, e_I

def get_absolute_index(w,f,e,v,berv,wcen=None,dv1=None,dv2=None,line_name='',plot=False,verbose=False,ax=None,plotline=True,zoomin=True,zoomin_window=15.):
    """
    Calculate absolute line index
    
    INPUT:
        w - wavelength in A in vacuum wavelengths (rest frame of Earth)
        f - flux
        e - error for flux
        v - vref in km/s for the target 12.4km/s for AD Leo
        berv - barycentric velocity in km/s
        wcen - center of band in air wavelengths 
        dv1 - offset in km/s (start of band)
        dv2 - offset in km/s (end of band)
        
    EXAMPLE:
        AD Leo:
        o = 3
        vref = 12.4
        get_absolute_index(spt.S.w[o],spt.S.f[o],spt.S.e[o],vref,spt.S.berv,line_name='CaIRT1',plot=True)
        get_absolute_index(spt.S.w[o],spt.S.f[o],spt.S.e[o],vref,spt.S.berv,line_name='CaIRT1a',plot=True)
        get_absolute_index(spt.S.w[o],spt.S.f[o],spt.S.e[o],vref,spt.S.berv,line_name='CaIRT1b',plot=True)
    """
    if line_name!='':
        wcen, dv1, dv2 = get_line_wcen_dv1_dv2(line_name,return_vacuum=True)
    w1 = shift_wavelength(wcen,v,berv,dv1)
    w2 = shift_wavelength(wcen,v,berv,dv2)
    m = (w1 < w) & (w < w2)
    
    if plot:
        if ax is None:
            fig, ax = plt.subplots(dpi=600)
        ax.plot(w,f,color='blue',zorder=-10)
        ax.plot(w[m],f[m],color='red')
        if plotline:
            ax.axvline(shift_wavelength(wcen,v,berv,0.),0,1,color='orange',linestyle='--',alpha=0.5)
        if zoomin:
            ax.set_xlim(shift_wavelength(wcen,v,berv,0.)-zoomin_window,shift_wavelength(wcen,v,berv,0.)+zoomin_window)
    
    # Absolute index
    I = np.mean(f[m])
    try: 
        e_I = 1. / m.sum() * np.sqrt(np.sum(e[m]**2))
    except Exception as ee:
        print(ee) 
        e_I = np.nan

    return I, e_I

def get_ca_irt_indices_for_hpf_spectrum(sp,vabs,verbose=True,plot=True,savename='',title='',zoomin=True):
    if plot:
        fig, (ax,bx,cx) = plt.subplots(nrows=3,dpi=600,figsize=(10,10))
        title += ', RV = {}km/s'.format(vabs)
        ax.set_title(title)
        cx.set_xlabel('Wavelength [A]')
        for xx in (ax,bx,cx):
            xx.tick_params(pad=2,labelsize=8)
            xx.set_ylabel('Flux')
        fig.subplots_adjust(hspace=0.08,right=0.95,left=0.05,top=0.95,bottom=0.05)

    _o = 3
    irt1 = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1',plot=plot,ax=ax,zoomin=zoomin)
    irt1a = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1a',plot=plot,ax=ax,plotline=False,zoomin=False)
    irt1b = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT1b',plot=plot,ax=ax,plotline=False,zoomin=False)
    irt_ind1_v, irt_ind1_e = get_line_index(irt1,irt1a,irt1b)
    if verbose: 
        print('irt1={:4.2f},irt1a={:4.2f},irt1b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(irt1[0],irt1a[0],irt1b[0],irt_ind1_v,irt_ind1_e))

    _o = 4
    irt2 = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2',plot=plot,ax=bx,zoomin=zoomin)
    irt2a = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2a',plot=plot,ax=bx,zoomin=zoomin)
    irt2b = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT2b',plot=plot,ax=bx,zoomin=zoomin)
    irt_ind2_v, irt_ind2_e = get_line_index(irt2,irt2a,irt2b)
    if verbose:
        print('irt2={:4.2f},irt2a={:4.2f},irt2b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(irt2[0],irt2a[0],irt2b[0],irt_ind2_v,irt_ind2_e))

    _o = 5
    irt3 = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3',plot=plot,ax=cx,zoomin=zoomin)
    irt3a = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3a',plot=plot,ax=cx,zoomin=zoomin)
    irt3b = get_absolute_index(sp.S.w[_o],sp.S.f[_o],sp.S.e[_o],vabs,sp.S.berv,line_name='CaIRT3b',plot=plot,ax=cx,zoomin=zoomin)
    irt_ind3_v, irt_ind3_e = get_line_index(irt3,irt3a,irt3b)
    if verbose:
        print('irt3={:4.2f},irt3a={:4.2f},irt3b={:4.2f},ind={:5.3f}+-{:5.3f}'.format(irt3[0],irt3a[0],irt3b[0],irt_ind3_v,irt_ind3_e))

    if savename!='':
        utils.make_dir(os.path.dirname(savename))
        print('Saving to: {}'.format(savename))
        fig.savefig(savename,dpi=600) ; plt.close()
    return irt1, irt1a, irt1b, irt_ind1_v, irt_ind1_e, irt2, irt2a, irt2b, irt_ind2_v, irt_ind2_e, irt3, irt3a, irt3b, irt_ind3_v, irt_ind3_e


def plot_hf_variable_panel_rvs_for_orders(hf,varname,orders,savename=''):
    """
    varname = dLWo,resfactor
    """
    fig, axx = plt.subplots(nrows=len(orders),ncols=2,dpi=600,figsize=(12,12))
    bjd = hf['rv/bjd'][:]
    rv = hf['rv/rv'][:]
    e_rv = hf['rv/e_rv'][:]
    var = hf['rv/{}'.format(varname)][:]
    var_MIN = np.nanmin(var)
    var_MAX = np.nanmax(var)
    for i, o in enumerate(orders):
        ax, bx = axx[i][0], axx[i][1]
        _ax = ax.twinx()
        _bx = bx.twinx()
        v = var[:,o]
        RV = rv[:,o]
        e_RV = e_rv[:,o]
        ax.plot(v,marker='o',lw=0.5,markersize=1,alpha=1,color='black',zorder=10)
        _ax.errorbar(range(len(bjd)),RV,e_RV,marker='o',markersize=1,lw=0,alpha=0.3,color=CP[0],elinewidth=0.5,capsize=0.5,label='o{}'.format(o),zorder=0)

        bx.plot(utils.jd2datetime(bjd),v,marker='o',lw=0.5,markersize=1,alpha=1,color='black',zorder=10)
        _bx.errorbar(utils.jd2datetime(bjd),RV,e_RV,marker='o',markersize=1,lw=0,alpha=0.3,color=CP[0],elinewidth=0.5,capsize=0.5,label='o{}'.format(o),zorder=0)

        ax.set_ylabel('{}: o{}'.format(varname,o))
        _bx.set_ylabel('RV [m/s] (blue)')
        ax.set_ylim(var_MIN,var_MAX)
        bx.set_ylim(var_MIN,var_MAX)
        utils.ax_apply_settings(ax)
        utils.ax_apply_settings(bx)
        ax.axes.get_xaxis().set_visible(False)
        _ax.axes.get_yaxis().set_visible(False)
        bx.axes.get_yaxis().set_visible(False)
        bx.axes.get_xaxis().set_visible(False)
    ax.set_xlabel('Obs #')
    bx.set_xlabel('Date [UT]')
    ax.axes.get_xaxis().set_visible(True)
    bx.axes.get_xaxis().set_visible(True)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05,wspace=0.01)
    bx.tick_params(labelsize=8)
    if savename!='':
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))
    


