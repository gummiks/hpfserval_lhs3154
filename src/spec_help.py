from __future__ import print_function
import sys
import os
import barycorrpy
import scipy.signal
import matplotlib.pyplot as plt
import scipy.interpolate
import numpy as np
import pandas as pd
import scipy.ndimage.filters
from scipy.ndimage.filters import correlate1d
import astropy.constants as aconst
import seaborn as sns
import scipy.interpolate
import h5py
import utils
import astropy.time
import astropy.io
cp = sns.color_palette("colorblind")
c = 299792.4580   # [km/s]

def vacuum_to_air(wl):
    """
    Converts vacuum wavelengths to air wavelengths using the Ciddor 1996 formula.

    :param wl: input vacuum wavelengths
    :type wl: numpy.ndarray

    :returns: numpy.ndarray

    .. note::

        CA Prieto recommends this as more accurate than the IAU standard.

    """
    if not isinstance(wl, np.ndarray):
        wl = np.array(wl)

    sigma = (1e4 / wl) ** 2
    f = 1.0 + 0.05792105 / (238.0185 - sigma) + 0.00167917 / (57.362 - sigma)
    return wl / f

def get_flux_from_file(filename,o=None,ext=1):
    """
    Get flat flux for a given order

    NOTES:
        f_flat = get_flat_flux('MASTER_FLATS/20180804/alphabright_fcu_march02_july21_deblazed.fits',5)
    """
    hdu = astropy.io.fits.open(filename)
    if o is None:
        return hdu[ext].data
    else:
        return hdu[ext].data[o]

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize=12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.5,alpha=0.5)

def jd2datetime(times):
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])

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

def compress_mask(m):
    """
    Compresses a boolean mask to return only the values where there is a jump. Useful for SERVAL masks
    
    INPUT:
    
    EXAMPLE:
        o = 5
        x = img.data[o]
        m = generate_telluric_mask_from_terraspec(x)
        mm = compress_mask(m)
        x = np.arange(len(m))
        plt.plot(x,m)
        plt.plot(x[mm],m[mm])
    """
    mm = np.zeros(len(m))==1. # create a boolean false array
    i = 0
    while i < len(m):
        if i==0:
            mm[0] = True
            i += 1
        while (m[i] == m[i-1]) & (i < len(m)-1):
            i += 1
        else:
            mm[i] = True
            mm[i-1] = True
            i += 1
        if i==(len(m)):
            mm[i-1] == True
    return mm


def group_tracks(bjds,threshold=0.05,plot=False):
    """
    Group HET tracks
    
    INPUT:
        bjds - bjd times
        threshold - the threshold to define a new track
                    assumes a start of a new track if it is more than threshold apart
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(bjds))
    diff = np.diff(bjds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(bjds))
        for i in x:
            plt.plot(i,bjds[i],marker='o',color=cc[groups[i]])
    return groups

def group_inds(inds,threshold=1,plot=False):
    """
    Group indices
    
    INPUT:
        inds - indices times
        threshold - the threshold to define a new group
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(inds))
    diff = np.diff(inds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(inds))
        for i in x:
            plt.plot(i,inds[i],marker='o',color=cc[groups[i]])
    return groups

def detrend_maxfilter_gaussian(flux,n_max=300,n_gauss=500,plot=False):
    """
    A function useful to estimate spectral continuum

    INPUT:
        flux: a vector of fluxes
        n_max: window for max filter
        n_gauss: window for gaussian filter smoothing

    OUTPUT:
        flux/trend - the trend corrected flux
        trend - the estimated trend

    EXAMPLE:
        f_norm, trend = spec_help.detrend_maxfilter_gaussian(df_temp.flux,plot=True)
    """
    flux_filt = scipy.ndimage.filters.maximum_filter1d(flux,n_max)
    trend = scipy.ndimage.filters.gaussian_filter1d(flux_filt,sigma=n_gauss)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(flux)
        ax.plot(trend)
        fig, ax = plt.subplots()
        ax.plot(flux/trend)
    return flux/trend, trend

def average_ccf(ccfs):
    """
    A function to average ccfs
    
    INPUT:
        An array of CCFs
        
    OUTPUT:
    
    """
    ccfs = np.sum(ccfs,axis=0)
    ccfs /= np.nanmedian(ccfs)
    return ccfs


def barshift(x, v=0.,def_wlog=False):
   """
   Convenience function for redshift.

   x: The measured wavelength.
   v: Speed of the observer [km/s].

   Returns:
      The true wavelengths at the barycentre.

   """
   return redshift(x, vo=v,def_wlog=def_wlog)


def redshift(x, vo=0., ve=0.,def_wlog=False):
   """
   x: The measured wavelength.
   v: Speed of the observer [km/s].
   ve: Speed of the emitter [km/s].

   Returns:
      The emitted wavelength l'.

   Notes:
      f_m = f_e (Wright & Eastman 2014)

   """
   if np.isnan(vo): vo = 0     # propagate nan as zero (@calibration in fib B)
   a = (1.0+vo/c) / (1.0+ve/c)
   if def_wlog:
      return x + np.log(a)   # logarithmic
      #return x + a          # logarithmic + approximation v << c
   else:
      return x * a
      #return x / (1.0-v/c)


def broaden_binarymask(x,y,step=0.01,thres=0.01,v=[1,1,1]):
    xx = np.arange(x[0],x[-1],step)
    yy = (scipy.interpolate.interp1d(x,y)(xx)>thres)*1.
    yy = np.convolve(yy,v,mode='same')>1.
    #y = (scipy.interpolate.interp1d(xx,yy,fill_value='extrapolate')(x)>thres)*1.
    return xx, yy






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
        RV[epoch], E_RV[epoch] = weighted_rv_average(rv[epoch,orders],e_rv[epoch,orders])
        if sub_mean:
            RV -= np.nanmean(RV)
    if plot:
        nbjd, nRV, e_nRV = bin_rvs_by_track(bjd,RV,E_RV)
        plot_RVs(bjd,tRV=RV,tRV_err=E_RV,nbjd=nbjd,nRV=nRV,nRV_err=e_nRV,ax=ax)
    return RV, E_RV

def plot_RVs(bjd,pRV=None,pRV_err=None,tRV=None,tRV_err=None,nbjd=None,nRV=None,nRV_err=None,ax=None,title=""):
    """
    Make an overview comparison RV plot of pre-rvs, template RVs and nightly binned RVs

    plot_RVs(df.bjd,df.pRV,df.pRV_err,df.tRV,df.tRV_err,nbjd,nRV,nRV_err)
    
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(10,6),sharex=True,dpi=600)
    if pRV is not None:
        #pRV -= np.mean(pRV)
        label1="Pre-RV: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(pRV),np.median(pRV_err))
        ax.errorbar(jd2datetime(bjd),-pRV,pRV_err,marker="o",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=4,label=label1,alpha=0.2)
    if tRV is not None:
        #tRV -= np.mean(tRV)
        label2="Unbinned: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(tRV),np.median(tRV_err))
        ax.errorbar(jd2datetime(bjd),tRV,tRV_err,marker="o",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=4,label=label2,alpha=0.4)
    if nRV is not None:
        #nRV -= np.mean(nRV)
        label3="Binned Track: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(nRV),np.median(nRV_err))
        ax.errorbar(jd2datetime(nbjd),nRV,nRV_err,marker="h",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=6,label=label3,color='crimson')
    ax.grid(lw=0.5,alpha=0.5)
    ax.tick_params(axis="both",labelsize=16,pad=3)
    ax.minorticks_on()
    #ax.set_title("HPF PreRVs: GJ 699 - no drift correction",fontsize=16)
    ax.set_ylabel("RV [m/s]",fontsize=20)
    ax.set_xlabel("Time [UT]",fontsize=20)
    ax.legend(loc="upper left",fontsize=8)
    ax.tick_params(axis='x',pad=3,labelsize=9)
    #ax.margins(y=0.3)
    ax.set_title(title)

def bin_rvs_by_track(bjd,RV,RV_err,exclude_nans=True):
    """
    Bin RVs in an HET track

    INPUT:
        bjd
        RV
        RV_err

    OUTPUT:

    """
    track_groups = group_tracks(bjd,threshold=0.05,plot=False)
    date = [str(i)[0:10] for i in jd2datetime(bjd)]
    df = pd.DataFrame(zip(date,track_groups,bjd,RV,RV_err),
            columns=['date','track_groups','bjd','RV','RV_err'])
    g = df.groupby(['track_groups'])
    ngroups = len(g.groups)

    # track_bins
    nbjd, nRV, nRV_err = np.zeros(ngroups),np.zeros(ngroups),np.zeros(ngroups)
    for i, (source, idx) in enumerate(g.groups.items()):
        cut = df.loc[idx]
        #nRV[i], nRV_err[i] = wsem(cut.RV.values,cut.RV_err)
        if exclude_nans:
            _m = (np.isfinite(cut.RV.values)) & (np.isfinite(cut.RV_err.values))
            numnan = len(cut.RV.values)-np.sum(_m)
            if numnan > 0:
                print('Masked {} nans'.format(numnan))
            nRV[i], nRV_err[i] = weighted_rv_average(cut.RV.values[_m],cut.RV_err.values[_m])
        else:
            nRV[i], nRV_err[i] = weighted_rv_average(cut.RV.values,cut.RV_err.values)
        nbjd[i] = np.mean(cut.bjd.values)
    return nbjd, nRV, nRV_err

def group_tracks(bjds,threshold=0.05,plot=False):
    """
    Group HET tracks
    
    INPUT:
        bjds - bjd times
        threshold - the threshold to define a new track
                    assumes a start of a new track if it is more than threshold apart
    
    EXAMPLE:
        g = group_tracks(bjds,threshold=0.05)
        cc = sns.color_palette(n_colors=len(g))
        x = np.arange(len(bjds))
        for i in range(len(bjds)):
        plt.plot(x[i],bjds[i],marker='o',color=cc[g[i]])
    """
    groups = np.zeros(len(bjds))
    diff = np.diff(bjds)
    groups[0] = 0
    for i in range(len(diff)):
        if diff[i] > threshold:
            groups[i+1] = groups[i] + 1
        else:
            groups[i+1] = groups[i]
    groups = groups.astype(int)
    if plot:
        fig, ax = plt.subplots()
        cc = sns.color_palette(n_colors=len(groups))
        x = range(len(bjds))
        for i in x:
            plt.plot(i,bjds[i],marker='o',color=cc[groups[i]])
    return groups

def weighted_rv_average(x,e):
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

def weighted_average(error):
    """
    Weighted average. Useful to add rv errors together.
    """
    w = np.array(error)**(-2.)
    return np.sum(w)**(-0.5)






def bin_data_with_groups(x,y,yerr,group):
    """
    A function to bin a data according to groups
    
    EXAMPLE:
        group = spec_help.group_tracks(df_all.time,0.002)
        x = df_all.time.values
        y = df_all.rv.values
        yerr = df_all.e_rv.values
        group = df_all.groups.values
        df_bin = bin_data_with_groups(x,y,yerr,group)
    """
    df = pd.DataFrame(zip(x,y,yerr,group),columns=['x','y','yerr','group'])
    xx = []
    yy = []
    yyerr = []
    for i in df.group.unique():
        #print(i)
        _d = df[group==i]

        if len(_d)==1:
            #print('len',len(_d))
            #print(_d.x.values)
            xx.append(_d.x.values[0])
            yy.append(_d.y.values[0])
            yyerr.append(_d.yerr.values[0])

        if len(_d)>1:
            #print('len',len(_d))
            xx.append(np.mean(_d.x.values))
            #print('mean',np.mean(_d.x.values))
            _y, _yerr = weighted_rv_average(_d.y.values,_d.yerr.values)
            yy.append(_y)
            yyerr.append(_yerr)
        #print('lenxx=',len(xx))
    df_bin = pd.DataFrame(zip(xx,yy,yyerr),columns=['x','y','yerr'])
    return df_bin

def bin_data_with_errors(x,y,yerr,nbin):
    """
    Bin data with errorbars
    
    EXAMPLE:
        bin_data_with_errors(df_bin.x.values,df_bin.y.values,df_bin.yerr.values,2)
    """
    xx = []
    yy = []
    yyerr = []
    nbin = int(nbin)
    #print(len(x)/nbin)
    for i in range(int(len(x)/nbin)):
        #print(x[i*nbin:(i+1)*nbin])
        xx.append(np.mean(x[i*nbin:(i+1)*nbin]))
        _y, _yerr = weighted_rv_average(y[i*nbin:(i+1)*nbin],yerr[i*nbin:(i+1)*nbin])
        yy.append(_y)
        yyerr.append(_yerr)
    #print(x[(i+1)*nbin:])
    if len(x[(i+1)*nbin:])>0:
        xx.append(np.mean(x[(i+1)*nbin:]))
        _y, _yerr = weighted_rv_average(y[(i+1)*nbin:],yerr[(i+1)*nbin:])
        yy.append(_y)
        yyerr.append(_yerr)
    df_bin = pd.DataFrame(zip(xx,yy,yyerr),columns=['x','y','yerr'])
    return df_bin
