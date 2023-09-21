import os
import astropy.time
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import re
import pandas as pd
import radvel
import filepath
import shutil
from PyAstronomy.pyasl import binningx0dt
import subprocess

norm_mean     = lambda x: x/np.nanmean(x)

def pickle_dump(filename,obj):
    savefile = open(filename,"w")
    pickle.dump(obj, savefile)
    savefile.close()
    print("Saved to {}".format(filename))

def pickle_load(filename,python3=True):
    if python3:
        openfile = open(filename,"rb")
    return pickle.load(openfile,encoding='latin1')


def jd2datetime(times):
    return np.array([astropy.time.Time(time,format="jd",scale="utc").datetime for time in times])

def iso2jd(times):
    return np.array([astropy.time.Time(time,format="iso",scale="utc").jd for time in times])

def make_dir(dirname,verbose=True):
    try:
        os.makedirs(dirname)
        if verbose==True: print("Created folder:",dirname)
    except OSError:
        if verbose==True: print(dirname,"already exists. Skipping")



def vac2air(wavelength,P,T,input_in_angstroms=True):
    """
    Convert vacuum wavelengths to air wavelengths

    INPUT:
        wavelength - in A if input_in_angstroms is True, else nm
        P - in Torr
        T - in Celsius

    OUTPUT:
        Wavelength in air in A if input_in_angstroms is True, else nm
    """
    if input_in_angstroms:
        nn = n_air(P,T,wavelength/10.)
    else:
        nn = n_air(P,T,wavelength)
    return wavelength/(nn+1.)

def n_air(P,T,wavelength):
    """
    The edlen equation for index of refraction of air with pressure
    
    INPUT:
        P - pressure in Torr
        T - Temperature in Celsius
        wavelength - wavelength in nm
        
    OUTPUT:
        (n-1)_tp - see equation 1, and 2 in REF below.
        
    REF:
        http://iopscience.iop.org/article/10.1088/0026-1394/30/3/004/pdf

    EXAMPLE:
        nn = n_air(763.,20.,500.)-n_air(760.,20.,500.)
        (nn/(nn + 1.))*3.e8
    """
    wavenum = 1000./wavelength # in 1/micron
    # refractivity is (n-1)_s for a standard air mixture
    refractivity = ( 8342.13 + 2406030./(130.-wavenum**2.) + 15997./(38.9-wavenum**2.))*1e-8
    return ((P*refractivity)/720.775) * ( (1.+P*(0.817-0.0133*T)*1e-6) / (1. + 0.0036610*T) )

def get_cmap_colors(cmap='jet',p=None,N=10):
    """

    """
    cm = plt.get_cmap(cmap)
    if p is None:
        return [cm(i) for i in np.linspace(0,1,N)]
    else:
        normalize = matplotlib.colors.Normalize(vmin=min(p), vmax=max(p))
        colors = [cm(normalize(value)) for value in p]
        return colors

def ax_apply_settings(ax,ticksize=None):
    """
    Apply axis settings that I keep applying
    """
    ax.minorticks_on()
    if ticksize is None:
        ticksize=12
    ax.tick_params(pad=3,labelsize=ticksize)
    ax.grid(lw=0.3,alpha=0.3)

def ax_add_colorbar(ax,p,cmap='jet',tick_width=1,tick_length=3,direction='out',pad=0.02,minorticks=False,*kw_args):
    """
    Add a colorbar to a plot (e.g., of lines)

    INPUT:
        ax - axis to put the colorbar
        p  - parameter that will be used for the scaling (min and max)
        cmap - 'jet', 'viridis'

    OUTPUT:
        cax - the colorbar object

    NOTES:
        also see here:
            import matplotlib.pyplot as plt
            sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=1))
            # fake up the array of the scalar mappable. Urgh...
            sm._A = []
            plt.colorbar(sm)

    EXAMPLE:
        cmap = 'jet'
        colors = get_cmap_colors(N=len(bjds),cmap=cmap)
        fig, ax = plt.subplots(dpi=200)
        for i in range(len(bjds)):
            ax.plot(vin[i],ccf_sub[i],color=colors[i],lw=1)
        ax_apply_settings(ax,ticksize=14)
        ax.set_xlabel('Velocity [km/s]',fontsize=20)
        ax.set_ylabel('Relative Flux ',fontsize=20)
        cx = ax_add_colorbar(ax,obs_phases,cmap=cmap,pad=0.02)
        ax_apply_settings(cx)
        cx.axes.set_ylabel('Phase',fontsize=20,labelpad=0)


        cbar.set_clim(1.4,4.1)
        cbar.set_ticks([1.5,2.0,2.5,3.0,3.5,4.0])
    """
    cax, _ = matplotlib.colorbar.make_axes(ax,pad=pad,*kw_args)
    normalize = matplotlib.colors.Normalize(vmin=np.nanmin(p), vmax=np.nanmax(p))
    cm = plt.get_cmap(cmap)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=normalize)
    if minorticks:
        cax.minorticks_on()
    cax.axes.tick_params(width=tick_width,length=tick_length,direction=direction)
    return cax, cbar

def get_indices_of_items(arr,items):
    return np.where(pd.DataFrame(arr).isin(items))[0]


def remove_items_from_list(l,bad_items):
    ibad = np.where(pd.DataFrame(l).isin(bad_items))[0]
    return np.delete(l,ibad)

def savefigure(fig,savename,s1='{}',p1='',s2='{}',p2='',dpi=200):
    """
    Handy function to save figures and append suffixes to filenames
    
    EXAMPLE:
        savefigure(fig,'MASTER_FLATS/COMPARE_PLOTS/testing.png',s1='_o{}',p1=5,s2='_spi{}',p2=14)
    """
    fp = FilePath(savename)
    make_dir(fp.directory)
    fp.add_suffix(s1.format(p1))
    fp.add_suffix(s2.format(p2))
    fig.tight_layout()
    fig.savefig(fp._fullpath,dpi=dpi)
    print('Saved figure to: {}'.format(fp._fullpath))

def grep_date(string,intype="isot",outtype='iso'):
    """
    A function to extract date from string.

    INPUT:
        string: string
        intype: "isot" - 20181012T001823
                "iso"  - 20181012
        outtype: "iso" - iso
                 "datetime" - datetime

    OUTPUT:
        string with the date
    """
    if intype == "isot":
        date = re.findall(r'\d{8}T\d{6}',string)[0]
    elif intype == "iso":
        date = re.findall(r'\d{8}',string)[0]
    else:
        print("intype has to be 'isot' or 'iso'")
    if outtype == 'iso':
        return date
    elif outtype == 'datetime':
        return pd.to_datetime(date).to_pydatetime()
    else:
        print('outtype has to be "iso" or "datetime"')

def grep_dates(strings,intype="isot",outtype='iso'):
    """
    A function to extract date from strings

    INPUT:
        string: string
        intype: "isot" - 20181012T001823
                "iso"  - 20181012
        outtype: "iso" - iso
                 "datetime" - datetime

    OUTPUT:
        string with the date

    EXAMPLE:
        df = grep_dates(files,intype="isot",outtype='series')
        df['2018-06-26':].values
    """
    if outtype=='series':
        dates = [grep_date(i,intype=intype,outtype='datetime') for i in strings]
        return pd.Series(index=dates,data=strings)
    else:
        return [grep_date(i,intype,outtype) for i in strings]

def replace_dir(files,old,new):
    for i,f in enumerate(files):
        files[i] = files[i].replace(old,new)
    return files

def get_header_df(fitsfiles,keywords=["OBJECT","DATE-OBS"],verbose=True):
    """
    A function to read headers and returns a pandas dataframe

    INPUT:
    fitsfiles - a list of fitsfiles
    keywords - a list of header keywords to read

    OUTPUT:
    df - pandas dataframe with column names as the passed keywords
    """
    headers = []
    for i,name in enumerate(fitsfiles):
        if verbose: print(i,name)
        head = astropy.io.fits.getheader(name)
        values = [name]
        for key in keywords:
            values.append(head[key])
        headers.append(values)
    df_header = pd.DataFrame(headers,columns=["filename"]+keywords)
    return df_header



def get_phases_sorted(t, P, t0,rvs=None,rvs_err=None,sort=True,centered_on_0=True,tdur=None):
    """
    Get a sorted pandas dataframe of phases, times (and Rvs if supplied)
    
    INPUT:
    t  - times in jd
    P  - period in days
    t0 - time of periastron usually
    
    OUTPUT:
    df - pandas dataframe with columns:
     -- phases (sorted)
     -- time - time
     -- rvs  - if provided
    
    NOTES:
    Useful for RVs.    
    """
    phases = np.mod(t - t0,P)
    phases /= P
    df = pd.DataFrame(zip(phases,t),columns=['phases','time'])
    if rvs is not None:
        df['rvs'] = rvs
    if rvs_err is not None:
        df['rvs_err'] = rvs_err
    if centered_on_0:
        _p = df.phases.values
        m = df.phases.values > 0.5
        _p[m] = _p[m] - 1.
        df['phases'] = _p
    if tdur is not None:
        df["intransit"] = np.abs(df.phases) < tdur/(2.*P)
        print("Found {} in transit".format(len(df[df['intransit']])))
    if sort:
        df = df.sort_values('phases').reset_index(drop=True)
    return df


def bin_data(x,y,numbins):
        """
        Returns a dataframe with binned data.
        
        INPUT:
            numbins is the number of bins 
        
        OUTPUT:
            Dataframe with the binned data. Columns x, y
        """
        timestamp = x
        rel_flux  = y
        rel_flux_rms_per_bin = np.zeros(numbins)
        bins = np.arange(numbins)+1
        for i in bins:
            if i == 1:
                rms = np.std(rel_flux)
            else:
                bin_timeAndRelFlux, pointsPerBin = binningx0dt(timestamp, rel_flux,useBinCenter=True, removeNoError=True, reduceBy=i,useMeanX=True)
                rms = np.std(bin_timeAndRelFlux[::,1])
                #print("len calc",len(bin_timeAndRelFlux[::,1]))
            rel_flux_rms_per_bin[i-1] = rms
        df_binned = pd.DataFrame(zip(bin_timeAndRelFlux[::,0],bin_timeAndRelFlux[::,1]),columns=["x","y"])
        return df_binned #, rel_flux_rms_per_bin

def copy_files_to_new_dir(files,folder,suffix="",prefix="",verbose=True):
    """
    Copy a list of files to another directory
    
    EXAMPLE:
    copy_files_to_new_dir(fitsfiles,'/media/gks/macwin/LCO_PROJECT/20180411/gj388/ip/',prefix='AA',suffix='BB')
    """
    make_dir(folder,verbose=verbose)
    for i in files:
        fp = filepath.FilePath(i)
        fp.directory = folder
        fp.add_prefix(prefix)
        fp.add_suffix(suffix)
        if verbose: print('Copying {} to: {}'.format(i,fp._fullpath),end=' ')
        shutil.copy(i,fp._fullpath)
        if verbose: print("Done")

def save_string_to_file(s, filename,verbose=True):
    """
    Save a string to a file
    """
    with open(filename, "w") as f:
        f.write(s)
    if verbose:
        print('Saved to {}'.format(filename))

def runcmd(cmd, verbose = True, *args, **kwargs):
    """
    runcmd('echo "Hello, World!"', verbose = True)
    """

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass
