import matplotlib.pyplot as plt
import serval_config
import scipy.interpolate
import serval
import filepath
import sspectrum
import collections
import os
import numpy as np
import pandas as pd
import serval_help
import seaborn as sns
from wstat import wsem
import utils
import html_help
import seaborn as sns
import scipy.stats
import lomb_scargle_plot
import pickle
import pdb
import spec_help

cp = [(0.0, 0.4470588235294118, 0.6980392156862745),
      (0.0, 0.6196078431372549, 0.45098039215686275),
      (0.8352941176470589, 0.3686274509803922, 0.0)]

def plot_pre_RVs(bjd,RV,e_RV,ax=None):
    """
    Plot a nice plot of pre RVs with errors and median errorbars.

    EXAMPLE:
    bjd,rv,e_rv,RV,e_RV = serval_help.calculate_pre_RVs(splist,spt,orders=[5],plot=False,verbose=False)
    
    """
    if ax == None:
        fig, ax = plt.subplots(figsize=(10,6),sharex=True,dpi=600)
    label1="STD={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(RV),np.median(e_RV))
    ax.errorbar(utils.jd2datetime(bjd),RV,e_RV,marker="o",lw=0,elinewidth=0.5,
                barsabove=True,mew=0.5,capsize=2,markersize=4,label=label1)
    ax.grid(lw=0.5,alpha=0.5)
    ax.tick_params(axis="both",labelsize=16)
    ax.minorticks_on()
    ax.set_ylabel("RV [m/s]",fontsize=20)
    ax.set_xlabel("Time [UT]",fontsize=20)
    ax.legend(loc="upper left",fontsize=16)
    ax.tick_params(axis="both",pad=5)
    ax.margins(x=0.05,y=0.15)

def plot_rvs_by_order(bjd,rv,e_rv,orders,start=None,stop=None,plot_errorbar=False,ax=None):
    """
    Plot a nice RV plot by order.

    INPUT:
    bjd - array of bjds
    rv - serval rv array
    e_rv - serval e_rv array (frames, orders) in size
    
    EXAMPLE:
        start = "2018-03-01 00:00:00"
        stop  = "2018-03-05 00:00:00"
        plot_rvs_by_order(df.bjd.values,_rv,_e_rv,orders=[5,14],plot_errorbar=False,start=start,stop=stop)
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=600)
    date = utils.jd2datetime(bjd)
    df_rv = pd.DataFrame(rv,index=date)
    df_e_rv = pd.DataFrame(e_rv,index=date)
    if start is not None and stop is not None:
        df_rv = df_rv[start:stop]
        df_e_rv = df_e_rv[start:stop]
    
    for o in orders:
        lab = 'o{} $\sigma$={:0.3f}m/s med(err)={:0.3f}m/s'.format(o,np.std(df_rv[o].values),np.median(df_e_rv[o].values))
        #lab = ''
        if plot_errorbar: 
            ax.errorbar(df_rv.index,df_rv[o].values,df_e_rv[o].values,lw=0,elinewidth=0.5,label=lab,
                        marker='o',barsabove=True,mew=0.5,capsize=2.,markersize=3)
        else: 
            ax.errorbar(df_rv.index,df_rv[o].values,np.zeros(len(df_rv)),lw=0,elinewidth=0.,label=lab,marker='o',markersize=3)
    ax.minorticks_on()
    ax.set_xlabel("Date [UT]")
    ax.set_ylabel("RV [m/s]")
    ax.grid(lw=0.5,alpha=0.5)
    ax.legend(fontsize=10,bbox_to_anchor=(1,1))
    ax.tick_params(axis='both',labelsize=8,pad=3)
    ax.tick_params(axis='x',labelsize=6,pad=3)

def plot_template(sps,spt,orders,interp_badpm=False,inst='HPF',plot=True,ax=None,plot_template=True,verbose=True,select_tellurics=True):
    """
    Main function to get serval template

    INPUT:
        sps - a list of Sp instances of all the spectra
        spt - highest signal to noise spectrum to be used for template
        interp_badpm
        select_tellurics - be able to select tellurics in an interactive mode

    OUTPUT:
        ST - the template
    """
    spt.verbose = False
    bjd,rv,e_rv,RV,e_RV = serval_help.calculate_pre_RVs(sps.splist,spt,orders=orders,plot=False,verbose=True,interp_badpm=False)
    print('#############')
    print("Interpolating")
    spt.interpolate_and_baryshift_orders(interp_badpm=False)
    print('#############')
    print("Generating template")
    ST = serval_help.generate_template(spt,sps,RV,orders=orders,plot=plot_template,inst=inst,select_tellurics=select_tellurics)
    return ST

def get_rvs_no_prervs(spok,spt,orders=[3,4,5,6,14,15,16,17,18],
                      template_filename='TEMPLATES/template.fits',npass=5,plot_rvs=False,
                      estimate_optimal_knots=True,calculate_activity=True,vref=0.,plot_CaIRT=True,
                      savedir_CaIRT='plots_ca_irt/',plot_template_rvs=False,savedir_template_rvs='0_template_rvs/'):
    """
    Get RVs without any pre-rvs
    """
    spt.interpolate_and_baryshift_orders(interp_badpm=False,vref=vref)
    ST = serval_help.generate_template(spt,
                                       spok,
                                       RV=0.*np.ones(len(spok.splist)),
                                       orders=orders,plot=False,
                                       inst="HPF",template_filename=template_filename,
                                       estimate_optimal_knots=estimate_optimal_knots,vref=vref)
    res = serval_help.calculate_rvs_from_final_template(spok.splist,
                                                        spt,
                                                        orders=orders,
                                                        inst='HPF',
                                                        verb=True,calculate_activity=calculate_activity,
                                                        vref=vref,
                                                        plot_CaIRT=plot_CaIRT,
                                                        savedir_CaIRT=savedir_CaIRT,
                                                        plot=plot_template_rvs,
                                                        savedir_template_rvs=savedir_template_rvs)
    _bjd,_rv,_snr,_e_rv,_RV,_e_RV,_vgrid,_chi2map,_res_activity = res
    if npass != 0:
        ordinal = lambda n: "%d%s"%(n,{1:"st",2:"nd",3:"rd"}.get(n if n<20 else n%10,"th"))
        for thisiter in range(1,int(npass)):
            print('#####################')
            print('Running {} iteration!!!'.format(ordinal(thisiter+1)))
            print('#####################')
            spt.interpolate_and_baryshift_orders(interp_badpm=False,vref=vref)
            ST = serval_help.generate_template(spt,
                                            spok,
                                            RV=-_RV,
                                            orders=orders,plot=False,
                                            inst="HPF",template_filename=template_filename,
                                            estimate_optimal_knots=estimate_optimal_knots,vref=vref)
            res = serval_help.calculate_rvs_from_final_template(spok.splist,
                                                                spt,
                                                                orders=orders,
                                                                inst='HPF',
                                                                verb=True,
                                                                calculate_activity=calculate_activity,
                                                                vref=vref,
                                                                plot_CaIRT=plot_CaIRT,
                                                                savedir_CaIRT=savedir_CaIRT,
                                                                plot=plot_template_rvs,
                                                                savedir_template_rvs=savedir_template_rvs)
            _bjd,_rv,_snr,_e_rv,_RV,_e_RV,_vgrid,_chi2map,_res_activity = res
    nbjd, nRV, nRV_err = bin_rvs_by_track(_bjd,_RV,_e_RV)
    # plot RVs
    if plot_rvs:
        plot_RVs(_bjd,tRV=_RV,tRV_err=_e_RV,nbjd=nbjd,nRV=nRV,nRV_err=nRV_err)
    return _bjd,_rv,_snr,_e_rv,_RV,_e_RV,nbjd,nRV,nRV_err,_vgrid,_chi2map,_res_activity


def get_RVs(sps,spt,orders,template_filename='TEMPLATES/template.fits',interp_badpm=False,inst='HPF',plot=True,ax=None,
            plot_template=False,verbose=True,title_prefix="",vref=0.):
    """
    Run the full SERVAL pipeline to get both pre-and template RVs

    INPUT:
        sps
        spt
        orders

    OUTPUT:
        return ST,df,nbjd,nRV,nRV_err,rv,e_rv,_rv,_e_rv
    """
    print('#############')
    print("PRE RVs")
    spt.verbose = False
    bjd,rv,e_rv,RV,e_RV = serval_help.calculate_pre_RVs(sps.splist,spt,orders=orders,plot=False,verbose=True,interp_badpm=False)
    print('#############')
    print("Interpolating")
    spt.interpolate_and_baryshift_orders(interp_badpm=False)
    print('#############')
    print("Generating template")
    ST = serval_help.generate_template(spt,sps,RV,orders=orders,plot=plot_template,inst=inst,template_filename=template_filename)
    print('#############')
    print("Calculating Template RVs")
    _bjd,_rv,_snr,_e_rv,_RV,_e_RV=serval_help.calculate_rvs_from_final_template(sps.splist,ST,orders=orders,inst=inst,verb=verbose,vref=vref)

    # Group by bin
    track_groups = group_tracks(bjd,threshold=0.05,plot=False)
    date = [str(i)[0:10] for i in utils.jd2datetime(_bjd)]
    df = pd.DataFrame(zip(date,track_groups,bjd,RV,e_RV,_RV,_e_RV),
                      columns=['date','track_groups','bjd','pRV','pRV_err','tRV','tRV_err'])
    g = df.groupby(['track_groups'])
    ngroups = len(g.groups)

    # Nightly bins
    nbjd, nRV, nRV_err = np.zeros(ngroups), np.zeros(ngroups), np.zeros(ngroups)
    for i, (source, idx) in enumerate(g.groups.items()):
        cut = df.ix[idx]
        #nRV[i], nRV_err[i] = wsem(cut.tRV.values,cut.tRV_err)
        nRV[i], nRV_err[i] = serval_help.weighted_average(cut.tRV.values,cut.tRV_err.values)
        nbjd[i] = np.mean(cut.bjd.values)
    if plot:
        title = "{} Pre RVs vs Template, orders #{}".format(title_prefix,str(orders))
        plot_RVs(df.bjd,df.pRV,df.pRV_err,df.tRV,df.tRV_err,nbjd,nRV,nRV_err,ax=ax,title=title)
    return ST,df,nbjd,nRV,nRV_err,rv,e_rv,_rv,_e_rv

    
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
        ax.errorbar(utils.jd2datetime(bjd),-pRV,pRV_err,marker="o",lw=0,elinewidth=0.5,
                    barsabove=True,mew=0.5,capsize=2,markersize=4,label=label1,alpha=0.2)
    if tRV is not None:
        #tRV -= np.mean(tRV)
        label2="Unbinned: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(tRV),np.median(tRV_err))
        ax.errorbar(utils.jd2datetime(bjd),tRV,tRV_err,marker="o",lw=0,elinewidth=0.5,
                    barsabove=True,mew=0.5,capsize=2,markersize=4,label=label2,alpha=0.4)
    if nRV is not None:
        #nRV -= np.mean(nRV)
        label3="Binned Track: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(np.std(nRV),np.median(nRV_err))
        ax.errorbar(utils.jd2datetime(nbjd),nRV,nRV_err,marker="h",lw=0,elinewidth=0.5,
                    barsabove=True,mew=0.5,capsize=2,markersize=6,label=label3,color='crimson')
    ax.grid(lw=0.5,alpha=0.5)
    ax.tick_params(axis="both",labelsize=16,pad=3)
    ax.minorticks_on()
    ax.set_ylabel("RV [m/s]",fontsize=20)
    ax.set_xlabel("Time [UT]",fontsize=20)
    ax.legend(loc="upper left",fontsize=8)
    ax.tick_params(axis='x',pad=3,labelsize=9)
    ax.set_title(title)

def plot_RV_overview(df,nbjd,nRV,nRV_err,rv,e_rv,start,stop,orders=[5,14,15,17],
                     savename='rv_overview.pdf',savefolder='rv_overview/'):
    """
    Plot an RV overview plot and save as a PDF
    
    EXAMPLE:
        dates = [ ('2018-02-25 00:00:00','2018-02-26 00:00:00'),
          ('2018-02-26 00:00:00','2018-02-27 00:00:00'),
          ('2018-03-02 00:00:00','2018-03-03 00:00:00'),
          ('2018-03-03 00:00:00','2018-03-04 00:00:00'),
          ('2018-03-04 00:00:00','2018-03-05 00:00:00'),
          ('2018-03-05 00:00:00','2018-03-06 00:00:00'),
          ('2018-03-06 00:00:00','2018-03-07 00:00:00') ] 
    start, stop = zip(*dates)
    res_5 = serval_plotting.get_RVs(spok2,spt2,orders=[5,14,15,17])
    ST,df,nbjd,nRV,nRV_err,rv,e_rv,_rv,_e_rv = res_5
    plot_RV_overview(df,nbjd,nRV,nRV_err,rv,e_rv,start,stop)
    """
    NROWS = len(start) + 1
    fig, ax = plt.subplots(nrows=NROWS,figsize=(15,20))
    plot_RVs(df.bjd,df.pRV,df.pRV_err,df.tRV,df.tRV_err,nbjd,nRV,nRV_err,ax=ax.flatten()[0])
    for i in range(len(start)):
        plot_rvs_by_order(df.bjd,rv,e_rv,orders=orders,
                          plot_errorbar=True,start=start[i],stop=stop[i],ax=ax.flatten()[i+1])
    [bx.set_ylabel('RV [m/s]',fontsize=8) for bx in ax]
    [bx.set_xlabel('Date',fontsize=8,labelpad=0) for bx in ax]
    fig.tight_layout()
    fig.subplots_adjust(right=0.7)
    utils.make_dir(savefolder)
    outname = savefolder + savename
    print("Saving to {}".format(outname))
    fig.savefig(outname,dpi=600) ; plt.close()


def get_RV_overview(sps,spt,orders,start,stop,inst='HPF',savename='rv_overview.pdf',savefolder='rv_overview/'):
    """
    Run SERVAL and save a master overview RV.pdf

    EXAMPLE:
    dates = [ ('2018-02-25 00:00:00','2018-02-26 00:00:00'),
          ('2018-02-26 00:00:00','2018-02-27 00:00:00'),
          ('2018-03-02 00:00:00','2018-03-03 00:00:00'),
          ('2018-03-03 00:00:00','2018-03-04 00:00:00'),
          ('2018-03-04 00:00:00','2018-03-05 00:00:00'),
          ('2018-03-05 00:00:00','2018-03-06 00:00:00'),
          ('2018-03-06 00:00:00','2018-03-07 00:00:00') ] 
    start, stop = zip(*dates)
    _=serval_plotting.get_RV_overview(spok2,spt,orders=[5,14,15,17],
                                      start=start,stop=stop,savename='20180405_skymask_terraspec_tellurics.pdf')
    """
    ST,df,nbjd,nRV,nRV_err,rv,e_rv,_rv,_e_rv = get_RVs(sps.splist,spt,orders,inst=inst,plot=False)
    plot_RV_overview(df,nbjd,nRV,nRV_err,rv,e_rv,start,stop,orders=orders,savename=savename,savefolder=savefolder)
    return ST,df,nbjd,nRV,nRV_err,rv,e_rv,_rv,_e_rv

def bin_rvs_by_track(bjd,RV,RV_err):
    """
    Bin RVs in an HET track

    INPUT:
        bjd
        RV
        RV_err

    OUTPUT:

    """
    track_groups = group_tracks(bjd,threshold=0.05,plot=False)
    date = [str(i)[0:10] for i in utils.jd2datetime(bjd)]
    df = pd.DataFrame(zip(date,track_groups,bjd,RV,RV_err),
                      columns=['date','track_groups','bjd','RV','RV_err'])
    g = df.groupby(['track_groups'])
    ngroups = len(g.groups)

    # track_bins
    nbjd, nRV, nRV_err = np.zeros(ngroups),np.zeros(ngroups),np.zeros(ngroups)
    for i, (source, idx) in enumerate(g.groups.items()):
        cut = df.loc[idx]
        #nRV[i], nRV_err[i] = wsem(cut.RV.values,cut.RV_err)
        nRV[i], nRV_err[i] = serval_help.weighted_average(cut.RV.values,cut.RV_err.values)
        nbjd[i] = np.mean(cut.bjd.values)
    return nbjd, nRV, nRV_err


def get_track_count_mask(bjd,threshold=0):
    """
    Get track mask with the number of observations done in a track
    """
    groups = group_tracks(bjd)
    counts = collections.Counter(groups)
    counts = np.array(counts.values())
    counts = np.array([counts[i] for i in groups])
    return counts > threshold

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



def plot_master_rv_overview(_bjd,_rv,_e_rv,_RV,_e_RV,orders=[3,4,5,6,14,15,16,17,18],
        title='',savename='master_rv_overview/rv_overview.png'):
    """
    Useful plot showing the master RVs and the RVs for all of the orders
    """
    NROWS = len(orders)+1
    fig, axx = plt.subplots(nrows=NROWS,sharex=True,figsize=(14,20),dpi=600)

    # All RVs
    tRV = _RV
    tRV_err = _e_RV
    nbjd, nRV, nRV_err = bin_rvs_by_track(_bjd,tRV,tRV_err)
    label1="{} RV: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format('All',np.std(tRV),np.median(tRV_err))
    axx[0].errorbar(utils.jd2datetime(_bjd),tRV,tRV_err,marker="o",lw=0,elinewidth=0.5,
                    barsabove=True,mew=0.5,capsize=2,markersize=4,label=label1,alpha=0.2)
    label2="{} Binned RV: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format('All',np.std(nRV),
                                                                             np.median(nRV_err))
    axx[0].errorbar(utils.jd2datetime(nbjd),nRV,nRV_err,marker="o",lw=0,elinewidth=0.5,
                    barsabove=True,mew=0.5,capsize=2,markersize=4,label=label2,alpha=0.9,color='firebrick')
    axx[0].minorticks_on()
    axx[0].grid(lw=0.5,alpha=0.5)
    axx[0].set_ylabel('RV [m/s]')

    # individual orders
    axx = np.array(axx)
    ylims = []
    for i, o in enumerate(orders):
        i+=1
        tRV = _rv[:,o]
        tRV_err = _e_rv[:,o]
        nbjd, nRV, nRV_err = bin_rvs_by_track(_bjd,tRV,tRV_err)
        label1="o={} RV: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(o,np.std(tRV),np.median(tRV_err))
        axx[i].errorbar(utils.jd2datetime(_bjd),tRV,tRV_err,marker="o",lw=0,elinewidth=0.5,
                        barsabove=True,mew=0.5,capsize=2,markersize=4,label=label1,alpha=0.3,color='forestgreen')
        label2="o={} Binned RV: $\sigma$={:0.2f}m/s, Med(errorbar)={:0.2f}m/s".format(o,np.std(nRV),
                                                                                 np.median(nRV_err))
        axx[i].errorbar(utils.jd2datetime(nbjd),nRV,nRV_err,marker="o",lw=0,elinewidth=0.5,
                        barsabove=True,mew=0.5,capsize=2,markersize=4,label=label2,alpha=0.9,color='firebrick')
        axx[i].minorticks_on()
        axx[i].grid(lw=0.5,alpha=0.5)
        axx[i].set_ylabel('RV [m/s]')
        ylims.append((np.min(tRV),np.max(tRV)))
        axx[i].tick_params(labelsize=10,pad=3)
    ymin = np.min(np.array(ylims)[:,0])
    ymax = np.max(np.array(ylims)[:,1])

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.05,right=0.7,top=0.98)
    [axx[i+1].set_ylim(ymin,ymax) for i in range(len(orders))]
    [axx[i].legend(fontsize=10,bbox_to_anchor=(1,1.05)) for i in range(NROWS)]

    fig.suptitle(title+' Orders = {}'.format(orders),y=0.99)

    axx[-1].set_xlabel('Date [UT]')

    fp = filepath.FilePath(savename)
    utils.make_dir(fp.directory)
    print("Saving file to {}".format(fp._fullpath))
    fig.savefig(fp._fullpath,dpi=600) ; plt.close()


def compare_telluric_sky_masks_to_template2(hf,o=15,si=0.,x_telluric=None,y_telluric=None,
        title='',savename='0_TEMPLATE_PLOTS/template.png',rv=0.,airwave=False,
        lines=None,bjd=None,cmap='coolwarm',alpha=0.5):
    """
    Compare templates to tellurics. Making the residual plot not sorted
    
    INPUT:
    
    OUTPUT:
        
    NOTES:
        - Need to be able to show the barycentric throw 
        - Can I keep track of which spectrum is causing which bad points ?
    """
    wmod = spec_help.redshift(hf['tstack_w'][o],ve=rv)
    ww = spec_help.redshift(hf['ww'][o],ve=rv)
    if airwave:
        print('Using air wavelengths')
        wmod = utils.vac2air(wmod,P=760.,T=20.)
        ww = utils.vac2air(ww,P=760.,T=20.)
    mod = hf['tstack_f'][o]
    ind = hf['tstack_ind'][o]
    ind0 = hf['tstack_ind0'][o]
    tellind = hf['tstack_tellind'][o]
    starind = hf['tstack_starind'][o]

    fig, axx = plt.subplots(figsize=(18,10),nrows=3,sharex=True)
    ax, bx, cx = axx[0], axx[1], axx[2]

    ax.plot(wmod[tellind==False],mod[tellind==False],label="Spectra: no tellurics",
            marker="o",lw=0,markersize=3,color="black",alpha=0.3)
    #ax.plot(wmod[ind],mod[ind],label="Spectra: no tellurics",
    #        marker="o",lw=0,markersize=3,color="black",alpha=0.3)
    ax.plot(wmod[starind==True],mod[starind==True],label="data: wmod[starind],mod[starind]",
            marker="o",lw=0,markersize=5,color="purple",alpha=0.3)
    ax.plot(ww,hf['ff'][o],lw=1)
    ax.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",
            color="green",markersize=5,label="Sigma Clipped")
    ax.set_ylabel('Counts')
    _ylim = ax.get_ylim()
    _xlim = ax.get_xlim()
    if lines is not None:
        ax.vlines(lines,ymin=_ylim[0],ymax=_ylim[1],color='orange')
        ax.set_xlim(_xlim[0],_xlim[1])

    _ax = ax.twinx()
    xlim = (wmod[0][0],wmod[0][-1])
    _ax.set_xlim(*xlim)
    _ax.set_ylim(0.9,1.0)
    cx.set_ylabel('Telluric [0-1]')
    # Old telluric mask
    telluric_mask = serval_help._get_telluric_mask_file()
    sky_mask = serval_help._get_sky_mask_file()
    _ax.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,
                     label='Telluric mask, one epoch')
    _ax.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask, one epoch')
    ax.legend(fontsize=10,bbox_to_anchor=(1.25,1))

    # bx
    bx.plot(wmod[tellind],mod[tellind],
            label="Spectra: tellurics or sky",marker="^",lw=0,
            markersize=5,color="firebrick",alpha=0.5,zorder=-100)
    bx.plot(ww,hf['ff'][o],lw=1)
    bx.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",
            color="green",markersize=5,label="Sigma Clipped")
    bx.set_xlabel('Wavelength [A]')
    bx.set_ylabel('Counts')
    bx.legend(fontsize=10,bbox_to_anchor=(1.25,1))
    _bx = bx.twinx()
    _bx.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,
                     label='Telluric mask, one epoch')
    _bx.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask one epoch')

    # Epochs / template to identify problem areas
    wmin = ww[0]
    wmax = ww[-1]
    colors = sns.color_palette(palette=cmap,n_colors=wmod.shape[0])
    #colors = sns.diverging_palette(255, 133, l=60, n=wmod.shape[0], center="dark")
    for i in range(wmod.shape[0]):
        x = wmod[i][(tellind[i]==False)&(starind[i]==False)]
        y = mod[i][(tellind[i]==False)&(starind[i]==False)]
        m = (x> wmin) & (x < wmax)
        xx = x[m]
        yy = scipy.interpolate.interp1d(ww,hf['ff'][o],fill_value='extrapolate')(xx)
        if bjd is None:
            cx.plot(xx,y[m]/yy+i*si,marker='.',alpha=alpha,markersize=10,lw=1,color=colors[i])#drawstyle='steps-mid')#label=label)
        if bjd is not None:
            cx.plot(xx,y[m]/yy+(bjd[i]/bjd[0]-1.)*5e3,marker='.',alpha=0.3,markersize=10,lw=1,color=colors[i])#drawstyle='steps-mid')#label=label)

    _cx = cx.twinx()
    #_cx.fill_between(df_telluric['w'].values,df_telluric['m'].values,alpha=0.2,color='orange',
    #                 label='Masked: Tellurics or Sky')
    _cx.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,
                     label='Telluric mask, one epoch')
    _cx.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask, one epoch')
    cx.set_xlabel('Wavelength [A]')
    cx.set_xlim(*xlim)
    _cx.set_ylim(0.9,1.0)
    if x_telluric is not None and y_telluric is not None:
        _cx.plot(x_telluric,y_telluric,color='orange',lw=1,alpha=0.8)

    cx.legend(fontsize=10,bbox_to_anchor=(1.25,1))
    
    for xx in [ax,bx,cx]:
        xx.grid(lw=0.5,alpha=0.5)
        xx.minorticks_on()
    for xx in [_ax,_bx,_cx]:
        xx.tick_params(pad=2,labelsize=12)
    fig.tight_layout()
    fig.suptitle('{} Order {}'.format(title,o),y=0.98,fontsize=24)     
    fig.subplots_adjust(hspace=0.005,wspace=0.005,right=0.75,top=0.95)
    
    fp = filepath.FilePath(savename)
    fp.add_suffix('_o_{}'.format(o))
    utils.make_dir(fp.directory)
    fig.savefig(fp._fullpath,dpi=600) ; plt.close()
    print('Saved to: {}'.format(fp._fullpath))

def compare_telluric_sky_masks_to_template(hf,x_telluric=None,y_telluric=None,
                                           o=15,title='',savename='0_TEMPLATE_PLOTS/template.png'):
    """
    Main template comparison plot: compare templates to tellurics.
    
    INPUT:
        hf - SERVAL hdf5 master result
        o - which order
    
    OUTPUT:
        plots a plot with template in blue and individual spectra on top
        also plots residuals
        
    NOTES:
        - Need to be able to show the barycentric throw 
        - Can I keep track of which spectrum is causing which bad points ?
    """
    wmod = hf['tstack_w'][o]
    mod = hf['tstack_f'][o]
    ind = hf['tstack_ind'][o]
    ind0 = hf['tstack_ind0'][o]
    tellind = hf['tstack_tellind'][o]
    starind = hf['tstack_starind'][o]

    fig, axx = plt.subplots(figsize=(18,10),nrows=3,sharex=True)
    ax, bx, cx = axx[0], axx[1], axx[2]

    # Ax
    ax.plot(wmod[tellind==False],mod[tellind==False],label="Tellurics",
            marker="o",lw=0,markersize=3,color="black",alpha=0.3)
    #ax.plot(wmod[ind],mod[ind],label="Spectra: no tellurics",
    #        marker="o",lw=0,markersize=3,color="black",alpha=0.3)
    ax.plot(wmod[starind==True],mod[starind==True],label="Starmask",marker="o",lw=0,markersize=5,color="purple",alpha=0.3)
    ax.plot(hf['ww'][o],hf['ff'][o],lw=1)
    ax.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",color="green",markersize=5,label="Sigma Clipped")
    ax.set_ylabel('Counts')
    _ax = ax.twinx()
    xlim = (wmod[0][0],wmod[0][-1])
    _ax.set_xlim(*xlim)
    _ax.set_ylim(0.9,1.0)
    cx.set_ylabel('Telluric [0-1]')
    # Old telluric mask
    telluric_mask = serval_help._get_telluric_mask_file()
    sky_mask = serval_help._get_sky_mask_file()
    _ax.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,label='Telluric mask, one epoch')
    _ax.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask, one epoch')
    ax.legend(fontsize=10,bbox_to_anchor=(1.25,1))

    # bx
    bx.plot(wmod[tellind],mod[tellind],label="Spectra: tellurics or sky",marker="^",lw=0,markersize=5,color="firebrick",alpha=0.5,zorder=-100)
    bx.plot(hf['ww'][o],hf['ff'][o],lw=1)
    bx.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",color="green",markersize=5,label="Sigma Clipped")
    bx.set_xlabel('Wavelength [A]')
    bx.set_ylabel('Counts')
    bx.legend(fontsize=10,bbox_to_anchor=(1.25,1))
    _bx = bx.twinx()
    _bx.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,
                     label='Telluric mask, one epoch')
    _bx.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask one epoch')

    # Epochs / template to identify problem areas
    wmin = hf['ww'][o][0]
    wmax = hf['ww'][o][-1]
    x = wmod[(tellind==False)&(starind==False)]
    y = mod[(tellind==False)&(starind==False)]
    inds = np.argsort(x)
    x = x[inds]
    y = y[inds]
    m = (x> wmin) & (x < wmax)
    xx = x[m]
    yy = scipy.interpolate.interp1d(hf['ww'][o],hf['ff'][o])(xx)
    sigma = np.nanstd(y[m]/yy)*1000.
    label = 'Spectra/Template: $\sigma$ = {:0.3f}ppt'.format(sigma)
    cx.plot(xx,y[m]/yy,marker='.',alpha=1.,markersize=4,color='k',lw=0,label=label)
    _cx = cx.twinx()
    #_cx.fill_between(df_telluric['w'].values,df_telluric['m'].values,alpha=0.2,color='orange',
    #                 label='Masked: Tellurics or Sky')
    _cx.fill_between(telluric_mask[:,0],telluric_mask[:,1],color=cp[0],alpha=0.1,
                     label='Telluric mask, one epoch')
    _cx.fill_between(sky_mask[:,0],sky_mask[:,1],color='firebrick',alpha=0.1,label='Skymask, one epoch')
    cx.set_xlabel('Wavelength [A]')
    cx.set_xlim(*xlim)
    _cx.set_ylim(0.9,1.0)
    if x_telluric is not None and y_telluric is not None:
        _cx.plot(x_telluric,y_telluric,color='orange',lw=1,alpha=0.8)

    cx.legend(fontsize=10,bbox_to_anchor=(1.25,1))
    
    for xx in [ax,bx,cx]:
        xx.grid(lw=0.5,alpha=0.5)
        xx.minorticks_on()
    for xx in [_ax,_bx,_cx]:
        xx.tick_params(pad=2,labelsize=12)
    fig.tight_layout()
    fig.suptitle('{} Order {}'.format(title,o),y=0.98,fontsize=24)     
    fig.subplots_adjust(hspace=0.005,wspace=0.005,right=0.75,top=0.95)
    
    fp = filepath.FilePath(savename)
    fp.add_suffix('_o_{}'.format(o))
    utils.make_dir(fp.directory)
    fig.savefig(fp._fullpath,dpi=600) ; plt.close()
    print('Saved to: {}'.format(fp._fullpath))

def master_rv_project(spall,spt,masterdir='0_RESULTS/',target='mytarget',
                      subdir='analysis',orders=[3,4,5,6,14,15,16,17,18],return_hf=True,
                      title='',overwrite=True,prervs=False,npass=5,
                      estimate_optimal_knots=True,calculate_activity=True,vref=0.,plot_CaIRT=True,
                      plot_template_rvs=False,onlyupdate=False):
    """
    Master RV project function. Generates a folder to save plots and results

    INPUT:
        spall - spall list 
        spt - template
        masterdir
        target
        subdir
        orders
        return_hf
        title
        overwrite

    OUTPUT:
        hf

    NOTES:
        hh = serval_plotting.master_rv_project(spall,spt,target='mytarget')
    """
    # Generate folders
    directory = os.path.join(masterdir,target,subdir)
    utils.make_dir(directory)
    fp = filepath.FilePath(directory)
    dir_html_base = os.path.join('..','data','targets')
    dir_html = os.path.join(dir_html_base,target,subdir)
    dir_html_cairt = os.path.join(dir_html_base,target,subdir,'plots_ca_irt')
    
    filename_master_hdf5         = os.path.join(fp._fullpath,'{}_results.hdf5'.format(target))
    filename_activity            = os.path.join(fp._fullpath,'{}_activity_results.pkl'.format(target))
    filename_template            = os.path.join(fp._fullpath,'{}_template.fits'.format(target))
    filename_rv_plot             = os.path.join(fp._fullpath,'{}_rvs.png'.format(target))
    filename_tess_photometry     = os.path.join(fp._fullpath,'{}_tess_photometry.png'.format(target))
    filename_rv_panel    = os.path.join(fp._fullpath,'{}_rv_activity_panel.png'.format(target))
    filename_rv_multiplot= os.path.join(fp._fullpath,'{}_rvs_multi.png'.format(target))
    filename_templateplot= os.path.join(fp._fullpath,'{}_template.png'.format(target))
    filename_chi2plot    = os.path.join(fp._fullpath,'{}_chi2map.png'.format(target))
    filename_rv_corner   = os.path.join(fp._fullpath,'{}_correlation_corner.png'.format(target))
    filename_corr_rv_dLW = os.path.join(fp._fullpath,'{}_correlation_rv_dLW.png'.format(target))
    filename_corr_rv_crx = os.path.join(fp._fullpath,'{}_correlation_rv_crx.png'.format(target))
    filename_corr_rv_irt1= os.path.join(fp._fullpath,'{}_correlation_rv_cairt1.png'.format(target))
    filename_corr_rv_irt1s= os.path.join(fp._fullpath,'{}_correlation_rv_cairt1_clipped.png'.format(target))
    filename_corr_rv_irt2= os.path.join(fp._fullpath,'{}_correlation_rv_cairt2.png'.format(target))
    filename_corr_rv_irt2s= os.path.join(fp._fullpath,'{}_correlation_rv_cairt2_clipped.png'.format(target))
    filename_corr_rv_irt3= os.path.join(fp._fullpath,'{}_correlation_rv_cairt3.png'.format(target))
    filename_corr_rv_irt3s= os.path.join(fp._fullpath,'{}_correlation_rv_cairt3_clipped.png'.format(target))
    filename_rv_vs_wave  = os.path.join(fp._fullpath,'{}_rv_vs_wave.png'.format(target))
    filename_vsini       = os.path.join(fp._fullpath,'{}_vsini.png'.format(target))
    filename_periodograms= os.path.join(fp._fullpath,'{}_periodograms.png'.format(target))
    filename_periodograms_all= os.path.join(fp._fullpath,'{}_periodograms_all.png'.format(target))
    filename_chi2plotDelta    = os.path.join(fp._fullpath,'{}_chi2mapDelta.png'.format(target))
    filename_chi2plotImage    = os.path.join(fp._fullpath,'{}_chi2mapImage.png'.format(target))
    filename_csv_unbin   = os.path.join(fp._fullpath,'{}_rv_unbin.csv'.format(target))
    filename_csv_bin     = os.path.join(fp._fullpath,'{}_rv_bin.csv'.format(target))
    filename_html        = os.path.join(fp._fullpath,'overview.html')
    filename_resfactor   = os.path.join(fp._fullpath,'{}_resfactors.csv'.format(target))
    filename_resfactor_panel= os.path.join(fp._fullpath,'{}_panel_resfactor.png'.format(target))
    filename_dLWo_panel     = os.path.join(fp._fullpath,'{}_panel_dLW.png'.format(target))
    savedir_CaIRT        = os.path.join(fp._fullpath,'plots_ca_irt/')
    savedir_template_rvs = os.path.join(fp._fullpath,'plots_template_rvs/')

    # Take out template from spectrum list
    spok = sspectrum.SSpectrumSet(np.array(spall.splist))
    #spok = sspectrum.SSpectrumSet(np.array(spall.splist)[~spall.df.istemplate.values])

    if os.path.isfile(filename_csv_unbin) and onlyupdate == True:
        _d = pd.read_csv(filename_csv_unbin)

        if np.array_equal(_d.filename.values,spok.df.filename.values):
            print('#############################')
            print('#############################')
            print('Onlyupdate==True, and {} spectra have already been extracted. Spok'.format(len(_d)))
            print('#############################')
            print('#############################')
            os.sys.exit()

    res = get_rvs_no_prervs(spok,spt,orders,template_filename=filename_template,
                                            plot_rvs=False,npass=npass,
                                            estimate_optimal_knots=estimate_optimal_knots,
                                            calculate_activity=calculate_activity,vref=vref,
                                            plot_CaIRT=plot_CaIRT,savedir_CaIRT=savedir_CaIRT,
                                            plot_template_rvs=plot_template_rvs,
                                            savedir_template_rvs=savedir_template_rvs)
    bjd,rv,snr,e_rv,RV,e_RV,nbjd,nRV,e_nRV,vgrid,chi2map,results_activity = res
    dLW = results_activity['dLW']
    e_dLW = results_activity['e_dLW']
    dLWo = results_activity['dLWo']
    e_dLWo = results_activity['e_dLWo']
    crx = results_activity['crx']
    e_crx = results_activity['e_crx']
    ln_order_center = results_activity['xo']
    ln_order_center_saving = '|'.join([str(i) for i in ln_order_center])
    irt1 = results_activity['irt1'][:,0]
    irt1a = results_activity['irt1a'][:,0]
    irt1b = results_activity['irt1b'][:,0]
    irt2 = results_activity['irt2'][:,0]
    irt2a = results_activity['irt2a'][:,0]
    irt2b = results_activity['irt2b'][:,0]
    irt3 = results_activity['irt3'][:,0]
    irt3a = results_activity['irt3a'][:,0]
    irt3b = results_activity['irt3b'][:,0]
    irt_ind1_v = results_activity['irt_ind1_v']
    irt_ind1_e = results_activity['irt_ind1_e']
    irt_ind2_v = results_activity['irt_ind2_v']
    irt_ind2_e = results_activity['irt_ind2_e']
    irt_ind3_v = results_activity['irt_ind3_v']
    irt_ind3_e = results_activity['irt_ind3_e']
    resfactor = results_activity['resfactor']
    results_activity['bjd'] = bjd
    results_activity['RV'] = RV
    results_activity['e_RV'] = e_RV
    results_activity['nbjd'] = nbjd
    results_activity['nRV'] = nRV
    results_activity['e_nRV'] = e_nRV
    results_activity['rv'] = rv
    results_activity['e_rv'] = e_rv
    results_activity['snr'] = snr

    ###########################################################################
    # Calculate RVs
    if prervs:
        print('################')
        print('Calculating PRERVS')
        print('################')
        pbjd,prv,pe_rv,pRV,pe_RV = serval_help.calculate_pre_RVs(spok.splist,spt,
                                                                 orders=orders,plot=False,
                                                                 verbose=True,
                                                                 interp_badpm=False)
    
    ############################################################################
    # Save results
    hf = serval_help.save_master_results_as_hdf5(spall,spt,rv,e_rv,RV,e_RV,orders,vgrid,chi2map,
                                                 dLW,e_dLW,dLWo,e_dLWo,crx,e_crx,ln_order_center,resfactor,
                                                 filename=filename_master_hdf5,return_hf=return_hf,
                                                 overwrite=overwrite)
    ############################################################################

    serval_help.plot_hf_variable_panel_rvs_for_orders(hf,'resfactor',orders=orders,savename=filename_resfactor_panel)
    serval_help.plot_hf_variable_panel_rvs_for_orders(hf,'dLWo',orders=orders,savename=filename_dLWo_panel)
    # Save results
    _f = open(filename_activity,'wb')
    pickle.dump(results_activity,_f)
    print('Saved activity to {}'.format(filename_activity))
    _f.close()
    #################
    # Save RVs as csv file
    if prervs == False:
        pRV = np.nan * np.ones(len(bjd))
        pe_RV = np.nan * np.ones(len(bjd))
    df_unbin = pd.DataFrame(zip(bjd,RV,e_RV,pRV,pe_RV,spok.df.sn18,spok.df.exptime,spok.df.berv,
                                spok.df.qprog,spok.df.airmass,spok.df.filename,dLW,e_dLW,crx,e_crx,
                                ln_order_center_saving,irt_ind1_v,irt_ind1_e,irt_ind2_v,irt_ind2_e,
                                irt_ind3_v,irt_ind3_e),
                            columns=['bjd','rv','e_rv','pre_rv','pre_e_rv','sn18','exptime','berv',
                                     'qprog','airmass','filename','dLW','e_dLW','crx','e_crx',
                                     'ln_order_center','irt_ind1','irt_ind1_e','irt_ind2','irt_ind2_e',
                                     'irt_ind3','irt_ind3_e'])

    df_bin   = pd.DataFrame(zip(nbjd,nRV,e_nRV),columns=['bjd','rv','e_rv'])
    df_unbin.to_csv(filename_csv_unbin,index=False)
    print('Saved CSV to: {}'.format(filename_csv_unbin))
    df_bin.to_csv(filename_csv_bin,index=False)
    print('Saved CSV to: {}'.format(filename_csv_bin))

    df_resfactor = pd.DataFrame(resfactor,columns=['o{}'.format(i) for i in range(28)])
    df_resfactor.to_csv(filename_resfactor,index=False)

    ############################################################################
    ############################################################################
    ############################################################################
    # Plots
    label = title + '\n{} Orders = {}'.format(spt.obj,orders)

    #################
    # 1. 1 panel RV plot
    plot_rv_df_panel(df_unbin,savename=filename_rv_panel,title=label)
    plot_rv_df_corner(df_unbin,savename=filename_rv_corner,title=label)

    #################
    # 1. 1 panel RV plot
    fig, ax = plt.subplots(dpi=600,figsize=(10,6))
    if prervs:
        # Should add pre-RVs here
        plot_RVs(bjd,tRV=RV,tRV_err=e_RV,nbjd=nbjd,nRV=nRV,nRV_err=e_nRV,ax=ax)
    else:
        plot_RVs(bjd,tRV=RV,tRV_err=e_RV,nbjd=nbjd,nRV=nRV,nRV_err=e_nRV,ax=ax)
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(filename_rv_plot,dpi=600) ; plt.close()
    print('Saved plot: {}'.format(filename_rv_plot))

    #################
    # Plot chi2map
    v = hf['rv/vgrid'][:]
    chi = hf['rv/chi2map'][:]
    plot_chi2maps(v,chi,targetname=target,savename=filename_chi2plot,orders=orders)
    plot_chi2mapsDeltaPlot(v,chi,targetname=target,savename=filename_chi2plotDelta,orders=orders)
    plot_chi2mapsImageGrid(v,chi,targetname=target,savename=filename_chi2plotImage,orders=orders)
    
    #################
    # 2. Multipanel RV plot
    plot_master_rv_overview(bjd,rv,e_rv,RV,e_RV,orders=orders,savename=filename_rv_multiplot,title=label)
    print('Saved plot: {}'.format(filename_rv_multiplot))
    
    #################
    # 3. Plot template plots
    for o in orders:
        compare_telluric_sky_masks_to_template(hf['template'],x_telluric=None,y_telluric=None,
                                               o=o,title=label,savename=filename_templateplot)
    #################
    # 4. Plot Correlation Plots
    ss = np.std(e_RV)
    mm = np.mean(e_RV)
    _m = e_RV < mm+4.*ss
    plot_correlation_plot(RV[_m],dLW[_m],e_RV[_m],e_dLW[_m],title=label+' dLW',savename=filename_corr_rv_dLW,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='dLW [1000 $\mathrm{m^2/s^2}$]'); 
    plot_correlation_plot(RV[_m],crx[_m],e_RV[_m],e_crx[_m],title=label+' CRX',savename=filename_corr_rv_crx,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CRX [m/s/Np]'); 
    plot_correlation_plot(RV[_m],irt_ind1_v[_m],e_RV[_m],irt_ind1_e[_m],title=label+' CaIRT1',savename=filename_corr_rv_irt1,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 
    plot_correlation_plot(RV[_m],irt_ind2_v[_m],e_RV[_m],irt_ind2_e[_m],title=label+' CaIRT2',savename=filename_corr_rv_irt2,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 
    plot_correlation_plot(RV[_m],irt_ind3_v[_m],e_RV[_m],irt_ind3_e[_m],title=label+' CaIRT3',savename=filename_corr_rv_irt3,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 

    _m = (e_RV < mm+4.*ss) & (np.abs(irt_ind1_v - np.nanmedian(irt_ind1_v)) < 3.*np.nanstd(irt_ind1_v))
    plot_correlation_plot(RV[_m],irt_ind1_v[_m],e_RV[_m],irt_ind1_e[_m],title=label+' CaIRT1',savename=filename_corr_rv_irt1s,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 
    _m = (e_RV < mm+4.*ss) & (np.abs(irt_ind2_v - np.nanmedian(irt_ind2_v)) < 3.*np.nanstd(irt_ind2_v))
    plot_correlation_plot(RV[_m],irt_ind2_v[_m],e_RV[_m],irt_ind2_e[_m],title=label+' CaIRT2',savename=filename_corr_rv_irt2s,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 
    _m = (e_RV < mm+4.*ss) & (np.abs(irt_ind3_v - np.nanmedian(irt_ind3_v)) < 3.*np.nanstd(irt_ind3_v))
    plot_correlation_plot(RV[_m],irt_ind3_v[_m],e_RV[_m],irt_ind3_e[_m],title=label+' CaIRT3',savename=filename_corr_rv_irt3s,
                      p=bjd,plabel='BJD',xlabel='RV [m/s]',ylabel='CaIRT Index'); 

    ind, = np.where(np.isfinite(rv[0]))
    l = hf['rv/ln_order_center'][:][0]
    plot_order_rvs_vs_wavelength(l,rv,e_rv,bjd,savename=filename_rv_vs_wave,title=label);

    lomb_scargle_plot.plot_multi_lomb_scargle(bjd,RV,e_RV,crx,e_crx,dLW,e_dLW,savename=filename_periodograms,title=label)
    lomb_scargle_plot.plot_multi_lomb_scargle7(bjd,RV,e_RV,crx,e_crx,dLW,e_dLW,irt_ind1_v,irt_ind1_e,irt_ind2_v,irt_ind2_e,
                                                         irt_ind3_v,irt_ind3_e,savename=filename_periodograms_all,title=label)
    hf.close() 
    print('Closed HDF5 database')

    ###########################################################################
    iframe_path = 'http://simbad.harvard.edu/simbad/sim-id?Ident={}'.format(target)
    # HTML
    print('Writing HTML string to {}'.format(filename_html))
    html_str = '<h1>{}</h1>'.format(target) + '\n'
    html_str += '<p><a href="{}">RVs Unbinned</a></p>'.format(os.path.join(dir_html,filename_csv_unbin.split(os.sep)[-1])) + '\n'
    html_str += '<p><a href="{}">RVs Binned</a></p>'.format(os.path.join(dir_html,filename_csv_bin.split(os.sep)[-1])) + '\n'
    html_str += '<div class="col-sm-9" class="float-left"><iframe src="{}" width="800px" height="700px"></iframe></div>'.format(iframe_path)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_rv_plot.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_tess_photometry.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_chi2plot.split(os.sep)[-1]),width=600)+'\n'
    html_str += '<h2>Correlations</h2>' + '\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_rv_corner.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_rv_panel.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_corr_rv_dLW.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_corr_rv_crx.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_rv_vs_wave.split(os.sep)[-1]),width=600)+'\n'
    html_str += '<h2>Periodograms</h2>' + '\n'
    #html_str += html_help.html_img(os.path.join(dir_html,filename_rv_multiplot.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_periodograms.split(os.sep)[-1]),width=600)+'\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_periodograms_all.split(os.sep)[-1]),width=600)+'\n'
    html_str += '<h2>Vsini</h2>' + '\n'
    html_str += html_help.html_img(os.path.join(dir_html,filename_vsini.split(os.sep)[-1]),width=600)+'\n'
    html_str += '<h2>Spectra</h2>' + '\n'
    for o in orders:
        html_str += html_help.html_img(os.path.join(dir_html,'{}_template_o_{}.png'.format(target,o)),width=600)+'\n'
    html_str += '<h2>First few Ca IRT spectra + measurements</h2>' + '\n'
    for i in range(1,5):
        filename_ca_irt   = '{}_000{}_ca_irt.png'.format(target,i)
    html_str += html_help.html_img(os.path.join(dir_html_cairt,filename_ca_irt),width=600)+'\n'
    html_str += '{}'.format(df_unbin.to_html()) + '\n'
    #print(html_str)
    with open(filename_html,'w') as f:
        f.write(html_str)
    ###########################################################################
    #return hf

def plot_epoch_div_template(hf,o,i,ax=None,airwave=True,rv=0.,plot=True,color=None,nbin=5,savename=None):
    """
    Plot epoch
    """
    #si = range(LEN)
    wmod = spec_help.redshift(hf['template/tstack_w'][o][i],ve=rv)
    ww = spec_help.redshift(hf['template/ww'][o],ve=rv)
    ff = hf['template/ff'][o]
    if airwave:
        wmod = utils.vac2air(wmod,P=760.,T=20.)
        ww = utils.vac2air(ww,P=760.,T=20.)
    sn18 = hf['rv/sn18'][:][i]
    mod = hf['template/tstack_f'][o][i]
    ind = hf['template/tstack_ind'][o][i]
    ind0 = hf['template/tstack_ind0'][o][i]
    tellind = hf['template/tstack_tellind'][o][i]
    starind = hf['template/tstack_starind'][o][i]
    filename = hf['rv/spectrum_basenames'][:][i]
    x = wmod[(tellind==False)&(starind==False)]
    y = mod[(tellind==False)&(starind==False)]
    wmin = np.nanmin(ww)
    wmax = np.nanmax(ww)
    m = (x> wmin) & (x < wmax)
    xx = x[m]
    yy = scipy.interpolate.interp1d(ww,ff,fill_value='extrapolate')(xx)
    if plot:
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,4))
        if color is None:
            ax.plot(xx,y[m]/yy,marker='.',alpha=0.3,markersize=2,lw=1,label='Residuals')#drawstyle='steps-mid')#label=label)
        else:
            ax.plot(xx,y[m]/yy,marker='.',alpha=0.3,markersize=10,lw=1,color=color,label='Residuals')#drawstyle='steps-mid')#label=label)
        df_bin = utils.bin_data(xx,y[m]/yy,nbin)
        ax.plot(df_bin.x,df_bin.y,marker='.',alpha=0.5,markersize=10,lw=1,color='green',label='NBin={}'.format(nbin))#drawstyle='steps-mid')#label=label)
        df_bin = utils.bin_data(xx,y[m]/yy,nbin*4)
        ax.plot(df_bin.x,df_bin.y,marker='.',alpha=0.8,markersize=10,lw=1,color='red',label='NBin={}'.format(nbin*4))#drawstyle='steps-mid')#label=label)
        ax.axhline(1,color='black')
        ax.legend(loc='upper right')
        _std = np.nanstd(y[m]/yy)
        ax.set_title('{}, SN18={:0.1f}, ExpSTD={:0.3f}ppt, STD={:0.3f}ppt'.format(filename,sn18,1000./sn18,1000.*_std),fontsize=16)
        ax.set_xlabel('Wavelength [A]',fontsize=12)
        ax.set_ylabel('Flux',fontsize=12)
        ax.set_ylim(1.-4.*_std,1+4.*_std)
        if savename is not None:
            fig.savefig(savename,dpi=600) ; plt.close()
            print('Saved to {}'.format(savename))
    return xx, y[m]/yy, filename

def plot_template_folded(hf,o,berv,rv=0,nrows=5,wcenters_good=None,wcenters_all=None,title=""):
    """
    Plot a template plot folded onto NROWS and plot wcenters also if provided
    
    INPUT:
        hf
        o
        berv
        nrows = 5
        wcenters - 

    OUTPUT:

    EXAMPLE:
    serval_plotting.plot_template_folded(hf,18,berv=spt.S.berv,nrows=5,wcenters=w_centers_good)
    """
    ww = spec_help.redshift(hf["template/ww"][:][o],0.,rv)
    ff = hf["template/ff"][:][o]
    ff, trend = spec_help.detrend_maxfilter_gaussian(ff,plot=False)

    # Mask tellurics 
    tellmask = np.loadtxt(serval_config.path_to_tellmask_file)
    tellmask_shifted = spec_help.redshift(tellmask[:,0],berv,rv)
    tellmask_int = (scipy.interpolate.interp1d(tellmask_shifted,tellmask[:,1])(ww)>0.01)*1.
    
    # Mask sky
    skymask = np.loadtxt(serval_config.path_to_skymask_file)#
    skymask_shifted = spec_help.redshift(skymask[:,0],berv,rv)
    skymask_int = (scipy.interpolate.interp1d(skymask_shifted,skymask[:,1])(ww)>0.01)*1.

    wt, mt = spec_help.broaden_binarymask(tellmask_shifted,tellmask[:,1],step=0.005,thres=0.01,v=np.ones(5))
    ws, ms = spec_help.broaden_binarymask(skymask_shifted,skymask[:,1],step=0.005,thres=0.01,v=np.ones(5))
    # Mask star
    #'../lib/masks/star_mask/gj699_stellarmask_LINET_01_20180816_1.53ms.txt')
    #starmask_int = (scipy.interpolate.interp1d(starmask[:,0],starmask[:,1])(ww)>0.01)*1.
    #all_mask_int = tellmask_int.astype(bool) | skymask_int.astype(bool)# | starmask_int.astype(bool)
    #all_mask_int = skymask_int.astype(bool)# | starmask_int.astype(bool)
    
    # PLOT
    fig, axx = plt.subplots(nrows=nrows,figsize=(18,8))
    
    # Full spectrum
    ax = axx.flatten()[0]
    ax.plot(ww,ff)
    if wcenters_all is not None:
        ax.vlines(wcenters_all,0.,1.,color='firebrick',lw=1.5,alpha=0.5)
    if wcenters_good is not None:
        ax.vlines(wcenters_good,0.,1.,color='green',lw=2.5,alpha=0.8)
    ax.fill_between(ww,tellmask_int,alpha=0.1,color='red',label="Telluric")
    ax.fill_between(ww,skymask_int,alpha=0.1,color='purple',label="Sky")
    utils.ax_apply_settings(axx.flatten()[0])
    
    delta = (ww.max()-ww.min())/(nrows-1.)
    for j in range(int(nrows-1.)):
        xx = axx.flatten()[j+1]
        wmin = ww.min()+delta*j
        wmax = ww.min()+delta*(j+1)
        mm = (ww>wmin)&(ww<wmax)
        xx.plot(ww[mm],ff[mm])
        xx.set_xlim(wmin,wmax)
        utils.ax_apply_settings(xx)
        if wcenters_all is not None:
            xx.vlines(wcenters_all,0.,1.05,color='firebrick',lw=1.5,alpha=0.5)
        if wcenters_good is not None:
            xx.vlines(wcenters_good,0.,1.05,color='green',lw=2,alpha=0.8)
            xx.set_ylim(ff[mm].min(),1.05)
            #xx.fill_between(ww,all_mask_int,alpha=0.1,color='red',label="Telluric & Sky Mask")
            xx.fill_between(ww,tellmask_int,alpha=0.1,color='red',label="Telluric")
            xx.fill_between(ww,skymask_int,alpha=0.1,color='purple',label="Sky")
            xx.set_ylabel("Flux")
            for i in range(hf['template/tstack_w'][o].shape[0]):#files_tell:
                wmod = spec_help.redshift(hf['template/tstack_w'][o][i],0.,rv)
                mod = hf['template/tstack_f'][o][i]
                ind = hf['template/tstack_ind'][o][i]
                ind0 = hf['template/tstack_ind0'][o][i]
                tellind = hf['template/tstack_tellind'][o][i]
                starind = hf['template/tstack_starind'][o][i]
                _tr = scipy.interpolate.interp1d(ww,trend,kind="cubic",fill_value="extrapolate")(wmod)
                xx.plot(wmod[tellind==False],mod[tellind==False]/_tr[tellind==False],marker=".",
                        markersize=3,lw=0,color="black",alpha=0.1)
                xx.plot(wmod[tellind==True],mod[tellind==True]/_tr[tellind==True],color="red",
                        marker=".",markersize=5,alpha=0.3,lw=0)
    xx.set_xlabel("Wavelength (A)")
    title = title + " rv="+str(rv)+"km/s"
    ax.set_title(title)
    return fig

def plot_lines_stacked(ww,ff,wcenters,width=0.5,ax=None,title=""):
    """
    Plot lines stacked on top of each other
    """
    if ax is None:
        fig, ax = plt.subplots(dpi=600)
    colors=utils.get_cmap_colors(cmap="coolwarm",p=wcenters)
    #ff = 
    for i, _w in enumerate(wcenters):
        wmin = _w - width
        wmax = _w + width
        mm = (ww>wmin)&(ww<wmax)
        ax.plot(ww[mm]-_w,ff[mm]-ff[mm][len(ww[mm])/2],color=colors[i])
    ax.axvline(x=0,ymin=0,ymax=1.,color="orange",alpha=0.5)
    ax.set_xlabel("Wavelength (A)")
    ax.set_ylabel("Flux")
    ax.set_title(title)




def plot_chi2maps(vgrid,chi2map,orders=[3,4,5,6,14,15,16,17,18],savename='',targetname='',cmap="coolwarm"):
    '''
    Function to plot chi2maps
    
    EXAMPLE:
        hf = serval_help.read_master_results_hdf5('/gpfs/group/cfb12/default/hpfrvs/software/hpfserval/site/hpfgto/docs/data/targets/GJ_4037/results/GJ_4037_results.hdf5')
        v = hf['rv/vgrid'][:]
        chi = hf['rv/chi2map'][:]
        plot_chi2maps(v,chi2map)
    '''
    if len(orders) <= 10:
        fig, axx = plt.subplots(nrows=2,ncols=5,figsize=(14,6),sharex=True)
    else:
        fig, axx = plt.subplots(nrows=np.ceil(len(orders)/5).astype(int),ncols=5,figsize=(21,12),sharex=True)

    for i,o in enumerate(orders):
        xx = axx.flatten()[i]
        colors = utils.get_cmap_colors(N=chi2map.shape[0],cmap=cmap)
        for j in range(chi2map.shape[0]):
            xx.plot(vgrid,chi2map[j][o]/np.max(chi2map[j][o]),color=colors[j])
            xx.set_title('o={}'.format(o))
            xx.set_xlabel('V [km/s]',labelpad=1)
            if i%5==0:
                xx.set_ylabel('$\chi^2$')
            utils.ax_apply_settings(xx)
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle(targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))

def plot_chi2mapsImageGrid(vgrid,chi2map,orders=[3,4,5,6,14,15,16,17,18],savename='',targetname='',cmap="jet"):
    '''
    Function to plot chi2maps
    
    EXAMPLE:
        hf = serval_help.read_master_results_hdf5('/gpfs/group/cfb12/default/hpfrvs/software/hpfserval/site/hpfgto/docs/data/targets/GJ_4037/results/GJ_4037_results.hdf5')
        v = hf['rv/vgrid'][:]
        chi = hf['rv/chi2map'][:]
        plot_chi2maps(v,chi2map)
    '''
    if len(orders) <= 10:
        fig, axx = plt.subplots(nrows=2,ncols=5,figsize=(14,6),sharex=True)
    else:
        fig, axx = plt.subplots(nrows=np.ceil(len(orders)/5).astype(int),ncols=5,figsize=(21,12),sharex=True)

    for i,o in enumerate(orders):
        c = chi2map[:,o]
        cnorm = np.array([(c[k]-c[k].min())/np.max(c[k]-c[k].min()) for k in range(len(c))])
        cnorm_mean = np.mean(cnorm,axis=0)
        xx = axx.flatten()[i]
        xx.imshow(cnorm-cnorm_mean,aspect="auto",cmap=cmap,origin="lower")
        xx.set_title('o={}'.format(o))
        xx.set_xlabel('V [km/s]',labelpad=1)
        if i%5==0:
            xx.set_ylabel('#obs')
        utils.ax_apply_settings(xx)
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle("Difference between scaled average chi2 profile: "+targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))

def plot_chi2mapsDeltaPlot(vgrid,chi2map,orders=[3,4,5,6,14,15,16,17,18],savename='',targetname='',cmap="coolwarm",xlim=None):
    '''
    Function to plot chi2maps
    
    EXAMPLE:
        hf = serval_help.read_master_results_hdf5('/gpfs/group/cfb12/default/hpfrvs/software/hpfserval/site/hpfgto/docs/data/targets/GJ_4037/results/GJ_4037_results.hdf5')
        v = hf['rv/vgrid'][:]
        chi = hf['rv/chi2map'][:]
        plot_chi2maps(v,chi2map)
    '''
    if len(orders) <= 10:
        fig, axx = plt.subplots(nrows=2,ncols=5,figsize=(14,6),sharex=True)
    else:
        fig, axx = plt.subplots(nrows=np.ceil(len(orders)/5).astype(int),ncols=5,figsize=(21,12),sharex=True)

    for i,o in enumerate(orders):
        c = chi2map[:,o]
        cnorm = np.array([(c[k]-c[k].min())/np.max(c[k]-c[k].min()) for k in range(len(c))])
        cnorm_mean = np.mean(cnorm,axis=0)
        xx = axx.flatten()[i]
        colors = utils.get_cmap_colors(N=len(cnorm),cmap=cmap)
        for j in range(len(c)):
            xx.plot(vgrid,cnorm[j]-cnorm_mean,color=colors[j])
        xx.set_title('o={}'.format(o))
        xx.set_xlabel('V [km/s]',labelpad=1)
        if i%5==0:
            xx.set_ylabel('$\Delta\chi^2$')
        if xlim is not None:
            xx.set_xlim(xlim[0],xlim[1])
            m = (vgrid > xlim[0]) & (vgrid < xlim[1])
            ymin = np.min(cnorm[:,m]-cnorm_mean[m])
            ymax = np.max(cnorm[:,m]-cnorm_mean[m])
            xx.set_ylim(ymin*1.1,ymax*1.1)
        utils.ax_apply_settings(xx)
    fig.subplots_adjust(hspace=0.3,wspace=0.3,top=0.9)
    if targetname!='':
        fig.suptitle("Difference between scaled average chi2 profile: "+targetname,y=0.98)
    if savename !='':
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))



def plot_correlation_plot(x,y,xerr,yerr,p=None,plabel='',title='',xlabel='',ylabel='',savename=None,cmap='Spectral'):
    """
    
    EXAMPLE:
        plot_correlation_plot(df.rv.values,df.dLW.values,df.e_rv.values,df.e_dLW.values,
                      p=df.bjd.values,plabel='BJD',xlabel='RV [m/s]',ylabel='dLW [1000 $\mathrm{m^2/s^2}$]')
        plot_correlation_plot(df.rv.values,df.dLW.values,df.e_rv.values,df.e_dLW.values,
                      p=df.bjd.values,plabel='BJD',xlabel='RV [m/s]',ylabel='dLW [1000 $\mathrm{m^2/s^2}$]')
    """
    fig, ax = plt.subplots(dpi=600)
    if p is not None:
        colors = utils.get_cmap_colors(cmap=cmap,p=p)
        for i in range(len(x)):
            ax.errorbar(x[i],y[i],yerr=yerr[i],xerr=xerr[i],marker='o',
                        elinewidth=0.5,capsize=4,mew=0.5,lw=0,color=colors[i])
        cbar, cx = utils.ax_add_colorbar(ax,p,cmap=cmap)
        cx.ax.set_ylabel(plabel)
    else:
         ax.errorbar(x,y,yerr=yerr,xerr=xerr,marker='o',
                     elinewidth=0.5,capsize=4,mew=0.5,lw=0)   
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    mm = np.isfinite(x) & np.isfinite(y)
    try:
        rho, p = scipy.stats.pearsonr(x[mm],y[mm])
    except Exception as e:
        print(e)
        rho, p = -99.,-99.
    ax.set_title('{}, rho={:0.3f}, p={:0.3f}'.format(title,rho,p))
    if savename is not None:
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))
    return fig

def plot_order_rvs_vs_wavelength(ln_order_center,rv,e_rv,bjd,cmap='Spectral',savename=None,title=''):
    """
    Plot a RV per order vs wavelength plot
    
    EXAMPLE:
        hf = serval_help.read_master_results_hdf5('/gpfs/group/cfb12/default/hpfrvs/software/hpfserval/site/hpfgto/docs/data/targets/GJ_1111/results/GJ_1111_results.hdf5')
        rv = hf['rv/rv'][:]
        e_rv = hf['rv/e_rv'][:]
        ind, = np.where(np.isfinite(rv[0]))
        l = hf['rv/ln_order_center'][:][0]
        bjd = hf['rv/bjd'][:]
        reload(serval_plotting)
        serval_plotting.plot_order_rvs_vs_wavelength(l,rv,e_rv,bjd)
    """
    ind, = np.where(np.isfinite(rv[0]))
    fig, ax = plt.subplots(dpi=600)
    colors = utils.get_cmap_colors(cmap,p=bjd)
    for i in range(rv.shape[0]):
        ax.errorbar(np.exp(ln_order_center[ind]),rv[i][ind]-np.mean(rv[i][ind]),yerr=e_rv[i][ind],elinewidth=0.5,mew=0.5,capsize=4,color=colors[i],marker='o',lw=0.5)
    ax.set_xlabel('Wavelength [A]')
    ax.set_ylabel('RV - mean(RV) [m/s]')
    cbar, cx = utils.ax_add_colorbar(ax,p=bjd,cmap=cmap)
    cx.ax.set_ylabel('BJD')
    ax.set_title(title)
    if savename is not None:
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to: {}'.format(savename))
    return fig

def plot_rv_df_panel(df,savename='',title=''):
    '''
    Example:
        df = pd.read_csv('../site/hpfgto/docs/data/targets/AD_Leo/results_20200118_start20180901_vref124/AD_Leo_rv_unbin.csv')
        plot_rv_df_panel(df)
    '''
    fig, axx = plt.subplots(nrows=6,sharex=True,dpi=600,figsize=(12,12))

    axx[0].set_title(title)
    axx[0].errorbar(df.bjd,df.rv,np.abs(df.e_rv),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[0].set_ylabel('RV [m/s]',fontsize=10)

    axx[1].errorbar(df.bjd,df.dLW,np.abs(df.e_dLW),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[1].set_ylabel('dLW',fontsize=10)

    axx[2].errorbar(df.bjd,df.crx,np.abs(df.e_crx),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[2].set_ylabel('CRX',fontsize=10)

    axx[3].errorbar(df.bjd,df.irt_ind1,np.abs(df.irt_ind1_e),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[3].set_ylabel('CaIRT 1',fontsize=10)

    axx[4].errorbar(df.bjd,df.irt_ind2,np.abs(df.irt_ind2_e),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[4].set_ylabel('CaIRT 2',fontsize=10)

    axx[5].errorbar(df.bjd,df.irt_ind3,np.abs(df.irt_ind3_e),marker='o',capsize=4,elinewidth=0.5,mew=0.5,lw=0,markersize=4)
    axx[5].set_ylabel('CaIRT 3',fontsize=10)
    axx[5].set_xlabel('BJD',fontsize=10)

    for xx in axx:
        xx.grid(lw=0.3,alpha=0.3)
    fig.subplots_adjust(hspace=0.05)   
    fig.subplots_adjust(right=0.97,left=0.08,top=0.95,bottom=0.05,hspace=0.05)
    if savename!='':
        print('Saved to: {}'.format(savename))
        fig.savefig(savename,dpi=600) ; plt.close()

def plot_rv_df_corner(df,savename='',title=''):
    """
    Plot a corner plot with errorbars, to check for correlations
    """
    NN = 6
    fig, axx = plt.subplots(nrows=NN,ncols=NN,dpi=600,figsize=(12,12))
    fig.suptitle(title)

    labels_values = ['rv','dLW','crx','irt_ind1','irt_ind2','irt_ind3']
    labels_errors = ['e_rv','e_dLW','e_crx','irt_ind1_e','irt_ind2_e','irt_ind3_e']

    # plot histograms
    for i in range(NN):
        x = df[labels_values[i]]
        m = np.isfinite(x)
        axx[i,i].hist(x[m])
        axx[i,i].set_title(labels_values[i])
        axx[i,i].tick_params(pad=1,labelsize=6)

    # remove upper corner
    for i in range(0,NN-1):
        for j in range(i+1,NN):
            axx[i,j].axes.set_visible(False)

    # plot lower corner
    for i in range(1,NN): # row
        for j in range(0,i): #column
            x = df[labels_values[j]]
            y = df[labels_values[i]]
            xerr = df[labels_errors[j]]
            yerr = df[labels_errors[i]]
            axx[i,j].errorbar(x,y,yerr=np.abs(yerr),xerr=np.abs(xerr),mew=0.5,capsize=4,marker='o',lw=0,elinewidth=0.5,markersize=3)
            axx[i,j].tick_params(pad=1,labelsize=6)       

    fig.subplots_adjust(right=0.95,left=0.05,top=0.95,bottom=0.05,hspace=0.1,wspace=0.1)
    if savename!='':
        print('Saving to: {}'.format(savename))
        fig.savefig(savename,dpi=600) ; plt.close()
