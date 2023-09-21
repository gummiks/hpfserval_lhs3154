import cubicSpline
from calcspec import barshift, redshift, calcspec
from matplotlib.widgets import SpanSelector
from airtovac import airtovac
import serval_config
import serval_help
import target
import os
import read_spec
import numpy as np
import matplotlib.pyplot as plt
from serval_config import flag
import logger
logger = logger.logger()
from scipy import interpolate
import pandas as pd
import os
import spectrum_widgets
import sys
import utils

class SSpectrum(object):
    """
    Working with SERVAL spectra
    """
    pmin = serval_config.pmin
    pmax = serval_config.pmax
    ptmin = serval_config.ptmin
    ptmax = serval_config.ptmax
    ntpix = serval_config.ntpix
    #pixx = serval_config.pixx
    pixxx = serval_config.pixxx
    osize = serval_config.osize
    verbose = serval_config.verbose
    ofac = serval_config.ofac
    orders = serval_config.orders
    nord = serval_config.iomax

    def __init__(self,filename,targetname,inst,read_data=True,jd_utc=None,verbose=True):
        """
        jd_utc can be used to overwrite jd_midpoint
        """
        self.filename = filename
        self.basename = filename.split(os.sep)[-1]
        #self.target = targ.getServalTarget(targetname,verbose=verbose)
        self.target = target.Target(targetname)
        self.obj = targetname
        self.date_str = serval_help.extract_datetime_from_array([filename])[0]
        
        # Read in spectrum
        self.S = read_spec.Spectrum(filename,
                                    inst=inst,
                                    drs=True,
                                    verb=verbose,
                                    pfits=2,
                                    wlog=False,
                                    fib=serval_config.fib,
                                    targ=self.target,
                                    jd_utc=jd_utc)
        if read_data: self.S.read_data()
            
        # Misc orders and arrays to set up
        self.telluric_mask = serval_help._get_telluric_mask_file()
        self.sky_mask = serval_help._get_sky_mask_file()
        self.star_mask = serval_help._get_stellar_mask_file()
        
        # initializing interpolation matrices
        self.ww = np.ones((self.nord,self.osize))
        self.ff = np.zeros((self.nord,self.osize))
        self.ee = np.zeros((self.nord,self.osize))
        self.bb = np.zeros((self.nord,self.osize), dtype=int)
        self.nn = np.zeros((self.nord,self.osize))

        # for post 3 -coadding
        self.nk = int(self.osize / 4 * self.ofac)
        self.wk = serval_help.nans((self.nord,self.nk))
        self.fk = serval_help.nans((self.nord,self.nk))
        self.ek = serval_help.nans((self.nord,self.nk))
        self.bk = np.zeros((self.nord,self.nk))
        self.kk = [[0]]*self.nord
        self.norm = np.ones((self.nord))
        self.template = False
        
        # Not used here, used when looping over spectra
        #self.rv, self.e_rv = nans((2, self.nspec, self.nord))
        #self.RV, self.e_RV = nans((2, self.nspec))

    def __repr__(self):
        if self.template is False:
            return 'SSpectrum(filename="{}", targetname="{}") with SN18={:0.1f}'.format(self.basename,
                                                             self.obj,self.S.sn18)
        else:
            return 'STSpectrum(filename="{}", targetname="{}") with SN18={:0.1f}'.format(self.basename,
                                                             self.obj,self.S.sn18)

    def get_order(self,o,baryshifted=False,berv=None,remove_badvalues=True):
        """
        Get orders
        
        INPUT:
            o: order number
            baryshifted:
            remove_badvalues: return only good values (take out outliers/nans)
        
        OUTPUT:
            w: wavelength
            f: flux
        """
        #assert o in self.orders
        if baryshifted is True:
            if berv is None:
                berv = self.S.berv
            w = barshift(self.S.w[o,:],berv) 
        else:
            w = self.S.w[o,:]
        f = self.S.f[o,:]
        s = self.S.f_sky[o,:]
        e = self.S.e[o,:]
        if remove_badvalues:
            pixx, idx = self._get_valid_template_range(o)
            w = w[idx]
            f = f[idx]
            e = e[idx]
            s = s[idx]
        return w, f, e, s

    def plot_interpolated_order(self,o=None,ax=None):
        """
        """
        if ax is None: self.fig, self.ax = plt.subplots(figsize=(12,8))
        else: self.ax = ax
        ww,kk,ff,ee,bb = self.interpolate_and_baryshift_order(o=o)
        self.plot_orders(o=o,ax=self.ax,baryshift=False)
        self.ax.plot(ww,ff,color="orange",label="Interpolated spectrum with tellurics",alpha=0.7)
        self.ax.plot(ww[bb==0],ff[bb==0],color="green",label="Interpolated spectrum without tellurics",alpha=0.8)
        self.ax.legend(loc="upper left",fontsize=14)
           
    def plot_orders(self,o=None,ax=None,bx=None,plot_tellurics=True,baryshift=True,berv=None,remove_badvalues=True,plot_sky_lines=True):
        """Plot order o, or all orders"""
        if ax is None: self.fig, self.ax = plt.subplots(figsize=(12,8))
        else: self.ax = ax
        if baryshift: 
            title = "Barycentric shifted"
        else: 
            title = "Not barycentric shifted"
        if o is None:
            for oo in self.orders:
                w, f, e, s = self.get_order(oo,baryshifted=baryshift,berv=None,remove_badvalues=remove_badvalues)
                if any(np.isfinite(f)):
                    self.ax.errorbar(w,f,yerr=e,label="Order {} star".format(oo),lw=1.,marker="o",elinewidth=0.1,markersize=0.5)
                    self.ax.plot(w,s,label="Order {} sky".format(oo),lw=1.,marker="o",markersize=0.5)
        else:
            w, f, e, s = self.get_order(o,baryshifted=baryshift,berv=None,remove_badvalues=remove_badvalues)
            self.ax.errorbar(w,f,yerr=e,label="Order {} sky subtracted".format(o),lw=1.,marker="o",elinewidth=0.1,markersize=0.5)
            self.ax.errorbar(w,f+s,yerr=e,label="Order {} not sky subtracted".format(o),lw=1.,marker="o",elinewidth=0.1,markersize=0.5)
            if bx is None:
                self.bx = self.ax.twinx()
            else:
                self.bx = bx
            self.bx.plot(w,s,label="Order {} sky".format(o),lw=1.,marker="o",markersize=0.5,color='crimson')
            self.bx.margins(y=0.8)
            ylim = self.bx.get_ylim()
            self.bx.set_ylim(ylim[0]*0.25,ylim[1]*4)
            self.bx.legend(loc='upper right',fontsize=12)

        self.ax.set_xlabel("Wavelength")
        self.ax.set_ylabel("Flux")
        self.ax.set_title(title)
        ylim = self.ax.get_ylim()

        if berv is None:
            berv = self.S.berv
        if plot_tellurics is True:
            xlim = self.ax.get_xlim()
            self.ax.fill_between(barshift(self.telluric_mask[:,0],berv),self.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="navy")
            self.ax.set_xlim(xlim[0],xlim[1])
        if plot_sky_lines is True:
            xlim = self.ax.get_xlim()
            self.ax.fill_between(barshift(self.sky_mask[:,0],berv),self.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
            self.ax.set_xlim(xlim[0],xlim[1])
        self.ax.legend(loc="upper left",fontsize=12)

    def get_telluric_mask(self):
        return self.telluric_mask[:,0], self.telluric_mask[:,1]

    def _get_valid_template_range(self,o):
        """
        To get valid indices, see line 1261 in Serval
        
        OUTPUT:
            pixx = array([   0,    1,    2, ..., 3697, 3698, 3699])
            idx = array([ 200,  201,  202, ..., 3897, 3898, 3899])
        """
        pixx = np.where((np.isfinite(self.S.w) & np.isfinite(self.S.f) & np.isfinite(self.S.e) & (self.S.e > 0.) & (self.S.bpmap[o,:]==0))[o,self.ptmin:self.ptmax])[0]
        #pixx = np.where((np.isfinite(self.S.w) & np.isfinite(self.S.f) & np.isfinite(self.S.e) & (self.S.bpmap[o,:]==0))[o,self.ptmin:self.ptmax])[0]
        #pixx = np.where((np.isfinite(self.S.w) & np.isfinite(self.S.f) & np.isfinite(self.S.e))[o,self.ptmin:self.ptmax])[0]
        idx = pixx + self.ptmin
        return pixx, idx

    def plot_template(self,o,ax=None,plot_baryshifted=False,select_tellurics=True):
        if hasattr(self,'tstack_w') is False:
            print("Have to generate template first!")
            sys.exit()
        wmod = self.tstack_w[o]
        mod = self.tstack_f[o]
        ind = self.tstack_ind[o]
        tellind = self.tstack_tellind[o]
        ind0 = self.tstack_ind0[o]
        if ax is None:
            fig, ax = plt.subplots()
        if plot_baryshifted:
            print('Plotting baryshifted')
            ax.plot(wmod[ind<ind0],mod[ind<ind0],lw=0,marker="D",color="green",markersize=10,label="clipped: wmod[ind<ind0],mod[ind<ind0]")
            ax.plot(wmod[ind],mod[ind],label="data: wmod[ind],mod[ind]",marker="o",lw=1,markersize=3,color="black",alpha=0.3)
            ax.plot(wmod[tellind],mod[tellind],label="atm: wmod[tellind],mod[tellind]",marker="o",lw=0,markersize=3,color="firebrick")
            ax.plot(self.wk[o],self.fk[o],label="Knots: (wk[o],fk[o])",alpha=0.2,marker="o",markersize=2,color="chocolate")
            ax.plot(self.ww[o],self.ff[o],label="Template: (ww[o],ff[o])")
        else:
            print('Plotting in telluric frame')
            ax.plot(barshift(wmod[ind<ind0],-self.S.berv),mod[ind<ind0],lw=0,marker="D",color="green",markersize=10,label="clipped: wmod[ind<ind0],mod[ind<ind0]")
            ax.plot(barshift(wmod[ind],-self.S.berv),mod[ind],label="data: wmod[ind],mod[ind]",marker="o",lw=1,markersize=3,color="black",alpha=0.3)
            ax.plot(barshift(wmod[tellind],-self.S.berv),mod[tellind],label="atm: wmod[tellind],mod[tellind]",marker="o",lw=0,markersize=3,color="firebrick")
            ax.plot(barshift(self.wk[o],-self.S.berv),self.fk[o],label="Knots: (wk[o],fk[o])",alpha=0.2,marker="o",markersize=2,color="chocolate")
            ax.plot(barshift(self.ww[o],-self.S.berv),self.ff[o],label="Template: (ww[o],ff[o])")
        # dummy plot
        #ax.plot(spt.S.w[o],spt.S.f[o],label='spt.S.w,spt.S.f',lw=0)

        cx = ax.twiny()
        #cx.set_xlim(
        #cx.set_xlim(0,serval_config.npix)
        #x2 = np.arange(serval_config.ptmin,serval_config.ptmax)
        x2 = np.arange(0,serval_config.npix)
        cx.plot(x2,x2,lw=0)#dummy plot
        cx.minorticks_on()

        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        ax.set_xlim(xlim)
        # Baryshifted
        #ax.fill_between(barshift(spt.telluric_mask[:,0],spt.S.berv),spt.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="k")
        # Not baryshifted
        if plot_baryshifted:
            ax.fill_between(barshift(self.telluric_mask[:,0],self.S.berv),self.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="navy")
            ax.fill_between(barshift(self.sky_mask[:,0],self.S.berv),self.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
        else:
            ax.fill_between(self.telluric_mask[:,0],self.telluric_mask[:,1]*ylim[1],alpha=0.2,label="Telluric mask",color="k")
            ax.fill_between(self.sky_mask[:,0],self.sky_mask[:,1]*ylim[1],alpha=0.2,label="Sky mask",color="orange")
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
            return span
    
    def interpolate_and_baryshift_order(self,o,berv=None,verbose=True,interp_badpm=False,vref=0.):
        """
        Interpolate over the order, and return the baryshifted order
        
        INPUT:
        o - order
        
        OUTPUT:
        ww - interpolated wavelengths, baryshifted
        kk - intermediate differentials from spline_cv
        ff - interpolated flux
        ee - interpolated errors
        
        EXAMPLE:
            # Plot a plot comparing spectrum and telluric mask baryshifted and not baryshifted:
                fig, ax = plt.subplots()
                S.plot_orders(o=56,ax=ax)
                w,k,f,e,b = S.interpolate_and_baryshift_order(o=56)
                ax.plot(w,b*(1./8)*ax.get_ylim()[1],alpha=0.5,label="Telluric Mask: resampled, baryshifted")
                ax.plot(w,f,label="Spectrum: Resampled, baryshifted")
                ax.legend(loc="upper left")

        NOTES:
        - VERIFIED GOOD
        """
        pixx, idx = self._get_valid_template_range(o)
        bb = np.zeros(self.osize, dtype=int) # flagged samples in array
        
        if berv is None:
            berv = self.S.berv
        #kktmp = serval_config.spline_cv(pixx,barshift(self.S.w[o,idx],berv))
        kktmp = serval_config.spline_cv(pixx,redshift(self.S.w[o,idx],vo=berv,ve=vref))
        ww = serval_config.spline_ev(self.pixxx, kktmp)
        #kk = serval_config.spline_cv(barshift(self.S.w[o,idx],berv), self.S.f[o,idx])
        kk = serval_config.spline_cv(redshift(self.S.w[o,idx],vo=berv,ve=vref), self.S.f[o,idx])
        ff = serval_config.spline_ev(ww,kk)
        #kktmp = serval_config.spline_cv(barshift(self.S.w[o,idx],berv), self.S.e[o,idx])
        kktmp = serval_config.spline_cv(redshift(self.S.w[o,idx],vo=berv,ve=vref), self.S.e[o,idx])
        #ee = serval_config.spline_ev(ww, kktmp) # can give negative errors

        ############??
        self.S.read_data()
        # all of these are True, even masked pixels, idx takes care of that
        ind = self.S.bpmap[o,idx] == 0  # let out zero errors, interpolate over
        ind[0] = True 
        ind[-1] = True
        #if verbose: print("len(ind)",np.sum(ind))
        ############??
        # Can error out on order 13,16 in GJ699 data
        # - use serval_config.spline_ev instead ?
        # CHANGED MARCH 21th 2018, to match with Linux, 
	# ORIGINAL:
        #ee = interpolate.interp1d(barshift(self.S.w[o,idx][ind],berv),self.S.e[o,idx][ind])(ww)
	# Trying this 20190605, as pre-rv was having error:
        #ee = interpolate.interp1d(barshift(self.S.w[o,idx][ind],berv),self.S.e[o,idx][ind], fill_value="extrapolate")(ww)
        ee = interpolate.interp1d(redshift(self.S.w[o,idx][ind],vo=berv,ve=vref),self.S.e[o,idx][ind], fill_value="extrapolate")(ww)
        ###########
        
        if interp_badpm:
            print("interpolating bad pixel map")
            _x_bb = np.arange(serval_config.npix)
            _xx_bb = np.linspace(_x_bb[0],_x_bb[-1],serval_config.osize)
            tmp_bb_interp_mask = serval_help.bpmap_interpolate(xx=_xx_bb,x=_x_bb,bpmap=self.S.bpmap[o])>0.01
            bb[tmp_bb_interp_mask] |= flag.nan

        # Flag bad samples that have tellurics
        #####
        #tmp_resampled_unshifted_mask_telluric = serval_help.telluric_mask_interpolate(barshift(ww,-self.S.berv))>0.01
        tmp_resampled_unshifted_mask_telluric = serval_help.telluric_mask_interpolate(redshift(ww,vo=-self.S.berv,ve=-vref))>0.01
        bb[tmp_resampled_unshifted_mask_telluric] |= flag.atm # flag all 0s as 8, bb is a picketfence
        # flag bad samples that have sky lines
        #tmp_resampled_unshifted_mask_sky = serval_help.sky_mask_interpolate(barshift(ww,-self.S.berv))>0.01
        tmp_resampled_unshifted_mask_sky = serval_help.sky_mask_interpolate(redshift(ww,vo=-self.S.berv,ve=-vref))>0.01
        bb[tmp_resampled_unshifted_mask_sky] |= flag.sky # flag all 0s as 8, bb is a picketfence
        #tmp_resampled_unshifted_mask_star = serval_help.stellar_mask_interpolate(barshift(ww,-self.S.berv))>0.01
        tmp_resampled_unshifted_mask_star = serval_help.stellar_mask_interpolate(redshift(ww,vo=-self.S.berv,ve=-vref))>0.01
        bb[tmp_resampled_unshifted_mask_star] |= flag.star # flag all 0s as 8, bb is a picketfence
        num_tellurics = np.sum(bb==flag.atm)
        num_sky = np.sum(bb==flag.sky)
        num_star = np.sum(bb==flag.star)
        if self.verbose: 
            logger.info("Flagged {}, {}, and {} / {} resampled points as tellurics, sky, star".format(num_tellurics,num_sky,num_star,len(bb)))
        # When errors are 0, flag as 1
        bb[ee<=0] = flag.nan
        # Flag negative flux values
        bb[ff<0] = flag.neg
        #if self.verbose: print("Flagged {}/{} resampled points with errors < 0".format(np.sum(bb==flag.nan),len(bb)))
        return ww, kk, ff, ee, bb

    #def _bpmask_interpolate(self,o):
    #    bpmasker = interp(self.pixxx+self.ptmin,self.S.bpmap[o])
    #    return bpmasker(self.xx)
        
    def interpolate_and_baryshift_orders(self,berv=None,interp_badpm=False,vref=0.):
        """
        Baryshift all orders. This loops over *interpolate_and_baryshift_order()*
        
        INPUT:
        
        OUTPUT:
        
        EXAMPLE:
        """
        for o in self.orders:
            print("Interpolating order {}".format(o))
            self.ww[o], self.kk[o], self.ff[o], self.ee[o], self.bb[o] = self.interpolate_and_baryshift_order(o=o,berv=berv,interp_badpm=interp_badpm,vref=vref)

    def get_flattened_spectrum(self,wavenum=True,savefolder='flattened_spectra/',savefile=True,berv=None,dropnan=True):
        """
        Get flattened spectrum, i.e., the full HPF spectrum including all orders
        
        INPUT:
            sp - a sspectrum object
            wavenum - 
            savefolder
            savefile - if True, save a file
        
        OUTPUT:
            ww - wavelengths (A) or wavenumbers in 1/cm. All in vacuum wavelengths
            ff - flux
        """
        ww = self.S.w.reshape(-1)
        if berv is None:
            berv = self.S.berv
        ww = barshift(ww,berv)
        if wavenum is True:
            ww = 1./(self.S.w.reshape(-1)*1e-8) # cm^-1
        ff = self.S.f.reshape(-1)
        utils.make_dir(savefolder)
        savename = savefolder + self.S.filename.split('/')[-1] + ".csv"
        
        df = pd.DataFrame(zip(ww,ff),columns=['w','f'])
        df = df.sort_values("w").reset_index(drop=True)
        if dropnan: df = df.dropna()
        if savefile:
            df.to_csv(savename,sep=' ',index=False,header=False)
            header_line = "#{} 0.0 0.0 {}\n#wavenum(cm^-1) flux\n".format(self.S.bjd,self.S.filename.split('/')[-1])
            with file(savename, 'r') as original: data = original.read()
            with file(savename, 'w') as modified: modified.write(header_line + data)
            print('Saved file to {}'.format(savename))
        return df.w.values, df.f.values        

class SSpectrumSet(object):
    """
    A class to deal with a number of SSpectrum classes
    """
    def __init__(self,splist):
        self.splist = splist
        self.ndim = len(splist)

    def remove_bad_frames(self,files):
        ibad = utils.get_indices_of_items(self.filename,files)
        self.splist = np.delete(self.splist,ibad)

    @property
    def medianflux18(self):
        return np.array([np.nanmedian(sp.S.f[18]) for sp in self.splist])

    @property
    def medianflux5(self):
        return np.array([np.nanmedian(sp.S.f[5]) for sp in self.splist])

    @property
    def qprog(self):
        return np.array([sp.S.qprog for sp in self.splist])
        
    @property
    def bjd(self):
        return np.array([sp.S.bjd for sp in self.splist])
    
    @property
    def sn55(self):
        return np.array([sp.S.sn55 for sp in self.splist])

    @property
    def sn5(self):
        return np.array([sp.S.sn5 for sp in self.splist])

    @property
    def sn14(self):
        return np.array([sp.S.sn14 for sp in self.splist])

    @property
    def sn15(self):
        return np.array([sp.S.sn15 for sp in self.splist])

    @property
    def sn17(self):
        return np.array([sp.S.sn17 for sp in self.splist])

    @property
    def sn18(self):
        return np.array([sp.S.sn18 for sp in self.splist])

    @property
    def berv(self):
        return np.array([sp.S.berv for sp in self.splist])

    @property
    def airmass(self):
        return np.array([sp.S.airmass for sp in self.splist])

    @property
    def gitcommit(self):
        return np.array([sp.S.gitcommit for sp in self.splist])

    @property
    def basename(self):
        return np.array([sp.basename for sp in self.splist])

    @property
    def filename(self):
        return np.array([sp.S.filename for sp in self.splist])

    @property
    def flag(self):
        return np.array([sp.S.flag for sp in self.splist])

    @property
    def exptime(self):
        return np.array([sp.S.exptime for sp in self.splist])

    @property
    def path_flat(self):
        return np.array([sp.S.path_flat for sp in self.splist])

    @property
    def istemplate(self):
        return np.array([sp.template for sp in self.splist])

    @property
    def datetime(self):
        return utils.jd2datetime(self.bjd)

    def __repr__(self):
        return str(self.splist)

    @property
    def df(self):
        df = pd.DataFrame(zip(self.basename,
                              self.bjd,
                              self.qprog,
                              self.datetime,
                              self.sn55,
                              self.sn5,
                              self.sn14,
                              self.sn15,
                              self.sn17,
                              self.sn18,
                              self.berv,
                              self.flag,
                              self.exptime,
                              self.airmass,
                              self.filename,
                              self.medianflux18,
                              self.medianflux5,
                              self.path_flat,
                              self.istemplate,
                              self.gitcommit),
                              columns = ['basename','bjd','qprog','datetime','sn55','sn5','sn14','sn15','sn17','sn18',
                                  'berv','flag','exptime','airmass','filename','medianflux18','medianflux5','path_flat','istemplate','gitcommit'])
        return df
        
    def loop_plot_orders(self,savefolder='loop_select/', filetype='png'):
        spectrum_widgets.loop_plot_orders(self.splist,savefolder=savefolder,filetype=filetype)


def print_mask(xmin,xmax,delta=0.036):
    print("#########################")
    print("{:0.3f}       0.0000000".format(xmin - delta))
    print("{:0.3f}       1.0000000".format(xmin))
    print("{:0.3f}       1.0000000".format(xmax))
    print("{:0.3f}       0.0000000".format(xmax + delta))
    print("#########################")
