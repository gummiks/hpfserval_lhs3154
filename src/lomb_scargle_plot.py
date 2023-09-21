from astropy.timeseries import LombScargle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def plot_multi_lomb_scargle7(bjd,rv,e_rv,crx,e_crx,dLW,e_dLW,irt1,irt1_e,irt2,irt2_e,irt3,irt3_e,minimum_frequency=0.002,maximum_frequency=10.,savename=None,title=''):

    fig, (ax,bx,cx,dx,ex,fx,gx) = plt.subplots(dpi=600,nrows=7,figsize=(12,10),sharex=True)

    # RV
    #ax.errorbar(bjd,rv,e_rv,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
    ## CRX
    #bx.errorbar(bjd,crx,e_crx,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
    ## dLW
    #cx.errorbar(bjd,dLW,e_dLW,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)

    # RV
    L = LombScargle(bjd,rv,e_rv)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    ax.plot(periods, power,lw=1)
    ax.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    ax.set_ylabel("RVs: Power",fontsize=12)
    ax.legend(loc='upper right')

    # CRX
    L = LombScargle(bjd,crx,e_crx)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    bx.plot(periods, power,lw=1)
    bx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    bx.set_ylabel("CRX: Power",fontsize=10)
    bx.legend(loc='upper right')

    # dLW
    L = LombScargle(bjd,dLW,e_dLW)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    cx.plot(periods, power,lw=1)
    cx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    cx.set_ylabel("dLW: Power",fontsize=10)
    cx.legend(loc='upper right')

    # Ca IRT 1
    L = LombScargle(bjd,irt1,irt1_e)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    dx.plot(periods, power,lw=1)
    dx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    dx.set_ylabel("Ca IRT1: Power",fontsize=10)
    dx.legend(loc='upper right')

    # Ca IRT 2
    L = LombScargle(bjd,irt2,irt2_e)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    ex.plot(periods, power,lw=1)
    ex.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    ex.set_ylabel("Ca IRT2: Power",fontsize=10)
    ex.legend(loc='upper right')

    # Ca IRT 3
    L = LombScargle(bjd,irt3,irt3_e)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    fx.plot(periods, power,lw=1)
    fx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    fx.set_ylabel("Ca IRT3: Power",fontsize=10)
    fx.legend(loc='upper right')

    # WF
    print('Calculating window function')
    w_power = fourier_periodogram(bjd,np.ones(len(bjd)),frequency)
    gx.plot(periods,w_power,lw=1)
    gx.axvline(1.,0,1,color='gray',linestyle='--',label='1day')
    gx.axvline(365.245,0,1,color='gray',linestyle='--',label='1Year')
    gx.legend(loc='upper right')
    gx.set_xlabel('Period [d]',fontsize=16)
    gx.set_ylabel('Normalized WF',fontsize=10)
    fig.subplots_adjust(hspace=0.05)
    ax.set_title(title)

    for xx in (ax,bx,cx,dx,ex,fx,gx):
        xx.grid(lw=0.5,alpha=0.5)
        xx.minorticks_on()
        xx.set_xscale("log")
    if savename is not None:
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))
    return fig

def plot_multi_lomb_scargle(bjd,rv,e_rv,crx,e_crx,dLW,e_dLW,minimum_frequency=0.002,maximum_frequency=10.,savename=None,title=''):
    fig, (ax,bx,cx,dx) = plt.subplots(dpi=600,nrows=4,figsize=(12,10),sharex=True)

    # RV
    #ax.errorbar(bjd,rv,e_rv,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
    ## CRX
    #bx.errorbar(bjd,crx,e_crx,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
    ## dLW
    #cx.errorbar(bjd,dLW,e_dLW,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)

    # RV
    L = LombScargle(bjd,rv,e_rv)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    ax.plot(periods, power,lw=1)
    ax.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    ax.set_ylabel("RVs: Power",fontsize=16)
    ax.legend(loc='upper right')

    # CRX
    L = LombScargle(bjd,crx,e_crx)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    bx.plot(periods, power,lw=1)
    bx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    bx.set_ylabel("CRX: Power",fontsize=16)
    bx.legend(loc='upper right')

    # dLW
    L = LombScargle(bjd,dLW,e_dLW)
    frequency, power = L.autopower(minimum_frequency=minimum_frequency,maximum_frequency=maximum_frequency)
    periods = 1./frequency
    cx.plot(periods, power,lw=1)
    cx.plot(periods[np.argmax(power)],power[np.argmax(power)],marker="o",markersize=10,color="firebrick",alpha=0.5,
                label="Max power: {:.3f}days".format(periods[np.argmax(power)]))
    cx.set_ylabel("dLW: Power",fontsize=16)
    cx.legend(loc='upper right')

    # WF
    print('Calculating window function')
    w_power = fourier_periodogram(bjd,np.ones(len(bjd)),frequency)
    dx.plot(periods,w_power,lw=1)
    dx.axvline(1.,0,1,color='gray',linestyle='--',label='1day')
    dx.axvline(365.245,0,1,color='gray',linestyle='--',label='1Year')
    dx.legend(loc='upper right')
    dx.set_xlabel('Period [d]',fontsize=16)
    dx.set_ylabel('Normalized WF',fontsize=16)
    fig.subplots_adjust(hspace=0.05)
    ax.set_title(title)

    for xx in (ax,bx,cx,dx):
        xx.grid(lw=0.5,alpha=0.5)
        xx.minorticks_on()
        xx.set_xscale("log")
    if savename is not None:
        fig.savefig(savename,dpi=600) ; plt.close()
        print('Saved to {}'.format(savename))
    return fig

class LombScarglePlot(object):
    """
    Plot a LombScargle periodogram
    
    INPUT:
    t - time
    y - flux, assumes around 1.
    y_err - the error on y
    
    EXAMPLE:
        t = d.HJD_UTC.values
        y = d.rel_flux_T1.values
        norm = np.mean(y)
        y = y/norm
        y_err = d.rel_flux_err_T1/norm
        LP = LombScarglePlot(t,y,y_err)
        LP.calc_lomb_scargle()
        LP.plot()
    """
    def __init__(self,t,y,y_err=None):
        self.t = t
        self.y = y
        self.y_err = y_err
        
    def calc_lomb_scargle(self,minimum_frequency=0.002,maximum_frequency=10.,verbose=True,force_model_period=None):
        """
        Calculate the Lomb-scarge periodogram
        """
        self.L = LombScargle(self.t, self.y-1.,self.y_err)
        self.frequency, self.power = self.L.autopower(minimum_frequency=minimum_frequency,
                                                 maximum_frequency=maximum_frequency)
        self.periods = 1./self.frequency
        self.idx_max = np.argmax(self.power)
        self.best_frequency = self.frequency[self.idx_max]
        self.best_power = self.power[self.idx_max]
        self.best_period = 1./self.best_frequency
        
        self.phase_fit = np.linspace(0, 1)
        if force_model_period is not None:
            print("Forcing model period: {}".format(force_model_period))
            self.period_used = force_model_period
            self.model = self.L.model(t=self.phase_fit * force_model_period,frequency=1./force_model_period)+1.
            self.phase = (self.t / force_model_period) % 1 
        else:
            self.period_used = self.best_period
            self.model = self.L.model(t=self.phase_fit / self.best_frequency,frequency=self.best_frequency)+1.
            self.phase = (self.t * self.best_frequency) % 1 
        self.model_amplitude = (self.model.max()-self.model.min())/2.

        if verbose:
            print("Best period: \t{:.8f}".format(self.best_period))
            print("Best frequency: {:.8f}".format(self.best_frequency))
            print("Best power: \t{:.8f}".format(self.best_power))
            print("Model amplitude: {:.8f}".format(self.model_amplitude))

        print('Calculating window function')
        self.w_frequency = self.frequency
        self.w_periods = 1./self.w_frequency
        self.w_power = fourier_periodogram(self.t,np.ones(len(self.t)),self.w_frequency)

        
    def plot(self,title,ylabel='RV [m/s]'):
        self.fig = plt.figure(figsize=(18,10))
        gs = gridspec.GridSpec(2, 2)
        ax = plt.subplot(gs[0, :])
        bx = plt.subplot(gs[1, 0])
        cx = plt.subplot(gs[1, 1])
        
        # Unfolded data
        ax.errorbar(self.t,self.y,self.y_err,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
        ax.grid(lw=0.5,alpha=0.5)
        ax.minorticks_on()
        ax.set_ylabel(ylabel,fontsize=16)
        
        # Periodogram
        bx.plot(self.periods, self.power,lw=1)
        bx.plot(self.best_period,self.best_power,marker="o",markersize=15,color="firebrick",
                label="Max power: {:.3f}days".format(self.best_period))
        bx.set_xscale("log")
        bx.set_xlabel("Period",fontsize=16)
        bx.set_ylabel("Power",fontsize=16)
        bx.set_title("Lomb - Scargle power distribution",fontsize=16)
        bx.grid(lw=0.5,alpha=0.5)
        bx.minorticks_on()
        bx.set_ylim(0.,1.5*self.best_power)
        try:
            fa_1 = self.L.false_alarm_level(0.1)  
            fa_2 = self.L.false_alarm_level(0.05)  
            fa_3 = self.L.false_alarm_level(0.01)  
            bx.axhline(fa_1,label='FAP=0.1',color='orange',linestyle='-')
            bx.axhline(fa_2,label='FAP=0.05',color='orange',linestyle='--')
            bx.axhline(fa_3,label='FAP=0.01',color='orange',linestyle='-.')
        except Exception as e:
            print(e)
        bx.legend(loc="upper left",fontsize=14)
        
        # Phase folded plot
        cx.plot(self.phase_fit,self.model,color="red",zorder=10)
        cx.errorbar(self.phase,self.y,yerr=self.y_err,lw=0,elinewidth=1.,marker="o",mew=0.5,
                    markersize=8,capsize=4,label="Model amplitude: {:.4f}".format(self.model_amplitude),zorder=1)
        cx.margins(x=0.05,y=0.3)
        cx.legend(loc="upper left",fontsize=14)
        cx.grid(lw=0.3,alpha=0.3)
        cx.set_xlabel("Phase",fontsize=16)
        cx.set_title("Phase folded",fontsize=16)
        cx.minorticks_on()
        cx.set_ylabel("")
        cx.set_ylabel(ylabel,fontsize=16)
        
        self.fig.tight_layout()
        self.fig.suptitle(title,y=1.01,fontsize=22)

    def plot2(self,title,ylabel='RV [m/s]'):
        self.fig, (ax, bx, cx, dx) = plt.subplots(figsize=(15,15),nrows=4)
        
        # Unfolded data
        ax.errorbar(self.t,self.y,self.y_err,lw=0,elinewidth=1.,marker="o",mew=0.5,markersize=8,capsize=4)
        ax.grid(lw=0.3,alpha=0.3)
        ax.minorticks_on()
        ax.set_ylabel(ylabel,fontsize=16)
        ax.set_title(title,fontsize=22)
        
        # Periodogram
        bx.plot(self.periods, self.power,lw=1)
        bx.plot(self.best_period,self.best_power,marker="o",markersize=15,color="firebrick",
                label="Max power: {:.3f}days".format(self.best_period))
        bx.set_xscale("log")
        bx.set_xlabel("Period",fontsize=16,labelpad=0)
        bx.set_ylabel("LS Power",fontsize=16)
        #bx.set_title("Lomb - Scargle power distribution",fontsize=16)
        bx.grid(lw=0.3,alpha=0.3)
        bx.minorticks_on()
        bx.set_ylim(0.,1.5*self.best_power)
        try:
            fa_1 = self.L.false_alarm_level(0.1)  
            fa_2 = self.L.false_alarm_level(0.05)  
            fa_3 = self.L.false_alarm_level(0.01)  
            bx.axhline(fa_1,label='FAP=0.1',color='orange',linestyle='-')
            bx.axhline(fa_2,label='FAP=0.05',color='orange',linestyle='--')
            bx.axhline(fa_3,label='FAP=0.01',color='orange',linestyle='-.')
        except Exception as e:
            print(e)
        bx.legend(loc="upper right",fontsize=12,frameon=False)

        # Window function
        cx.plot(self.w_periods,self.w_power/np.max(self.w_power),lw=1)
        cx.axvline(1.,0,1,color='gray',linestyle='--',label='1day')
        cx.axvline(365.245,0,1,color='gray',linestyle='--',label='1Year')
        cx.set_ylabel('Normalized Window Function',fontsize=14)
        cx.set_xscale('log')
        cx.grid(lw=0.3,alpha=0.3)
        cx.minorticks_on()
        cx.legend(loc="upper right",fontsize=12,frameon=False)
        cx.set_xlabel("Period",fontsize=16,labelpad=0)
        #cx.set_title('Normalized Window Function',fontsize=14)

        # Phase folded plot
        dx.plot(self.phase_fit,self.model,color="red",zorder=10)
        dx.errorbar(self.phase,self.y,yerr=self.y_err,lw=0,elinewidth=1.,marker="o",mew=0.5,
                markersize=8,capsize=4,label="Period={:0.3f}d, Amplitude: {:.4f}".format(self.period_used,self.model_amplitude),zorder=1)
        dx.margins(x=0.05,y=0.3)
        dx.legend(loc="lower right",fontsize=14)
        dx.grid(lw=0.3,alpha=0.3)
        dx.set_xlabel("Phase",fontsize=16,labelpad=0)
        #dx.set_title("Phase folded",fontsize=16)
        dx.minorticks_on()
        dx.set_ylabel(ylabel,fontsize=16)
        s = np.std(self.y)
        dx.set_ylim(np.min(self.y)-s,np.max(self.y)+s)
        
        self.fig.subplots_adjust(hspace=0.2)
        #self.fig.suptitle(title,y=1.0,fontsize=22)


def fourier_periodogram(x,y,f):
    """
    My implementation of using the classical discrete Fourier transform.
    Seems to be fairly fast ?
    
    INPUT:
        x - x values 
        y - y values
        f - frequency values to calculate the resulting periodogram
        
    EXAMPLE:
        
    NOTES:
        See Cochran 1996, EQ 16
        To get window function, then just put y = ones(len(x))
    """
    N = len(x)
    power = []
    for _f in f:
        cos = np.dot(y,np.cos(2.*np.pi*_f*x))**2.
        sin = np.dot(y,np.sin(2.*np.pi*_f*x))**2.
        _p = (cos+sin)/N
        power.append(_p)
    return np.array(power)


