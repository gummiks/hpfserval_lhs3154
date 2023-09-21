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
    lines = {'CaIRT1': (8498.02, -15., 15.),      
             'CaIRT1a': (8492, -40, 40),          
             'CaIRT1b': (8504, -40, 40),          
             'CaIRT2': (8542.09, -15., 15.),      
             'CaIRT2a': (8542.09, -300., -200.),  
             'CaIRT2b': (8542.09, 250., 350.),    
             'CaIRT3': (8662.14, -15., 15.),      
             'CaIRT3a': (8662.14, -300., -200.),  
             'CaIRT3b': (8662.14, 200., 300.),    
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

def get_absolute_index(w,f,e,v,berv,wcen=None,dv1=None,dv2=None,line_name='',plot=False,verbose=False,ax=None,plotline=True):
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
    
    # Absolute index
    I = np.mean(f[m])
    try: 
        e_I = 1. / m.sum() * np.sqrt(np.sum(e[m]**2))
    except Exception as ee:
        print(ee) 
        e_I = np.nan

    return I, e_I
