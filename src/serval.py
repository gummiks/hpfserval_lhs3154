#! /usr/bin/env python
__author__ = 'Mathias Zechmeister, heavily modified for HPF by GKS'
__version__ = '2020'
description = '''
SERVAL - SpEctrum Radial Velocity AnaLyser (%s)
     by %s
''' % (__version__, __author__)
import logger
logger = logger.logger()
import ctypes
from ctypes import c_double, c_int
import os
import resource
import sys
import numpy as np
from scipy.optimize import curve_fit
from gplot import *
from pause import pause, stop
from wstat import rms
from calcspec import calcspec
import matplotlib.pyplot as plt
from serval_config import v_lo, v_hi, v_step, safemode, def_wlog, flag, sflag

servalsrc = os.path.dirname(os.path.realpath(__file__)) + os.sep
ptr = np.ctypeslib.ndpointer
_pKolynomial0 = ctypes.CDLL(servalsrc+'polyregression.so')
_pKolynomial0.polyfit.restype = c_double
_pKolynomial = np.ctypeslib.load_library(servalsrc+'polyregression.so', '.')
_pKolynomial.polyfit.restype = c_double
_pKolynomial.polyfit.argtypes = [
   ptr(dtype=float),  # x2
   ptr(dtype=float),  # y2
   ptr(dtype=float),  # e_y2
   ptr(dtype=float),  # fmod
   #ptr(dtype=bool),  # ind
   c_double,             # ind
   c_int, c_double, c_int,  # n, wcen, deg
   ptr(dtype=float),  # p
   ptr(dtype=float),  # lhs
   ptr(dtype=float)   # pstat
]
_pKolynomial.interpol1D.argtypes = [
   ptr(dtype=float),  # xn
   ptr(dtype=float),  # yn
   ptr(dtype=float),  # x
   ptr(dtype=float),  # y
   c_int, c_int          # nn, n
]

c = 299792.4580   # [km/s] speed of light
# default values
review = 0       # review template
postiter = 3     # number of iterations for post rvs (postclip=3)
debug = 0
sp, fmod = None, None    # @getHalpha
apar = np.zeros(3)      # parabola parameters
astat = np.zeros(3*2-1)
alhs = np.zeros((3, 3))

if tuple(map(int,np.__version__.split('.'))) > (1,6,1):
   np.seterr(invalid='ignore', divide='ignore') # suppression warnings when comparing with nan.

def lam2wave(l, wlog=def_wlog):
   return np.log(l) if wlog else l

def wave2lam(w, wlog=def_wlog):
   return np.exp(w) if wlog else w

def nans(*args, **kwargs):
   return np.nan * np.empty(*args, **kwargs)

def minsec(t): return '%um%.3fs' % divmod(t, 60)   # format time

def gauss(x, a0, a1, a2, a3):
   z = (x-a0) / a2
   y = a1 * np.exp(-z**2 / 2.) + a3 #+ a4 * x + a5 * x**2
   return y

def Using(point, verb=False):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if verb: print('%s: usertime=%s systime=%s mem=%s mb' % (point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 ))
    return (usage[2]*resource.getpagesize())/1000000.0

def nomask(x):   # dummy function, tellurics are not masked
   return 0. * x

def lineindex(l, r1, r2):
   if np.isnan(r1[0]) or np.isnan(r2[0]):
      return np.nan, np.nan
   s = l[0] / (r1[0]+r2[0]) * 2
   # error propagation:
   e = s * np.sqrt((l[1]/l[0])**2 + (r1[1]**2+r2[1]**2)/(r1[0]+r2[0])**2)
   return s, e

class Logger(object):
   """Not used for notebook"""

   def __init__(self):
      self.terminal = sys.stdout
      self.logfile = None # open(logfilename, "a")
      self.logbuf = ''

   def write(self, message):   # fork the output to stdout and file
      self.terminal.write(message)
      if self.logfile:
         self.logfile.write(message)
      else:
         self.logbuf += message

   def logname(self, logfilename):
       self.logfile = open(logfilename, 'a')
       self.logfile.write(self.logbuf)
       print('logging to', logfilename)

class interp:
   """
   Interpolation similar to interpolate.interp1d but faster

   INPUT:
       array must be sorted; 1D arrays only !!!

   EXAMPLE:
       mask = get_mask()
       tellmask = interp(lam2wave(mask[:,0]), mask[:,1]) # callable class
       tellmask
   """
   def __init__(self, x, y) :
      self.x = 1 * x # we like real arrays
      self.y = 1 * y
   def __call__(self, xx):
      yy = 0 * xx
      _pKolynomial.interpol1D(self.x, self.y, xx, yy, self.x.size, xx.size)
      return yy


def polyreg(x2, y2, e_y2, v, deg=1, retmod=True,use_python_version=False,verbose=True):   # polynomial regression
   """
   Polynomial regression, returns polynomial coefficients and goodness of fit.

   Steps through v in steps of 0.1

   INPUT:
   - x2 - wavelengths
   - y2 - flux
   - e_y2 - flux errors
   - v = 0
   - deg=1, normally 5 ? 

   OUTPUT:
   - p - the polynomial coefficients, they will be of same number as degree
   - SSR - goodness of fit number
   - fmod (if retmod == TRUE) - modified flux -- flux * polynomial ??

   NOTES:
   - depends on calcspec:
        fmod = calcspec(x2, v, 1.)  # get the shifted template, and applies polynomial coeffs

   EXAMPLE:
    IN opti()
        p, SSR[k] = polyreg(x2, y2, e_y2, vgrid[k], len(p), retmod=False)
        p, SSR, fmod = polyreg(x2=ww[o],y2=ff[o],e_y2=ee[o],v=0.,deg=5)

   DEBUG:
    polyreg(x2=ww[o],y2=ff[o],e_y2=ee[o],v=0.,deg=5)
    WARNING: Matrix is not positive definite. Zero or negative yerr values for  0 at []
    (array([nan, nan, nan, nan, nan]),
     -1.0,
     array([nan, nan, nan, ..., nan, nan, nan]))

   - when works:
   polyreg(x2=ww[o],y2=ff[o],e_y2=ee[o],v=0.,deg=5)
   (array([ 3.58756194e+00, -2.41596393e+01, -2.69577726e+04,  6.68788620e+05,
        -9.22475608e+08]),
 89156.93324503534,
 array([10.73886588, 17.44216286, 24.82915535, ...,  6.03410631,
        17.89608724, 35.85210355]))
   """
   fmod = calcspec(x2, v, 1.)  # get the shifted template
   if use_python_version: # python version
      #print("v=",v,"using ##### PYTHON ##### version")
      ind = fmod>0.01     # avoid zero flux, negative flux and overflow

      #p,stat = polynomial.polyfit(x2[ind]-calcspec.wcen, y2[ind]/fmod[ind], deg-1, w=fmod[ind]/e_y2[ind], full=True)
      #GKS EDIT
      p,stat = np.polynomial.polynomial.polyfit(x2[ind]-calcspec.wcen, y2[ind]/fmod[ind], deg-1, w=fmod[ind]/e_y2[ind], full=True)
      SSR = stat[0][0]
   else: # pure c version
      #print("v=",v,"not using python version")
      pstat = np.empty(deg*2-1)
      p = np.empty(deg)
      lhs = np.empty((deg, deg))
      # _pKolynomial.polyfit(x2, y2, e_y2, fmod, ind, ind.size,globvar.wcen, rhs.size, rhs, lhs, pstat, chi)
      ind = 0.0001
      # no check for zero division inside _pKolynomial.polyfit!
      #pause()
      SSR = _pKolynomial.polyfit(x2, y2, e_y2, fmod, ind, x2.size, calcspec.wcen, deg, p, lhs, pstat)
      if SSR < 0:
         ii, = np.where((e_y2<=0) & (fmod>0.01))
         # GKS: This error is called, is that because of _pKolynomial in mac ? ctypes ?
         print('WARNING: Matrix is not positive definite.', 'Zero or negative yerr values for ', ii.size, 'at', ii)
         p = 0*p
         pause(0)

      if 0: #abs(v)>200./1000.:
         gplot(x2,y2,calcspec(x2, v, *p), ' w lp,"" us 1:3 w l, "" us 1:($2-$3) t "res [v=%f,SSR=%f]"'%(v, SSR))
         pause('v, SSR: ',v,SSR)
   if retmod: # return the model
      return p, SSR, calcspec(x2, v, *p, fmod=fmod) # calcspec = fmod
   return p, SSR

def optidrift(ft, df, f2, e2=None):
   """
   Scale derivative to the residuals.

   Model:
      f(v) = A*f - A*df/dv * v/c
   """
   # pre-normalise
   #A = np.dot(ft, f2) / np.dot(ft, ft)   # unweighted (more robust against bad error estimate)
   A = np.dot(1./e2**2*ft, f2) / np.dot(1/e2**2*ft, ft)
   fmod = A * ft
   v = -c * np.dot(1./e2**2*df, f2-fmod) / np.dot(1./e2**2*df, df) / A #**2
   if 0:
      # show RV contribution of each pixel!
      print('median', np.median(-(f2-fmod)/df*c/A*1000), ' v', v*1000)
      gplot(-(f2-fmod)/df*c/A*1000, e2/df*c/A*1000, 'us 0:1:2 w e, %s' % (v*1000))
      pause()
   e_v = c / np.sqrt(np.dot(1./e2**2*df, df)) / A
   fmod = fmod - A*df*v/c
   #gplot(f2, ',', ft*A, ',', f2-fmod,e2, 'us 0:1:2 w e')
   #gplot(f2, ',', ft*A, ',', f2-A*ft,e2, 'us 0:1:2 w e')
   #gplot(f2, ft*A, A*ft-A*df/c*v, A*ft-A*df/c*0.1,' us 0:1, "" us 0:2, "" us 0:3, "" us 0:4, "" us 0:($1-$3) w lp,  "" us 0:($1-$4) w lp lt 7')
   return  type('par',(),{'params': np.append(v,A), 'perror': np.array([e_v,1.0])}), fmod


def SSRstat(vgrid, SSR, dk=1, plot='maybe'):
   """
   Returns a chi squared stat
   """
   # analyse peak
   k = SSR[dk:-dk].argmin() + dk   # best point (exclude borders)
   vpeak = vgrid[k-dk:k+dk+1]
   SSRpeak = SSR[k-dk:k+dk+1] - SSR[k]
   # interpolating parabola (direct solution) through the three pixels in the minimum
   a = np.array([0, (SSR[k+dk]-SSR[k-dk])/(2*v_step), (SSR[k+dk]-2*SSR[k]+SSR[k-dk])/(2*v_step**2)])  # interpolating parabola for even grid
   v = (SSR[k+dk]-SSR[k-dk]) / (SSR[k+dk]-2*SSR[k]+SSR[k-dk]) * 0.5 * v_step

   v = vgrid[k] - a[1]/2./a[2]   # parabola minimum
   e_v = np.nan
   if -1 in SSR:
      print('opti warning: bad ccf.')
   elif a[2] <= 0:
      print('opti warning: a[2]=%f<=0.' % a[2])
   elif not vgrid[0] <= v <= vgrid[-1]:
      print('opti warning: v not in [va,vb].')
   else:
      e_v = 1. / a[2]**0.5
   # #########################
   # #########################
   # REMOVING FUNCTIONALITY TO PAUSE 
   # 20200203 - gks
   # #########################
   # #########################
   #if (plot==1 and np.isnan(e_v)) or plot==2:
   #   gplot_set('set yrange [*:%f]'%SSR.max())
   #   gplot(vgrid, SSR-SSR[k], " w lp, v1="+str(vgrid[k])+", %f+(x-v1)*%f+(x-v1)**2*%f," % tuple(a), [v,v], [0,SSR[1]], 'w l t "%f km/s"'%v)
   #   ogplot(vpeak, SSRpeak, ' lt 1 pt 6; set yrange [*:*]')
   #   pause(v)
   return v, e_v, a

def opti(va, vb, x2, y2, e_y2, p=None, vfix=False, plot=False,use_python_version=False):
   """
   Optimize
   vfix to fix v for RV constant stars?
   performs a mini CCF; the grid stepping
   returns best v and errors from parabola curvature

   INPUT:
   - va - used to create vgrid, in steps of 0.1km/s
   - vb - used to create vgrid, in steps of 0.1km/s
   - x2 - wavelengths
   - y2 - fluxes
   - e_y2 - errors
   - p ? len(p) = 5 ?
   - vfix, optional, 
   - plot, 

   OUTPUT:
    class par:
    par.params - np.append(v,p)
    par.perror - np.array([e_v,1.0])
    par.ssr    - (vgrid,SSR)
    fmod - modified flux 

   EXAMPLE:
     v = 0
     v_lo = -5.5
     v_hi = 5.6
     v_step = 0.1
     par, fModkeep = opti(va=v+v_lo,vb=v+v_hi,x2=w2.take(keep,mode='clip'),y2=f2.take(keep,mode='clip'),e_y2=e_y.take(keep,mode='clip'),p= p[1:], vfix=vfix, plot=plot)
     ssr = par.ssr
   """
   vgrid = np.arange(va, vb, v_step)
   nk = len(vgrid)

   SSR = np.empty(nk)
   for k in range(nk):
      p, SSR[k] = polyreg(x2, y2, e_y2, vgrid[k], len(p), retmod=False,use_python_version=use_python_version)

   # analyse the CCF peak fitting
   v, e_v, a = SSRstat(vgrid, SSR, plot=(not safemode)*(1+plot))

   if np.isnan(e_v):
      v = vgrid[int(nk/2)]   # actually it should be nan, but may the next clipping loop or plot use vcen
      print(" Setting  v=" % v)
   if vfix: v = 0.
   p, SSRmin, fmod = polyreg(x2, y2, e_y2, v, len(p))   # final call with v

   if 0 and (np.isnan(e_v) or plot) and not safemode:
        gplot(x2, y2, fmod, ' w lp, "" us 1:3 w lp lt 3')
        pause(v)
   return type('par', (), {'params': np.append(v,p), 'perror': np.array([e_v,1.0]), 'ssr': (vgrid,SSR)}), fmod

def fitspec(wt, ft, tck, w2, f2, e_y=None, v=0, vfix=False, clip=None, nclip=1, keep=None, indmod=np.s_[:], v_step=True, df=None, plot=False, deg=3, chi2map=False,use_python_version=False):
   """
   Performs the robust least square fit via iterative clipping.

   INPUT:
   - wt - wavelength, template
   - ft - flux, template
   - tck - the kk array of derivatives
   - w2 - current wavelength
   - f2 - current flux
   - e_y - current error
   - v=0, v is fixed. For RV constant stars or in coadding when only the background polynomial is computed.
   - vfix = FALSE
   - clip : Kappa sigma clipping value.
   - nclip=1 Number of clipping iterations (default: 0 if clip else 1).
   - keep - what indices to use (NOT NONE)
   - indmod : Index range to be finally calculated-
   - v_step - Number of v steps (only background polynomial => TRUE 
   - df : Derivative for drift measurement.
   - plot = False, if true, plot a Chi^2 plot: chi^2 as a function of v grid, stepping from -5.5 to 5.6 km/s
   - deg = 3
   - chi2map - 

   OUTPUT:
      par
      fMod
      keep
      stat
      ssr (optional chi2map==True)

   EXAMPLE:
    par,fmod,keep,stat = fitspec(w2, f2, k2, ww[o],ff[o], e_y=ee[o],v=0., vfix=vtfix, keep=pind, deg=deg)
   """
   calcspec.wcen = np.mean(w2)
   ## WHY IS THIS BEING PASSED LIKE THIS ? 
   # wt and ft is not even being used ??
   calcspec.tck = tck
   if keep is None: keep = np.arange(len(w2))
   if e_y is None: e_y = np.mean(f2)**0.5 + 0*f2   # mean photon noise
   if clip is None: nclip = 0   # number of clip iterations; default 1

   p = np.array([v, 1.] + [0]*deg)   # => [v,1,0,0,0] # len = 5
   fMod = np.nan * w2
   #fres = 0.*w2     # same size
   for n in range(nclip+1):
      if df is not None:
         '''drift mode: scale derivative to residuals'''
         par, fModkeep = optidrift(ft.take(keep,mode='clip'), df.take(keep,mode='clip'), f2.take(keep,mode='clip'),
                                 e_y.take(keep,mode='clip'))
      elif v_step:
         '''least square mode'''
         par, fModkeep = opti(v+v_lo, v+v_hi, w2.take(keep,mode='clip'), f2.take(keep,mode='clip'),
                              e_y.take(keep,mode='clip'), p[1:], vfix=vfix, plot=plot,use_python_version=use_python_version)
         ssr = par.ssr
      else:
         '''only background polynomial'''
         p, SSR, fModkeep = polyreg(w2.take(keep,mode='clip'), f2.take(keep,mode='clip'), e_y.take(keep,mode='clip'), v, len(p)-1)
         par = type('par',(),{'params': np.append(v,p), 'ssr': SSR})

      p = par.params
      par.niter = n
      if 1:
         # exclude model regions with negative flux
         # can occur e.g. in background in ThAr, or due to tellurics
         ind = fModkeep > 0
         keep = keep[ind]      # ignore the pixels modelled with negative flux
         fModkeep = fModkeep[ind]
         # all might be negative (for low/zero S/N data)
      #fres[keep] = (f2[keep] - fMod[keep]) / fMod[keep]**0.5
      #res_std = rms(fres[keep])     # residual noise / photon noise
      # use error predicted by model
      #fres = (f2.take(keep,mode='clip')-fModkeep) / fModkeep**0.5
      # use external errors
      fres = (f2.take(keep,mode='clip')-fModkeep) / e_y.take(keep,mode='clip')
      res_std = rms(fres)     # residual noise / photon noise
      if n < nclip:
         ind = np.abs(fres) <= clip*res_std
         nreject = len(keep) - np.sum(ind)
         if nreject: keep = keep[ind]   # prepare next clip loop
         # else: break
      if len(keep)<10: # too much rejected? too many negative values?
         print("too much rejected, skipping")
         break
      if 0 and np.abs(par.params[0]*1000)>20:
         if df:
            fMod = ft * p[1]     # compute also at bad pixels
         else:
            fMod = calcspec(w2, *p)     # compute also at bad pixels
         gplot_set('set y2tics; set ytics nomir;set y2range [-5:35];')
         gplot(w2,fMod,' w lp pt 7 ps 0.5 t "fmod"',flush='');
         ogplot(w2[keep],fMod[keep],' w lp pt 7 ps 0.5 t "fmod[keep]"',flush='');
         ogplot(w2,f2,' w lp pt 7 ps 0.5 t "f2"',flush='');
         #ogplot(w2[keep],fres[keep],' w lp pt 7 ps 0.5 lc rgb "red" axis x1y2 t "residuals"',flush='')
         ogplot(w2[keep],fres,' w lp pt 7 ps 0.5 lc rgb "black" axis x1y2, 0 w l lt 2 axis x1y2 t"", '+str(res_std)+' w l lt 1 axis x1y2, '+str(-res_std)+ ' w l lt 1 t "" axis x1y2')
         pause('large RV', par.params[0]*1000)

   stat = {"std": res_std, "snr": np.mean(fModkeep)/np.mean(np.abs(f2.take(keep,mode='clip')-fModkeep))}
   #pause(stat["snr"], wmean(fModkeep)/wrms(f2.take(keep,mode='clip')-fModkeep), np.median(fModkeep)/np.median(np.abs(f2.take(keep,mode='clip')-fModkeep)) )
   if df is not None:
      fMod[indmod] = ft[indmod]*p[1] - df[indmod]*p[1]*p[0]/c  # compute also at bad pixels
   else:
      fMod[indmod] = calcspec(w2[indmod], *p)   # compute also at bad pixels

   if chi2map:
      return par, fMod, keep, stat, ssr
   else:
      return par, fMod, keep, stat




