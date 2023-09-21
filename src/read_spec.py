#!/usr/bin/env python
__author__ = 'Mathias Zechmeister with significant edits by Gummi'
__version__ = '2020'
import datetime
import glob
import gzip
import os
import sys
import tarfile
import time
import warnings
from collections import namedtuple
import astropy
import astropy.io
import astropy.io.fits as pyfits
import numpy as np
from pause import pause, stop
from gplot import *
import serval_config
from serval_config import def_wlog, flag
import serval_help
from airtovac import airtovac
import utils

class Spectrum:
    """
    Spectrum class to deal with spectra
    """
    brvref = "WE"
    def __init__(self, filename, inst='HPF', pfits=True, orders=None,
               wlog=def_wlog, drs=False, fib=None, targ=None, verb=False,jd_utc=None):
       """
       INPUT:
            targ
       
       pfits : fits reader
              True: pyfits (safe, e.g. to get the full header of the start refence spectrum)
              2: my own simple pure python fits reader, faster than pyfits
              It was developed for HARPS e2ds, but the advent ADP.*.tar was difficult.
       """
       self.filename = filename
       self.drsname = 'DRS'
       self.drs = drs
       self.flag = 0
       self.bflag = 0
       self.tmmean = 0
       self.f = None
       self.w = None
       self.e = None
       self.bpmap = None
       self.mod = None
       self.fib = fib
       self.fo = None
       self.wo = None
       self.eo = None
       self.bo = None
       self.drift = np.nan
       self.e_drift = np.nan
       self.utc = None
       self.ra = None
       self.de = None
       self.obs = type('specdata', (object,), dict(lat=None, lon=None))
       self.airmass = np.nan
       if '.gz' in filename: pfits=True
       self.ccf = type('ccf',(), dict(rvc=np.nan, err_rvc=np.nan, bis=np.nan, fwhm=np.nan, contrast=np.nan, mask=0, header=0))
       self.read_spec(filename, inst=inst, pfits=pfits, verb=verb)   # scan fits header
       if inst != self.inst:
           pause('WARNING:', filename, 'from', self.inst, ', but mode is', inst)
       self.obj = self.header['OBJECT']

       #------------------------------------
       ### Barycentric correction ###
       #------------------------------------
       if self.inst != "HPF":
           jd_utc = [self.mjd + 2400000.5 + self.exptime*self.tmmean/24./3600.]
       else:
           if jd_utc is not None:
               print(("Using supplied jd_utc={} instead of jd_utc = {}".format(jd_utc,self.jd_midpoint)))
               jd_utc = [jd_utc]
           else:
               jd_utc = [self.jd_midpoint]
       obsname = {'CARM_VIS':'ca', 'CARM_NIR':'ca', 'FEROS':'eso', 'HARPS':'eso', 'HARPN':'lapalma', "HPF": "McDonald Observatory"}[inst]
       if verb:
           print("Using barycorrpy to calculate berv and bjd")
           #print(("jd_utc {} ra {} dec {} obsname {} pmra {} pmdec {} plx {}mas rv {}m/s".format(jd_utc[0], ra, de, obsname, targ.pmra,
           #                                                                                      targ.pmde, targ.plx, targ.abs_rv)))
       #self.bjd, self.berv = targ.get_bjd_berv(jd_utc[0],obsname=obsname)
       self.bjd, self.berv = targ.calc_barycentric_velocity(jd_utc[0],obsname=obsname)
       if verb:
           print(("berv = {}km/s and bjd = {}".format(self.berv, self.bjd)))
       #------------------------------------
       # End barycentric correction
       #------------------------------------
       if self.fib == 'B':
           self.berv = np.nan
           self.drsberv = np.nan
       self.header['HIERARCH SERVAL BREF'] = (self.brvref, 'Barycentric code')
       self.header['HIERARCH SERVAL BJD'] = (self.bjd, 'Barycentric Julian Day')
       self.header['HIERARCH SERVAL BERV'] = (self.berv, '[km/s] Barycentric correction')
       if self.utc:
           date = self.utc.year, self.utc.month, self.utc.day
           ut = self.utc.hour + self.utc.minute/60. +  self.utc.second/3600.
           utend = ut + self.exptime/3600.
       if orders is not None:
           self.read_data(orders=orders, wlog=wlog)

    def get_data(self, orders=np.s_[:], wlog=def_wlog, verb=False, **kwargs):
        """Returns only data."""
        o = orders
        if self.w is not None:
            w, f, e, b = self.w[o], self.f[o], self.e[o], self.bpmap[o]
        else:
            w, f, e, b = self.read_spec(self.filename, inst=self.inst, orders=orders, verb=verb,**kwargs)
            w = np.log(w) if wlog else w.astype(float)
            f = f.astype(float)
            e = e.astype(float)
            #self.bflag |= np.bitwise_or.reduce(b.ravel())
        return type('specdata', (object,),dict(w=w, f=f, e=e, bpmap=b, berv=self.berv, o=o))

    def read_data(self, verb=False, **kwargs):
        """Read only data, no header, store it as an attribute."""
        data = self.get_data(**kwargs)
        if isinstance(kwargs.get('orders'), int):
            self.wo, self.fo, self.eo, self.bo = data.w, data.f, data.e, data.bpmap
        else:
            self.w, self.f, self.e, self.bpmap = data.w, data.f, data.e, data.bpmap

    def read_spec(self, s, inst, plot=False, **kwargs):
        """
        Main function to read spectra. Currently only inst='HPF' is supported
        """
        if inst == "HPF":
            sp = self.read_hpf(s, **kwargs)
        else:
            print('Current instrument {} not supported',format(inst))
            sp = None
        return sp

    def read_hpf(self, s, orders=None,pfits=False,verb=True):
        """
        Read HPF data

        INPUT:
            s - filename

        OUTPUT:
            w - wavelengths for all 28 orders
            f - fluxes for all 28 orders
            e - flux errors for all 28 orders
            bpmap - bad pixel map

        EXAMPLE:
            
        """
        self.inst = "HPF"
        # Open main file
        self.filename = s
        self.longbasename = serval_help.hpfspeclongbasename(s)
        self.hdulist = astropy.io.fits.open(self.filename)
        self.header = astropy.io.fits.getheader(self.filename)

        # Get some header keywords
        self.exptime = self.header["EXPLNDR"]
        try:
            self.qprog = self.header["QPROG"]
        except Exception as e:
            print(e)
            self.qprog = "UNKNOWN"
        self.dateobs = self.header["DATE-OBS"]
        self.utc = datetime.datetime.strptime(self.dateobs, '%Y-%m-%dT%H:%M:%S.%f')
        self.timeid = self.dateobs
        self.jd_start = astropy.time.Time(self.dateobs).jd
        self.tmmean = 0.5 # assume 0.5 for now #hdr[k_tmmean]
        try:
            self.extrmeth = self.header['EXTRMETH'] # 'FlatRelativeOptimal'
        except Exception as e:
            print(e)
            self.extrmeth = 'OldOptimal'

        ##################################
        # FLATS BEGIN
        ##################################
        if serval_config.flat_mode=='old':
            path_flat = serval_config.instruments[self.inst]["path_flat"]
        if serval_config.flat_mode=='may25':
            path_flat = serval_config.instruments[self.inst]["path_flat_may25"]
        if serval_config.flat_mode=='sept18':
            path_flat = serval_config.instruments[self.inst]["path_flat_sept18"]
        if serval_config.flat_mode=='march02':
            path_flat = serval_config.instruments[self.inst]["path_flat_march02"]
        if serval_config.flat_mode=='noflat':
            path_flat = serval_config.instruments[self.inst]["path_flat_noflat"]
        if serval_config.flat_mode=='standard':
            path_flat = serval_config.instruments[self.inst]["path_flat_march_july"]
        if serval_config.flat_mode=='june':
            path_flat = serval_config.instruments[self.inst]["path_flat_june"]
        if serval_config.flat_mode=='auto':
            if self.extrmeth == 'FlatRelativeOptimal':
                path_flat = serval_config.instruments[self.inst]["path_flat_noflat"]
            elif self.extrmeth == 'OldOptimal':
                path_flat = serval_config.instruments[self.inst]["path_flat_sept18"]
            else:
                print('Extraction method {} not recognized'.format(self.extrmeth))
                sys.exit()

        if serval_config.flat_mode=='ABC':
            ## FLAT: split flat ABCD
            ##if self.jd_start > 2458239.5:
            time_A = ['2018-02-02 13:46:05','2018-03-25 01:08:28']
            time_B = ['2018-03-25 01:08:29','2018-05-25 23:20:30'] # modified July 28th
            time_C = ['2018-05-25 23:20:31','2022-06-28 00:13:38']
            jdA = utils.iso2jd(time_A)
            jdB = utils.iso2jd(time_B)
            jdC = utils.iso2jd(time_C)
            #jdD = utils.iso2jd(time_D)
            if jdA[0] < self.jd_start < jdA[1]:
                path_flat = serval_config.instruments[self.inst]["path_flat_A"]
            elif jdB[0] < self.jd_start < jdB[1]:
                path_flat = serval_config.instruments[self.inst]["path_flat_B"]
            elif jdC[0] < self.jd_start < jdC[1]:
                path_flat = serval_config.instruments[self.inst]["path_flat_C"]
            #elif jdD[0] < self.jd_start < jdD[1]:
            #    path_flat = serval_config.instruments[self.inst]["path_flat_D"]
        print(('Using FLAT: {}'.format(path_flat)))
        print(('FLAT MODE: {}'.format(serval_config.flat_mode)))
        hdulist = astropy.io.fits.open(path_flat)
        self.flat_sci = hdulist[1].data# flat[2::3,:]
        self.flat_sky = hdulist[2].data# flat[2::3,:]
        self.path_flat = path_flat
        self.flat_mode = serval_config.flat_mode
        self.path_telluric_mask = serval_config.path_to_tellmask_file
        self.path_sky_mask = serval_config.path_to_skymask_file
        self.path_stellar_mask = serval_config.path_to_stellarmask_file
        ##################################
        # FLATS END
        ##################################
        self._f_sci = self.hdulist[1].data*self.exptime
        self.f_sci = self.hdulist[1].data*self.exptime/self.flat_sci
        # Setting edges as nan (those pixels don't carry any information)
        self.f_sci[:,0:4] = np.nan
        self.f_sci[:,-4:] = np.nan

        path_wavelength_solution_sky = serval_config.instruments[self.inst]["path_wavelength_solution_sky"]
        self.w_sky = astropy.io.fits.getdata(path_wavelength_solution_sky)

        # WAVELENGTH
        if serval_config.fixed_wavelength_solution is False:
            try: 
                print('Using drift-corrected wavelength solution')
                w = self.hdulist[7].data #+ 1
            except Exception as e:
                print(('ERROR',e,'Reverting to fixed wavelength'))
                path_wavelength_solution_sci = serval_config.instruments[self.inst]["path_wavelength_solution_sci"]
                w = astropy.io.fits.getdata(path_wavelength_solution_sci)
        else:
            print('Using fixed wavelength solution')
            path_wavelength_solution_sci = serval_config.instruments[self.inst]["path_wavelength_solution_sci"]
            w = astropy.io.fits.getdata(path_wavelength_solution_sci)

        self.sky_subtract = serval_config.sky_subtract
        self.SKY_SCALING_FACTOR = serval_config.SKY_SCALING_FACTOR#/1.035
        self.f_sky = (self.hdulist[2].data*self.exptime/self.flat_sky)*self.SKY_SCALING_FACTOR
        self._f_sky = self.hdulist[2].data*self.exptime
        #self.f_sky = (self.hdulist[2].data*self.exptime)*SKY_SCALING_FACTOR
        self.f_cal = self.hdulist[3].data*self.exptime#/flat_cal
        self._f_cal = self.hdulist[3].data*self.exptime#/flat_cal
        #self.f_sky = np.zeros(w.shape)

        # Interpolating sky
        #for o in range(w.shape[0]):
        #    self.f_sky[o] = np.interp(w[o],self.w_sky[o],self._f_sky[o])

        #self.f = self.f_sci 
        if serval_config.sky_subtract:
            print(('SKY SUBTRACTING by {}'.format(self.SKY_SCALING_FACTOR)))
            self.f = self.f_sci - self.f_sky
            self._f = self._f_sci - self._f_sky
        else:
            print('NOT SKY SUBTRACTING')
            self.f = self.f_sci
            self._f = self._f_sci
        if bool(self.filename.find('Goldilocks')+1):
            print('Removing ends of the Goldilocks fits')
            self.f[self.f == 0] = np.nan
        #self.f_sci_flat = self.hdulist[1].data*self.exptime/self.flat_sci
        #self.f_sky_flat = self.hdulist[2].data*self.exptime/self.flat_sky
        #self.f_cal_flat = self.hdulist[3].data*self.exptime/self.flat_cal
        #self.f_flat = self.f_sci_flat - self.f_sky_flat
        #print("Sigma clipping sigma={}".format(sigma))
        #self.f, self.mask = serval_help.median_sigma_clip(self.f,window=49,sigma_upper=sigma,sigma=sigma)
        #print("Found {} outliers".format(np.sum(self.mask)))

        if serval_config.simulation_mode:
            print('SIMULATION MODE, scaling variance array by {}'.format(serval_config.variance_scaling_factor))
            for i in range(len(self.f)):
                self.f[i] = self.f[i]/np.nanmedian(self.f[i])
            self.e_sci = self.f_sci/serval_config.variance_scaling_factor
            self.e_sky = self.f_sky/serval_config.variance_scaling_factor
            self.e_cal = self.f_cal/serval_config.variance_scaling_factor
            self.e = self.e_sci
            # normalizing and adding actual error
            f_original = np.copy(self.f)
            print('ADDING ERROR')
            for i in range(len(self.f)):
                NN = len(self.f[i])
                z = np.zeros(NN)
                err = np.random.normal(loc=z,scale=np.sqrt(self.f[i])/serval_config.variance_scaling_factor,size=NN)
                self.f[i] = self.f[i] + err
                res = self.f[i]/(f_original[i]/np.nanmedian(f_original[i]))
                mm = np.isfinite(res)
                print('Order = {}, Simulated SNR: {}'.format(i,1./np.nanstd(res[mm])))
        else:
            print('NORMAL MODE')
            if bool(self.filename.find('Goldilocks')+1):
                self.e_sci = (self.hdulist[4].data)*self.exptime
                self.e_sky = (self.hdulist[5].data)*self.exptime*self.SKY_SCALING_FACTOR
                self.e_cal = (self.hdulist[6].data)*self.exptime
                if serval_config.sky_subtract:
                    self.e = np.sqrt(self.hdulist[4].data**2 + self.hdulist[5].data**2)*self.exptime
                else:
                    self.e = (self.hdulist[4].data)*self.exptime
            else:
                self.e_sci = np.sqrt(self.hdulist[4].data)*self.exptime
                self.e_sky = np.sqrt(self.hdulist[5].data)*self.exptime*self.SKY_SCALING_FACTOR
                self.e_cal = np.sqrt(self.hdulist[6].data)*self.exptime
                if serval_config.sky_subtract:
                    self.e = np.sqrt(self.hdulist[4].data + self.hdulist[5].data)*self.exptime
                else:
                    self.e = np.sqrt(self.hdulist[4].data)*self.exptime

        try:
            if bool(self.filename.find('Goldilocks')+1):
                self.jd_midpoint = astropy.time.Time(self.header['DATE-OBS'],format='isot').jd
                print('Using date observed from Goldilocks')
            else:
                midpoint_keywords = ['JD_FW{}'.format(i) for i in range(28)]
                self.jd_midpoint = np.median(np.array([self.header[i] for i in midpoint_keywords]))
                print('Using FW midpoints')
        except Exception as e:
            print('Does not have an FW exposuretime midpoint, setting as naive midpoint')
            self.jd_midpoint = self.jd_start + self.tmmean*self.exptime/(24.*60.*60.)
        self.flag = 0
        self.bflag = 0
        self.drift = 0.
        self.sn5  = np.nanmedian(self.f[5]/self.e[5])
        self.sn14 = np.nanmedian(self.f[14]/self.e[14])
        self.sn15 = np.nanmedian(self.f[15]/self.e[15])
        self.sn17 = np.nanmedian(self.f[17]/self.e[17])
        self.sn18 = np.nanmedian(self.f[18]/self.e[18])
        self.sn55 = self.sn18
        print('SNR18',self.sn18)
        # get maxexptime for 
        self.gitcommit = self.header.get('E_GIT','')
        self.airmass = self.header.get('AIRMASS', np.nan)
        self.ra = None#hdr['RA']
        self.de = None#hdr['DEC']
        self.obs.lon = serval_config.instruments[self.inst]['longitude']
        self.obs.lat = serval_config.instruments[self.inst]['latitude']
        self.drsbjd = None 
        self.drsberv = None
        self.calmode = 'OBJ,SKY'
        if verb: 
            print(("Read HPF:", self.timeid, self.header['OBJECT'], self.drsbjd, self.sn55, self.drsberv, self.drift, self.flag, self.calmode))
        self.bpmap = (np.isnan(w) | np.isnan(self.f) | np.isnan(self.e))*1.
        self.bpmap[17,1135:1190] = flag.sky # issue with order 17 in sky fib

        # Used originally
        #self.bpmap = np.zeros([self.f.shape[0],self.f.shape[1]])
        ##self.bpmap[17,1105] = flag.sky # issue with order 17
        ##self.bpmap[17,1106] = flag.sky # issue with order 17
        #self.bpmap[4,166:169] = flag.sky # issue with order 17
        #self.bpmap[16,69:76] = flag.sky     #
        ##self.bpmap[17,1151:1161] = flag.sky # issue with order 17
        #self.bpmap[18,647:650] = flag.sky # issue with order 17
        #self.bpmap[26,1358:1360] = flag.sky # issue with order 17
        #self.bpmap[26,720:760] = flag.sky   # testing masking out bright absorption line
        ##self.bpmap[17,1107] = flag.sky # issue with order 17
        #######
        # Try variance downweigting for badpixels
        #bp = np.array([[17,1156]])
        #for b in bp:
        #    self.e[b[0],b[1]] *=serval_config.badpixel_downweight
        #bp = np.array([[17,1156]])
        ##Interpolating over bad pixels
        #for _b in bp:
        #    o,i=_b[0],_b[1]
        #    self.f[o,i] = (self.f[o,i-1]+self.f[o,i-1])/2.
        #    self.e[o,i] = (self.e[o,i-1]+self.e[o,i-1])/2.
        ########
        #interpolate bad columns
        #bp = self.bpmap
        #f[bp[0],bp[1]] = (f[bp[0],bp[1]-1]+f[bp[0],bp[1]+1]) / 2
        return w, self.f, self.e, self.bpmap

def write_template(filename, flux, wave, header=None, hdrref=None, clobber=False):
    """
    Write template to a fits file
    """
    if not header and hdrref: header = pyfits.getheader(hdrref)
    hdu = pyfits.PrimaryHDU(header=header)
    warnings.resetwarnings() # supress nasty overwrite warning http://pythonhosted.org/pyfits/users_guide/users_misc.html
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    hdu.writeto(filename, overwrite=clobber, output_verify='fix')
    warnings.resetwarnings()
    warnings.filterwarnings('always', category=UserWarning, append=True)

    if isinstance(flux, np.ndarray):
        pyfits.append(filename, flux)
        pyfits.append(filename, wave)
    else:
        # pad arrays with zero to common size
        maxpix = max(arr.size for arr in flux if isinstance(arr, np.ndarray))
        flux_new = np.zeros((len(flux), maxpix))
        wave_new = np.zeros((len(flux), maxpix))
        for o,arr in enumerate(flux):
            if isinstance(arr, np.ndarray): flux_new[o,:len(arr)] = arr
        for o,arr in enumerate(wave):
            if isinstance(arr, np.ndarray): wave_new[o,:len(arr)] = arr
        pyfits.append(filename, flux_new)
        pyfits.append(filename, wave_new)

    pyfits.setval(filename, 'EXTNAME', value='SPEC', ext=1)
    pyfits.setval(filename, 'EXTNAME', value='WAVE', ext=2)
    #fitsio.write(filename, flux)

def write_res(filename, datas, extnames, header='', hdrref=None, clobber=False):
   if not header and hdrref: header = pyfits.getheader(hdrref)
   hdu = pyfits.PrimaryHDU(header=header)
   warnings.resetwarnings() # supress nasty overwrite warning http://pythonhosted.org/pyfits/users_guide/users_misc.html
   warnings.filterwarnings('ignore', category=UserWarning, append=True)
   hdu.writeto(filename, overwrite=clobber, output_verify='fix')
   warnings.resetwarnings()
   warnings.filterwarnings('always', category=UserWarning, append=True)

   for i,extname in enumerate(extnames):
     data = datas[extname]
     if isinstance(data, np.ndarray):
        pyfits.append(filename, data)
     else:
        1/0
     pyfits.setval(filename, 'EXTNAME', value=extname, ext=i+1)
   #fitsio.write(filename, flux)

def write_fits(filename, data, header='', hdrref=None, clobber=True):
   if not header and hdrref: header = pyfits.getheader(hdrref)
   warnings.resetwarnings() # supress nasty overwrite warning http://pythonhosted.org/pyfits/users_guide/users_misc.html
   warnings.filterwarnings('ignore', category=UserWarning, append=True)
   pyfits.writeto(filename, data, header, overwrite=clobber, output_verify='fix')
   warnings.resetwarnings()
   warnings.filterwarnings('always', category=UserWarning, append=True)

def read_template(filename):
   hdu = pyfits.open(filename)
   #return hdu[1].data, hdu[0].data  # wave, flux
   return hdu[2].data, hdu[1].data, hdu[0].header  # wave, flux

