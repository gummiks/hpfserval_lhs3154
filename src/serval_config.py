from scipy import interpolate
import cubicSpline
import os
import numpy as np
from nameddict import nameddict

##########################################
def_wlog = False
inst="HPF"
flat_mode= "auto"
sky_subtract = True
simulation_mode = False
SKY_SCALING_FACTOR = 1.
spltype = 1#3
fixed_wavelength_solution = False
flag_template_tellurics = False
##########################################
# PATHS
path_master = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
path_targets = os.path.join(path_master,'targets')
path_src = os.path.join(path_master,'src')
path_lib = os.path.join(path_master,'lib')
path_libwave  = os.path.join(path_lib,'wavelength_solution')
path_libflats = os.path.join(path_lib,'master_flats')
path_libmasks = os.path.join(path_lib,'masks')
path_to_badfiles = os.path.join(path_master,'lib','bad_files','bad_files.csv')
OUTPUT_DIR = os.path.join(path_master,'website','uploading_RC','data','targets')
print("USING INSTRUMENT: {}".format(inst))
##########################################
# SPLINE METHOD
spline_cv = {1: interpolate.splrep, 2: cubicSpline.spl_c,  3: cubicSpline.spl_cf }[spltype]
spline_ev = {1: interpolate.splev,  2: cubicSpline.spl_ev, 3: cubicSpline.spl_evf}[spltype]
if spltype==1:
    print("Using spline_cv = interpolate.splrep")
if spltype==2:
    print("Using spline_cv = cubicSpline.spl_c")
if spltype==3:
    print("Using spline_cv = cubicSpline.spl_cf")
##########################################

inst_hpf = {"iomax"    : 28,
            "location" : "McDonald Observatory",
            "oset"     : slice(0,28),
            "fib"      : "A",
            "longitude": -104.0147,
            "latitude" : 30.6814,
            "altitude" : 2025.,
            "pmin"     : 100,  
            "pmax"     : 1950, 
            "npix"     : 2048,
            "path_wavelength_solution_sci" : path_libwave + os.sep + "20181012" + os.sep + "LFC_wavecal_scifiber_v2.fits", 
            "path_wavelength_solution_sky" : path_libwave + os.sep + "20181012" + os.sep + "LFC_wavecal_skyfiber_v2.fits", 
            "path_flat_noflat": path_libflats + os.sep +  "20180804" + os.sep + "alphabright_fcu_noflat.fits",
            "path_flat_sept18":  path_libflats + os.sep +  "20180918" + os.sep + "alphabright_fcu_sept18_deblazed.fits",
            "path_flat_may25": path_libflats + os.sep + "20180808" + os.sep + "alphabright_fcu_may25_july21_deblazed.fits",
            "path_telluricmask" : path_libmasks + os.sep + "telluric_mask" + os.sep + "telfit_telmask_conv17_thres0.995_with17area.dat",
            "path_skymask"      : path_libmasks + os.sep + "sky_mask" + os.sep + "HPF_SkyEmmissionLineWavlMask_broadened_11111_Compressed.txt",
            "path_stellarmask"  : path_libmasks + os.sep + "star_mask" + os.sep + "gj699_stellarmask_LINET_01_20180816_1.53ms_shifted_withdeeplineo17.txt", 
            "snmin"    : 10,
            "snmax"    : 700,
            "good_orders" : [4, 5, 6, 14, 15, 16, 17, 18] 
            }

instruments = { "HPF"   : inst_hpf}

iomax                    = instruments[inst]["iomax"]
oset                     = instruments[inst]["oset"]
obs_longitude            = instruments[inst]["longitude"]
obs_latitude             = instruments[inst]["latitude"]
obs_altitude             = instruments[inst]["altitude"]
fib                      = instruments[inst]["fib"]
pmin                     = instruments[inst]["pmin"]
pmax                     = instruments[inst]["pmax"]
npix                     = instruments[inst]["npix"]
path_to_tellmask_file    = instruments[inst]["path_telluricmask"]
path_to_skymask_file     = instruments[inst]["path_skymask"]
path_to_stellarmask_file = instruments[inst]["path_stellarmask"]
snmin                    = instruments[inst]["snmin"]
snmax                    = instruments[inst]["snmax"]
print('PMIN',pmin)
print('PMAX',pmax)

##########################################
# GENERAL SETTINGS
badpixel_downweight = 10000.
orders = np.arange(iomax)[oset]
ptmin = pmin - 50 # Oversize template
ptmax = pmax + 50
ntpix = ptmax - ptmin
pixx = np.arange(ntpix)
pixxx = np.arange((ntpix-1)*4.+1) / 4.
knotoptmult = 1. 
osize = len(pixxx)
verbose = True
ofac = 1. # oversampling factor in coadding'
kapsig = 3 # kappa sigma clip value
deg = 5
nclip = 2
niter = 2 # iterations for sigma clipping
telluric_downweight_factor = 0.1
pspllam = 0.0000001 # Coadding smoothing value
ckappa = (4.,4.) #coadding kappa kappa sigma (or lower and upper) clip value in coadding. Zero values for no clipping
msksky = [0] * iomax 
v_lo = -5.5 
v_hi = 5.6  
v_step = 0.1 
safemode = False
##########################################
print('USING Stellarmask: {}'.format(path_to_stellarmask_file))

# bpmap flagging
flag = nameddict(
   ok=       0, # good pixel
   nan=      1, # nan flux in pixel of spectrum
   neg=      2, # significant negative flux in pixel of spectrum (f < -3*f_err < 0 && )
   sat=      4, # too high flux (saturated)
   atm=      8, # telluric at wavelength of spectrum
   sky=     16, # sky emission at wavelength of spectrum
   out=     32, # region outside the template
   clip=    64, # clipped value
   lowQ=   128, # low Q in stellar spectrum (mask continuum)
   badT=   256, # bad corresponding region in the template spectrum
   star=   512, # star active region
)

# spectrum flagging
sflag = nameddict(
   iod=      2,
   eggs=     4,
   dist=    16, # coordinates too much off
   daytime= 32, # not within nautical twilight
   lowSN=   64, # too low S/N
   hiSN=   128, # too high S/N
   led=    256, # LED on during observation (CARM_VIS)
   rvnan=  512
)

# bpmap flags
flag_cosm = flag.sat  # @ FEROS for now use same flag as sat
def_wlog = False
brvref = ['DRS', 'MH', 'WEhtml', 'WEidl', 'WE']
