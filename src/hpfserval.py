import matplotlib
import pickle
matplotlib.use("agg")
import glob
import seaborn as sns
sns.set()
sns.set_style("white")
cp = sns.color_palette("colorblind")
import serval_help
import os, subprocess
import serval_help
import serval_config
import utils
import numpy as np
import serval_plotting
from matplotlib import rcParams
import argparse
import pdb
import spec_help
import pandas as pd
import filepath
from six.moves.urllib.request import urlopen
rcParams["axes.formatter.useoffset"] = False
rcParams["mathtext.fontset"] = "stix"
rcParams["font.family"] = "STIXGeneral"
rcParams["ytick.major.size"] = 7
rcParams["xtick.major.size"] = 7
rcParams["ytick.minor.size"] = 3
rcParams["xtick.minor.size"] = 3
rcParams["ytick.direction"] = "in"
rcParams["xtick.direction"] = "in"

# Parse arguments
parser = argparse.ArgumentParser(description="Reduce HPF RVs")
parser.add_argument("Object",type=str,help="HPF object to reduce (SIMBAD Queryable)")
parser.add_argument("--simbadname",type=str,default="",help="Specify SIMBADNAME to reduce")
parser.add_argument("--datestart",type=str,default="",help="Date to start, e.g., 2019-09-01")
parser.add_argument("--dateend",type=str,default="",help="Date to end, e.g., 2019-09-01")
parser.add_argument("--inputfolder",type=str,default="",help="Specify input foldername to save (e.g., ../my_star)")
parser.add_argument("--inputfile",type=str,default="",help="Specify input file with list of files to analyze")
parser.add_argument("--foldername",type=str,default="results",help="Specify foldername to save (e.g. results_123)")
parser.add_argument("--prervs",help="Run Pre RVs",action="store_true")
parser.add_argument("--onlyupdate",help="Only run if there are new observations to run",action="store_true")
parser.add_argument("--plottrv",help="Plot Template RV Panels",action="store_true")
parser.add_argument("--skipfilterbadfiles",help="Skip Filtering Bad files",action="store_true")
parser.add_argument("--npass",help="Perform npass RV iterations instead of one, set to 0 if you want only 1 pass",type=int,default=5)
parser.add_argument("--skipknots",help="Skip knot optimization",action="store_true")
parser.add_argument("--skiprv",help="Skip reducing RVs",action="store_true")
parser.add_argument("--startloc",type=int,default=0,help="File location to start reducing")
parser.add_argument("--snrmin",type=int,default=10,help="Minimum SNR to use")
parser.add_argument("--snrmax",type=int,default=800,help="Maximum SNR to use")
parser.add_argument("--vref",type=float,default=0.,help="Absolute RV")
parser.add_argument("--orders",type=int,default=[4,5,6,14,15,16,17,18],help="Orders to use for RV extraction, e.g., --orders 4 5 6",nargs='+')

args = parser.parse_args()
target = args.Object
print('will save to',args.foldername+os.sep)

# Database paths
path_to_badfiles = serval_config.path_to_badfiles
OUTPUT_DIR = serval_config.OUTPUT_DIR
print('################')
print('Saving files to: ',OUTPUT_DIR)
print('################')

if args.inputfolder!='':
    print('Looking for .fits files in folder: {}'.format(args.inputfolder))
    _files1 = sorted(glob.glob(args.inputfolder+os.sep+"*.fits"))
    _files2 = sorted(glob.glob(args.inputfolder+os.sep+"*/*/*.fits"))
    files = sorted(_files1+_files2)
    print('Found the following fits files: {}'.format(files))
else:
    inputfile = args.inputfile
    files = pd.read_csv(inputfile,header=None,names=['files'],comment='#').files.values
    m = np.array([os.path.isfile(i) for i in files])
    print('Following files do not exist:')
    print(files[~m])
    files = files[m]

# Only using good dates
dff = utils.grep_dates(files,intype="isot",outtype="series")

if args.datestart != "" and args.dateend != "":
    print("Starting ",args.datestart,"ending",args.dateend)
    files = sorted(dff[args.datestart+" 00:00:00.00":args.dateend+" 00:00:00.00"].values)
elif args.datestart != "" and args.dateend == "":
    print("Starting ",args.datestart)
    files = sorted(dff[args.datestart+" 00:00:00.00":"2030-09-14 00:00:00.00"].values)
elif args.datestart == "" and args.dateend != "":
    print("Ending ",args.dateend)
    files = sorted(dff["2018-04-20 00:00:00.00":args.dateend+" 00:00:00.00"].values)
else:
    #_files1 = dff["2018-01-20 00:00:00.00":"2018-09-14 00:00:00.00"].values
    _files1 = dff["2018-04-20 00:00:00.00":"2018-09-14 00:00:00.00"].values
    _files2 = dff["2018-09-16 00:00:00.00":"2018-12-24 00:00:00.00"].values
    _files3 = dff["2018-12-28 00:00:00.00":"2019-06-06 00:00:00.00"].values
    _files4 = dff["2019-06-07 00:00:00.00":"2030-12-24 00:00:00.00"].values
    files = sorted(np.concatenate([_files1,_files2,_files3,_files4]))


if args.skipfilterbadfiles == False:
    files = np.array(files)
    df_badfiles = pd.read_csv(path_to_badfiles,comment='#')
    _base_bad = [os.path.basename(i) for i in df_badfiles.filename.values]
    _base_files = [os.path.basename(i) for i in files]
    m = pd.DataFrame(_base_files).isin(_base_bad)[0].values
    print('Excluding following bad files:')
    print(files[m])
    files = files[~m]

if args.simbadname != "":
    target_simbadname = args.simbadname
else:
    target_simbadname = target
print("#########################")
print("Using {} for SIMBAD querying. TARGET={}".format(target_simbadname,target))
print("#########################")

if args.skiprv!=True:
    directory = os.path.join(OUTPUT_DIR,target,args.foldername+os.sep)
    fp = filepath.FilePath(directory)
    filename_csv_unbin   = os.path.join(fp._fullpath,'{}_rv_unbin.csv'.format(target))
    if os.path.isfile(filename_csv_unbin) and args.onlyupdate == True:
        _d = pd.read_csv(filename_csv_unbin)
        #print(_d.filename.values,len(_d))
        #print(np.array(files[args.startloc:]),len(np.array(files[args.startloc:])))
        #print(_d.filename.values==np.array(files[args.startloc:]))

        if np.array_equal(_d.filename.values,np.array(files[args.startloc:])):
            print('#############################')
            print('#############################')
            print('Onlyupdate==True, and {} spectra have already been extracted'.format(len(_d)))
            print('#############################')
            print('#############################')
            os.sys.exit()

    spall, spok, spt, spi = serval_help.read_spectra(np.array(files[args.startloc:]),
                                                     targetname=target_simbadname,
                                                     read_data=True,
                                                     inst="HPF")
    print("#########################")
    print("Making SNR cut of: {:5.1f}<SNR<{:5.1f}".format(args.snrmin,args.snrmax))
    print("#########################")
    snrmask = (spall.df.sn18 > args.snrmin) & (spall.df.sn18 < args.snrmax)
    print("Found {}/{} spectra with SNR within SNRmask".format(np.sum((snrmask)),len(snrmask)))
    #pdb.set_trace()
    spall.splist = np.array(spall.splist)[snrmask]

    # With Pre RVs
    estimate_optimal_knots = args.skipknots == False
    hh = serval_plotting.master_rv_project(spall,spt,masterdir=OUTPUT_DIR,
                                           target=target,orders=args.orders,
                                           subdir=args.foldername+os.sep,
                                           title='HPF RVs',
                                           prervs=args.prervs,
                                           npass=args.npass,
                                           estimate_optimal_knots=estimate_optimal_knots,
                                           vref=args.vref,
                                           plot_template_rvs=args.plottrv,
                                           onlyupdate=args.onlyupdate)


