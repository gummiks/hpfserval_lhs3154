# HPF-SERVAL for LHS 3154 b

Adapted version of SERVAL for use with HPF.

Scripts are included to a) extract the RVs, and b) reproduce the RV figure from Stefansson et al. 2023 (Fig 1; https://ui.adsabs.harvard.edu/abs/2023arXiv230313321S/abstract).


# Installation
You will need to run the following:

### Linux

```
cd src/
python -m numpy.f2py -c -m spl_int spl_int.f
gcc -c  -Wall -O2 -ansi -pedantic -fPIC cbspline.c; gcc -o cbspline.so -shared cbspline.o
gcc -c  -Wall -O2 -ansi -pedantic -fPIC polyregression.c; gcc -o polyregression.so -shared polyregression.o
```

### Mac
If you are on a newer Mac (M1 chip), then add 'arch -x86_64 in front':
```
cd src/
python -m numpy.f2py -c -m spl_int spl_int.f
arch -x86_64 gcc -c  -Wall -O2 -ansi -pedantic -fPIC cbspline.c
arch -x86_64 gcc -o cbspline.so -shared cbspline.o
arch -x86_64 gcc -c  -Wall -O2 -ansi -pedantic -fPIC polyregression.c
arch -x86_64 gcc -o polyregression.so -shared polyregression.o
```

If you have issues with the f2py command (e.g., due to it calling an old gcc compiler) saying `'for' loop initial declarations are only allowed in C99 mode'`, then try running:
```
export CFLAGS=-std=c99
```
and then rerunning the command.

# Dependencies
- seaborn
- glob2
- colorlog
- barycorrpy

# Getting Started - Running LHS 3154 RVs from Stefansson et al. 2023
Try running the minimal example script of:

```
cd scripts/
./run_lhs3154.sh
```

Which runs the following command:
```
python ../src/hpfserval.py --npass 1 --inputfolder ../input_data/lhs_3154_spectra/ --foldername results --vref 9.609 LHS_3154
```
This extracts the RVs and activity indicators from the 137 HPF spectra that are available in the `lhs_3154_spectra/` directory.

Once the script has finished running, results will be saved in the `website/uploading_RC/data/targets/LHS_3154/results/` directory.
The `LHS_3154_unbin.csv` file contains the RV per extracted spectrum and associated uncertainties (in m/s) and activitiy indicators.
The LHS_3154_bin.csv` file contains RVs binned per HET track (see Stefansson et al. 2023 for a discusison on that).

# Regenerating figures from Stefansson et al. 2023
There are three steps:
- a) Download the full LHS 3154 spectra here: (insert Zenodo link here with 1D spectra).
- b) Run the script in the scripts/ directory. This creates the output files in the website/ directory.
- c) Run the ipython notebook in the notebook/ directory to regenerate Figure 1 in Stefansson et al. 2023.

# Citations
If you use this code, kindly cite the following papers:
- a) Zechmeister et al. 2018: https://ui.adsabs.harvard.edu/abs/2018A%26A...609A..12Z/abstract
- b) Stefansson et al. 2020: https://ui.adsabs.harvard.edu/abs/2020AJ....159..100S/abstract
- c) Stefansson et al. 2023: https://ui.adsabs.harvard.edu/abs/2023arXiv230313321S/abstract
