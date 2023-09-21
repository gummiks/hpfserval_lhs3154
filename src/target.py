from __future__ import print_function
import barycorrpy
import barycorrpy.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import bary
import configparser
import os
import serval_config
from astroquery.mast import Catalogs
from astropy.time import Time
from six.moves.urllib.request import urlopen
path_targets = serval_config.path_targets

class Target(object):
    """
    Simple target class. Capable of querying SIMBAD. Can calculate barycentric corrections.
    
    EXAMPLE:
        T = Target('GJ_699')
        T.calc_barycentric_velocity(H.jd_midpoint,'McDonald Observatory')
    """
    def __init__(self,name,config_folder=path_targets):
        self.config_folder = config_folder
        self.config_filename = self.config_folder + os.sep + name + '.config'
        self.name = name
        try:
            self.data = self.from_file()
        except Exception as e:
            print(e,'File does not exist!')
            if 'TIC' in name:
                print('Querying TIC for data')
                self.data = self.query_tic(name)
            elif 'TOI' in name:
                print('Querying TOI list for data')
                self.data = self.query_toi(name)
            else:
                print('Querying SIMBAD for data')
                self.data, self.warning = barycorrpy.utils.get_stellar_data(name)
            self.to_file(self.data)
        self.ra = self.data['ra']
        self.dec = self.data['dec']
        self.pmra = self.data['pmra']
        self.pmdec = self.data['pmdec']
        self.px = self.data['px']
        self.epoch = self.data['epoch']
        if self.data['rv'] is None or self.data['rv'] is np.nan:
            self.rv = 0.
        else:
            self.rv = self.data['rv']/1000.

    def query_tic(self,ticname):
        name = ticname.replace('-',' ').replace('_',' ')
        df = Catalogs.query_object(name, radius=0.0003, catalog="TIC").to_pandas()[0:1]
        data = {}
        data['ra'] = df.ra.values[0]
        data['dec'] = df.dec.values[0]
        data['pmra'] = df.pmRA.values[0]
        data['pmdec'] = df.pmDEC.values[0]
        data['px'] = df.plx.values[0]
        data['epoch'] = Time(2015.5, format = 'decimalyear').jd if (df.loc[0,'PMflag'] == 'gaia2') else 2451545.0
        data['rv'] = 0.
        return data

    def query_toi(self,toiname):
        name = toiname.replace('-',' ').replace('_',' ')
        toicat = pd.read_csv(urlopen('https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=pipe'),sep='|',comment='#')
        toicat = toicat[np.isin(toicat.TOI.values.astype(int),[int(name.split()[-1])])].reset_index(drop=True)
        df = Catalogs.query_object('TIC {:0.0f}'.format(toicat.loc[0,'TIC ID']), radius=0.0003, catalog="TIC").to_pandas()[0:1]
        data = {}
        data['ra'] = df.ra.values[0]
        data['dec'] = df.dec.values[0]
        data['pmra'] = df.pmRA.values[0]
        data['pmdec'] = df.pmDEC.values[0]
        data['px'] = df.plx.values[0]
        data['epoch'] = Time(2015.5, format = 'decimalyear').jd if (df.loc[0,'PMflag'] == 'gaia2') else 2451545.0
        data['rv'] = 0.
        return data

    def from_file(self):
        """
        Read from file
        """
        print('Reading from file {}'.format(self.config_filename))
        #if os.path.exists(self.config_filename):
        config = configparser.ConfigParser()
        config.read(self.config_filename)
        data = dict(config.items('targetinfo'))
        for key in data.keys():
            if data[key] == 'None':
                data[key] = None
            else:
                data[key] = float(data[key])
        return data

    def to_file(self,data):
        """
        Save to file
        """
        print('Saving to file {}'.format(self.config_filename))
        config = configparser.ConfigParser()
        config.add_section('targetinfo')
        for key in data.keys():
            config.set('targetinfo',key,str(data[key]))
            print(key,data[key])
        with open(self.config_filename,'w') as f:
            config.write(f)
        print('Done')
        
    def calc_barycentric_velocity(self,jdtime,obsname):
        """
        Calculate barycentric velocity 
        
        OUTPUT:
            BJD_TDB
            berv in km/s
        
        EXAMPLE:
            bjd, berv = bary.bjdbrv(H.jd_midpoint,T.ra,T.dec,obsname='McDonald Observatory',
                           pmra=T.pmra,pmdec=T.pmdec,rv=T.rv,parallax=T.px,epoch=T.epoch)
        """
        bjd, berv = bary.bjdbrv(jdtime,self.ra,self.dec,obsname='McDonald Observatory',
                                   pmra=self.pmra,pmdec=self.pmdec,rv=self.rv,parallax=self.px,epoch=self.epoch)
        return bjd, berv/1000.
    
    def __repr__(self):
        return "{}, ra={:0.4f}, dec={:0.4f}, pmra={}, pmdec={}, rv={:0.4f}, px={:0.4f}, epoch={}".format(self.name,
                                            self.ra,self.dec,self.pmra,self.pmdec,self.rv,self.px,self.epoch)
