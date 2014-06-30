'''
Created on Jun 27, 2014

@author: jared.oyler
'''
import os
from twx.db import add_utc_offset, create_nnr_subset, StationDataDb
from twx.utils import YEAR
import numpy as np

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    #Path for NCEP/NCAR Reanalysis data downloaded from
    #ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis
    FPATH_NNR = os.path.join(PROJECT_ROOT, 'reanalysis_data')
    FPATH_NNR_SUBSETS = os.path.join(PROJECT_ROOT, 'reanalysis_data','na_subset')
    
    #Add a UTC offset attribute to the homogenized station data
    #This is needed for extracting NCEP/NCAR Reanalysis observations
    #that most closely align to a station's typical local timing of Tmin/Tmax
    path_homog_db = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc')
    geonames_usrname = open('/home/jared.oyler/.geonames_username').readline().strip()
    add_utc_offset(path_homog_db, geonames_usrname)
    
    #Create NCEP/NCAR Reanalysis North American subsets of variables used in the
    #infilling of missing station observations.
    
    stnda = StationDataDb(path_homog_db)
    days = stnda.days
    yrs = np.unique(days[YEAR])
    #Conversion functions
    k_to_c = lambda k: k - 273.15
    no_conv = lambda x: x
    
    #850mb Temperature at 12z, 18z, and 24z
    #Requires air.[year].nc files as input
    create_nnr_subset(FPATH_NNR,os.path.join(FPATH_NNR_SUBSETS,'nnr_tair_12z.nc'),
                      yrs, days,'air','tair', np.array([850]), 12, k_to_c)
    create_nnr_subset(FPATH_NNR,os.path.join(FPATH_NNR_SUBSETS,'nnr_tair_18z.nc'),
                      yrs, days,'air','tair', np.array([850]), 18, k_to_c)
    create_nnr_subset(FPATH_NNR,os.path.join(FPATH_NNR_SUBSETS,'nnr_tair_24z.nc'),
                  yrs, days,'air','tair', np.array([850]), 24, k_to_c)

