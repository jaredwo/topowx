'''
Script for preparing NCEP/NCAR reanalysis data from
ftp://ftp.cdc.noaa.gov/Datasets/ncep.reanalysis
for input to the TopoWx missing value infilling procedures.
'''

import os
from twx.db import add_utc_offset, create_nnr_subset, StationDataDb,\
create_thickness_nnr_subset, create_nnr_subset_nolevel
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
    #Path to timezone shapefile that has UTC offsets
    #Downloaded from http://www.sharegeo.ac.uk/handle/10672/285
    fpath_timezone_shp = os.path.join(PROJECT_ROOT, 'dem', 'timezones', 'world_timezones.shp')
    add_utc_offset(path_homog_db, fpath_timezone_shp, geonames_usrname)
    
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
    
    #500mb-1000mb Thickness at 12z, 18z, and 24z
    #Requires hgt.[year].nc files as input
    create_thickness_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_thick_12z.nc'),
                                yrs, days, 500, 1000, 12)
    create_thickness_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_thick_18z.nc'),
                                yrs, days, 500, 1000, 18)
    create_thickness_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_thick_24z.nc'),
                                yrs, days, 500, 1000, 24)
    
    #500mb and 700mb height at 12z, 18z, and 24z
    #Requires hgt.[year].nc files as input
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_hgt_12z.nc'),
                      yrs, days,'hgt','hgt', np.array([500,700]), 12, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_hgt_18z.nc'),
                      yrs, days,'hgt','hgt', np.array([500,700]), 18, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_hgt_24z.nc'),
                      yrs, days,'hgt','hgt', np.array([500,700]), 24, no_conv)

    #850mb relative humidity at 12z, 18z, and 24z
    #Requires rhum.[year].nc files as input
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_rhum_12z.nc'),
                      yrs, days,'rhum','rhum', np.array([850]), 12, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_rhum_18z.nc'),
                      yrs, days,'rhum','rhum', np.array([850]), 18, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_rhum_24z.nc'),
                      yrs, days,'rhum','rhum', np.array([850]), 24, no_conv)
    
    #850mb v-wind at 12z, 18z, and 24z
    #Requires vwnd.[year].nc files as input
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_vwnd_12z.nc'),
                      yrs, days,'vwnd','vwnd', np.array([850]), 12, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_vwnd_18z.nc'),
                      yrs, days,'vwnd','vwnd', np.array([850]), 18, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_vwnd_24z.nc'),
                      yrs, days,'vwnd','vwnd', np.array([850]), 24, no_conv)
    
    #850mb u-wind at 12z, 18z, and 24z
    #Requires uwnd.[year].nc files as input
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_uwnd_12z.nc'),
                      yrs, days,'uwnd','uwnd', np.array([850]), 12, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_uwnd_18z.nc'),
                      yrs, days,'uwnd','uwnd', np.array([850]), 18, no_conv)
    create_nnr_subset(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_uwnd_24z.nc'),
                      yrs, days,'uwnd','uwnd', np.array([850]), 24, no_conv)

    #Sea level pressure at 12z, 18z, and 24z
    #Requires slp.[year].nc files as input
    create_nnr_subset_nolevel(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_slp_12z.nc'),
                              yrs, days, 'slp', 'slp', 12, no_conv)
    create_nnr_subset_nolevel(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_slp_18z.nc'),
                              yrs, days, 'slp', 'slp', 18, no_conv)
    create_nnr_subset_nolevel(FPATH_NNR, os.path.join(FPATH_NNR_SUBSETS,'nnr_slp_24z.nc'),
                              yrs, days, 'slp', 'slp', 24, no_conv)
