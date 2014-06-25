'''
Script to homogenize station data using the Pairwise Homogenization Algorithm:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.
'''
import os
import twx
from twx.homog import HomogDaily
from twx.utils import DATE

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_PHA_RUNS = os.path.join(PROJECT_ROOT, 'inhomo_software')
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    path_pha_tar = os.path.join(FPATH_PHA_RUNS,'pha_src','phav52i.tar.gz')
    
    path_tmin_pha_src = os.path.join(FPATH_PHA_RUNS, 'pha_v52i_tmin', 'src')
    path_tmin_pha_run = os.path.join(FPATH_PHA_RUNS, 'pha_v52i_tmin', 'run')
    
    path_tmax_pha_src = os.path.join(FPATH_PHA_RUNS, 'pha_v52i_tmax', 'src')
    path_tmax_pha_run = os.path.join(FPATH_PHA_RUNS, 'pha_v52i_tmax', 'run')
    
    path_stndb = os.path.join(FPATH_STNDATA, 'all', 'tair_tobs_adj_1948_2012.nc')
    
    stnda = twx.db.StationDataDb(path_stndb)
    
    yr_begin = 1948
    yr_end = 2012
    stns = stnda.stns
    
    #Perform PHA setup for Tmin
    mthly_tmin = stnda.ds.variables['tmin_mth'][:]
    twx.homog.setup_pha(path_pha_tar, path_tmin_pha_src, path_tmin_pha_run, 
                        yr_begin, yr_end, stns, mthly_tmin, 'tmin')
    #Remove monthly Tmin observations from memory
    mthly_tmin = None
    
    #Perform PHA setup for Tmax
    mthly_tmax = stnda.ds.variables['tmax_mth'][:]
    twx.homog.setup_pha(path_pha_tar, path_tmax_pha_src, path_tmax_pha_run, 
                        yr_begin, yr_end, stns, mthly_tmax, 'tmax')
    #Remove monthly Tmax observations from memory
    mthly_tmax = None
    
    #Run PHA for Tmin
    twx.homog.run_pha(path_tmin_pha_run, 'tmin')
    
    #Run PHA for Tmax
    twx.homog.run_pha(path_tmax_pha_run, 'tmax')
    
    #Use PHA results to homogenize daily station data and insert
    #into new homogenized database
    homog_tmin = HomogDaily(stnda, path_tmin_pha_run, 'tmin')
    homog_tmax = HomogDaily(stnda, path_tmax_pha_run, 'tmax')
    
    path_out_homog_db = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc')    
    insert_homog = twx.homog.InsertHomog(stnda, homog_tmin, homog_tmax, path_tmin_pha_run, path_tmax_pha_run)
    twx.db.create_netcdf_db(path_out_homog_db, stnda.days[DATE][0], stnda.days[DATE][-1], [insert_homog])
    twx.db.insert_data_netcdf_db(path_out_homog_db, [insert_homog])
    