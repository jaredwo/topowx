'''
Script to run station location quality assurance for stations in netCDF database.
'''
import os
import twx
from twx.db.station_data import StationDataDb
import numpy as np

if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')

    # Location quality assurance of stations in netCDF station database

    # 1.) Update locations of stations that had their locations previously corrected
    fpath_db = os.path.join(FPATH_STNDATA, 'all', 'all_1948_2012.nc')
    path_prev_locqa = os.path.join(FPATH_STNDATA, 'all', 'qa_elev_final.csv')
    twx.qa.update_stn_locs(fpath_db, path_prev_locqa)

    # 2.) Run new location quality assurance check
    #Set global username for Geonames
    twx.qa.set_usrname_geonames(open('/home/jared.oyler/.geonames_username').readline().strip())
    path_new_locqa = os.path.join(FPATH_STNDATA, 'all', 'qa_elev_new.csv')
    path_por = os.path.join(FPATH_STNDATA, 'all', 'all_por_1948_2012.csv')
    stndb = StationDataDb(fpath_db)
    a_por = twx.db.load_por_csv(path_por)
    mask_por_tmin, mask_por_tmax = twx.db.build_valid_por_masks(a_por)

    # Only run qa for stations that have required period of record and that
    # have not been qa'd before
    stns = stndb.stns[np.logical_or(mask_por_tmin, mask_por_tmax)]
    twx.qa.qa_stn_locs(stns, path_new_locqa, path_prev_locqa)
    # Manually investigate qa file for stations that have a elevation that
    # differs significantly from DEM elevation. Update LON_NEW, LAT_NEW, and ELEV_NEW
    # fields in the qa file for any station that should have its location corrected.

    # 3.) Combine new and old location qa csv files into final qa file
    path_fnl_new_locqa = os.path.join(FPATH_STNDATA, 'all', 'qa_elev_combine.csv')
    twx.qa.combine_locqa(path_prev_locqa, path_new_locqa, path_fnl_new_locqa)

    # 4.) Update station locations with any new corrected locations
    twx.qa.update_stn_locs(fpath_db, path_fnl_new_locqa)
