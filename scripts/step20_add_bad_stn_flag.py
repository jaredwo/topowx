'''
Script to set flags on bad and duplicate stations that should not be used for
interpolation.
'''

from twx.db import MASK, TDI, StationSerialDataDb, BAD, CLIMDIV
from twx.infill import set_bad_stations, find_dup_stns
from twx.interp import XvalOutlier
from twx.utils import TwxConfig
import numpy as np
import os


if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))

    stnda_tmin = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmin,
                                     'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmax,
                                     'tmax', mode='r+')
    stnda_infill_tmin = StationSerialDataDb(twx_cfg.fpath_stndata_nc_infill_tmin,
                                            'tmin')
    stnda_infill_tmax = StationSerialDataDb(twx_cfg.fpath_stndata_nc_infill_tmax,
                                            'tmax')
    
    # Load station ids that were marked as "bad" due to infilling issues
    bad_stnids = np.unique(np.loadtxt(twx_cfg.fpath_flagged_bad_stns, 
                                      delimiter=",", dtype=np.str, skiprows=1,
                                      usecols=(0,)))
    
    for a_stnda, a_istnda in zip([stnda_tmin, stnda_tmax],
                                 [stnda_infill_tmin, stnda_infill_tmax]):
        
        # Add a 'bad' station variable
        a_stnda.add_stn_variable(BAD, "bad station flag", units='', dtype='i1',
                                 fill_value=0)

        # Find stations that do not have a TDI value. All
        # stations should have a TDI value. If a TDI value could not be obtained for
        # a station, it lies far outside the applicable DEM domain.
        stnids_no_tdi = a_stnda.stn_ids[np.isnan(a_stnda.stns[TDI])]
    
        # Find stations that are in the interpolation mask, but do not have a climate division.
        # All stations within the interpolation mask should fall within a climate division
        mask_no_climdiv = np.logical_and(np.isnan(a_stnda.stns[CLIMDIV]),
                                         np.isfinite(a_stnda.stns[MASK]))
        stnids_no_climdiv = a_stnda.stn_ids[mask_no_climdiv]
        
        # Find duplicate stations.
        # The kriging interpolation algorithm cannot have point observations
        # at the exact same location. For two or more stations with the same
        # location, the one with the longest non-infilled period-of-record is
        # used and the others are considered duplicates.
        print "Finding duplicate stations for %s..." % (a_stnda.var_name,)
        dup_stnids = find_dup_stns(a_istnda)
        
        all_rm_stnids = np.unique(np.concatenate([bad_stnids, stnids_no_tdi,
                                                  stnids_no_climdiv, dup_stnids]))
        
        set_bad_stations(a_stnda.ds, all_rm_stnids)
        
        print "%d total stations removed for %s:" % (all_rm_stnids.size, a_stnda.var_name)
        print "%d stations removed due to infilling issues" % (bad_stnids.size,)
        print "%d stations removed due to no TDI values" % (stnids_no_tdi.size,)
        print "%d stations removed due no climate division value, but within domain mask" % (stnids_no_climdiv.size,)
        print "%d stations removed due to being duplicates" % (dup_stnids.size,)

    # Last step of marking "bad" stations. Run a cross validation of a simple geographically
    # weighted regression model that predicts annual temperature normals. 
    # Find stations with annual normals that are extremely different than what is predicted
    # by the model (e.g. error > 6 standard deviations from the mean error).
    
    # Reload station databases to make sure bad flags are correctly set    
    stnda_tmin.ds.close() 
    stnda_tmax.ds.close()
    stnda_infill_tmin.ds.close() 
    stnda_infill_tmax.ds.close() 
    
    stnda_tmin = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmin, 'tmin', mode='r+')
    stnda_tmax = StationSerialDataDb(twx_cfg.fpath_stndata_nc_serial_tmax, 'tmax', mode='r+')

    for a_stnda in [stnda_tmin, stnda_tmax]:
        
        print "Finding outlier stations for %s..." % (a_stnda.var_name,)
        
        out_xval = XvalOutlier(a_stnda)
        stn_ids = a_stnda.stn_ids[np.isnan(a_stnda.stns[BAD])]
        
        out_stnids = out_xval.find_xval_outliers(stn_ids)
        print out_stnids
        # Mark station bad for both Tmin and Tmax
        set_bad_stations(stnda_tmin.ds, out_stnids, reset=False)
        set_bad_stations(stnda_tmax.ds, out_stnids, reset=False)
        
        print "%d stations removed due to being outliers" % (out_stnids.size,)
        