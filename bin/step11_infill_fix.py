'''
Script for performing one-off daily missing value infilling for specific
stations. The step10_mpi_infill_stn_daily.py script writes out error
messages for stations for which the PPCA infilling routine did
not appear to converge to a reasonable solution. This script can
be used to manually test different PPCA parameter combinations to try
to fix the infilled values of these stations and update the output
infilled database.  

Copyright 2014, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''

import os
from twx.db import StationDataDb, NNRNghData
from twx.utils import YMD
from twx.infill import InfillMatrixPPCA, update_daily_infill

if __name__ == '__main__':
    
    PROJECT_ROOT = "/projects/topowx"
    FPATH_STNDATA = os.path.join(PROJECT_ROOT, 'station_data')
    
    fpath_nnr_subsets = os.path.join(PROJECT_ROOT, 'reanalysis_data','conus_subset')
    path_homog_db = os.path.join(FPATH_STNDATA, 'all', 'tair_homog_1948_2012.nc') 
    path_infill_out = os.path.join(FPATH_STNDATA, 'infill')
    
    stnda = StationDataDb(path_homog_db)
    ds_nnr = NNRNghData(fpath_nnr_subsets, (stnda.days[YMD][0], stnda.days[YMD][-1]))
    
    stn_id = '' #set to a station id for which infilling should be rerun
    tair_var = 'tmin' #or tmax
    
    #PPCA parameters that can be adjusted
    min_daily_nnghs = 3
    nnghs_nnr = 4
    max_nnr_var = 0.99
    chk_perf = True
    npcs = 0
    frac_obs_initnpcs = 0.5
    ppca_varyexplain = 0.99
    ppca_con_thres = 1e-5
    verbose = True
    
    #Performing infilling
    ppca_matrix = InfillMatrixPPCA(stn_id, stnda, tair_var, ds_nnr)
    fnl_tair, mask_infill, infill_tair = ppca_matrix.infill(min_daily_nnghs, nnghs_nnr, max_nnr_var,
                                                            chk_perf, npcs, frac_obs_initnpcs, ppca_varyexplain,
                                                            ppca_con_thres, verbose)
    
    #Update the output infilled database
    update_daily_infill(stn_id, tair_var, os.path.join(path_infill_out,'infill_%s.nc'%(tair_var,)),
                        fnl_tair, mask_infill, infill_tair)
    