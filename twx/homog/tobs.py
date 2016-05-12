'''
Functions for creating a time-of-observation database for GHCN-D
observations.

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

__all__ = ['create_tobs_db', 'create_tobs_file', 'InsertTobs']

import subprocess
import os
import datetime
from netCDF4 import Dataset, date2num
from twx.utils import StatusCheck, get_days_metadata, YEAR, MONTH, YMD, DAY, DATE
import numpy as np
import twx
from twx.db import STN_ID, LON, LAT, STN_NAME, STATE, ELEV, MISSING, DTYPE_STNOBS, NCDF_CHK_COLS


def create_tobs_file(path_ghcn_yrly, fpath_out, yrs, element='TMAX', fips_codes=('US', 'CA', 'MX')):
    '''
    Create a time-of-observation (tobs) file that has the tobs 
    for all non-calendar day GHCN-D observations.
    
    Parameters
    ----------
    path_ghcn_yrly : str
        The local path to the GHCN-Daily "by year" format data.
    fpath_out : str
        The file path to which to write the tobs data.
    yrs : sequence of ints
        The years for which to write tobs data.
    element : str, optional
        The observation element (e.g. TMIN, TMAX, PRCP) for
        which to write tobs data.
    fips_codes | sequence of str, optional
        The country fips codes (e.g. US, CA, MX) for which
        to write tobs data.
    '''

    elem_str = "'" + element + "'"
    fips_codes_str = "'^" + "|^".join(fips_codes) + "'"

    for yr in yrs:

        cmd = "".join(["zgrep ", elem_str, " ", os.path.join(path_ghcn_yrly, str(yr)), ".csv.gz | zgrep -E ",
                      fips_codes_str, " | zgrep '[0-9]$' | zgrep -v '2400$' >> ", fpath_out])
        print "Running cmd: " + cmd
        subprocess.call(cmd, shell=True)

def create_tobs_db(fpath_tobs_file, fpath_db, stnids, min_date, max_date):
    '''
    Create a time-of-observation (tobs) netCDF4 database from a
    tobs file generated from create_tobs_file.
    
    Parameters
    ----------
    fpath_tobs_file : str
        The file path to the tobs file from create_tobs_file
    fpath_db : str
        The file path to which to write the tobs database.
    stnids : sequence of str
        The sorted station ids of stns whose tobs should be written
        to the database
    min_date : datetime
        The earliest observation date
    max_date : datetime
        The latest observation date
    '''

    ds = Dataset(fpath_db, 'w')

    # Set global attributes
    ds.title = "Time-of-Observation Database"
    ds.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ds.history = "".join(["Created on: ", datetime.datetime.strftime(datetime.date.today(), "%Y-%m-%d")])

    days = get_days_metadata(min_date, max_date)

    print "Creating netCDF4 Time-of-Observation Database for " + str(min_date) + \
    " to " + str(max_date) + " for " + str(stnids.size) + " stations."

    ds.createDimension('time', days.size)
    ds.createDimension('stn_id', stnids.size)

    times = ds.createVariable('time', 'f8', ('time',), fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ", str(min_date.year), "-",
                            str(min_date.month), "-", str(min_date.day), " 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE], times.units)

    stations = ds.createVariable('stn_id', 'str', ('stn_id',))
    stations.long_name = "station id"

    for x in np.arange(stnids.size):

        ds.variables['stn_id'][x] = str(stnids[x])

    tobs = ds.createVariable('tobs', np.int16, ('time', 'stn_id'),
                              fill_value=-1, chunksizes=(days[DATE].size, NCDF_CHK_COLS))
    tobs.long_name = "time-of-observation"
    tobs.missing_value = -1

    stnidsOrig = np.char.replace(stnids, "GHCN_", "", 1)

    fileTobs = open(fpath_tobs_file)
    aline = fileTobs.readline()

    atobs = np.ones((days.size, stnids.size)) * -1

    curYmd = days[YMD][0]
    time_idx = 0
    stn_idx = 0

    n_obs = int(subprocess.check_output(["wc", "-l", fpath_tobs_file]).split()[0])

    stchk = StatusCheck(n_obs, 1000000)

    stn_idxs = {}
    for x in np.arange(stnidsOrig.size):
        stn_idxs[stnidsOrig[x]] = x

    while aline != "":

        try:

            stn_idx = stn_idxs[aline[0:11]]

            aYmd = np.int(aline[12:20])
            if aYmd != curYmd:
                time_idx = np.nonzero(days[YMD] == aYmd)[0][0]
                curYmd = aYmd

            atobs[time_idx, stn_idx] = np.int(aline[-5:])
#
        except KeyError:
            pass
        stchk.increment()
        aline = fileTobs.readline()
    tobs[:] = atobs

class InsertTobs(twx.db.Insert):
    '''
    Class for inserting stations and observations that have been modified for
    time-of-observation. Loads stations and observations from current netCDF
    station dataset and performs time-of-observation adjustments.
    '''
    
    def __init__(self,stnda, stns_tmin, stns_tmax):
        '''
        Parameters
        ----------
        stnda : twx.db.StationDataDb
            A StationDataDb object pointing to a netCDF station dataset
            from which stations and observations should be loaded
        stns_tmin : structured ndarray
            Stations for which Tmin observations should be inserted. 
            Structured station array must contain at least the following fields:
            STN_ID, LAT, LON, ELEV, STATE, STN_NAME and can be obtained from
            twx.db.StationDataDb.
        stns_tmax : structured ndarray
            Stations for which Tmax observations should be inserted. 
            Structured station array must contain at least the following fields:
            STN_ID, LAT, LON, ELEV, STATE, STN_NAME and can be obtained from
            twx.db.StationDataDb.
        '''
        
        twx.db.Insert.__init__(self,stnda.days[DATE][0], stnda.days[DATE][-1])
        
        self.stnda = stnda
        self.stns_tmin = stns_tmin
        self.stns_tmax = stns_tmax
        self.stns_all = np.concatenate((self.stns_tmin,
                                        self.stns_tmax[~np.in1d(self.stns_tmax[STN_ID],
                                                                self.stns_tmin[STN_ID],
                                                                assume_unique=True)]))
        self.stns_all = self.stns_all[np.argsort(self.stns_all[STN_ID])]
                
        self.stn_list = [(stn[STN_ID], stn[LAT], stn[LON], stn[ELEV], '',
                          stn[STN_NAME]) for stn in self.stns_all]
                
        #self.empty_obs_tmin = np.ones(stnda.days.size)*self.stnda.ds['tmin'].missing_value
        #self.empty_obs_tmax = np.ones(stnda.days.size)*self.stnda.ds['tmax'].missing_value
        self.empty_obs = np.ones(stnda.days.size)*MISSING
        self.empty_qa = np.zeros(stnda.days.size,dtype=np.str)
    
    def get_stns(self):
        
        return self.stn_list
                            
    def parse_stn_obs(self,stn_id):
        
        if stn_id in self.stns_tmin[STN_ID]:
            tmin = self.stnda.load_all_stn_obs_var(stn_id,'tmin')[0]
            tmin[np.isnan(tmin)] = MISSING
        else:
            tmin = self.empty_obs
        
        if stn_id in self.stns_tmax[STN_ID]:

            tmax = self.stnda.load_all_stn_obs_var(stn_id, 'tmax')[0]
            tobs = self.stnda.load_all_stn_obs_var(stn_id, 'tobs_tmax')[0]
            #tobs = self.ds_tobs.variables['tobs'][:,np.nonzero(self.ds_tobs_stnids==stn_id)[0][0]]
            tmax = _tobs_shift_tmax(tmax, tobs)
            tmax[np.isnan(tmax)] = MISSING
            
        else:
            
            tmax = self.empty_obs
                                
        obs = np.empty(self.stnda.days.size, dtype=DTYPE_STNOBS)                
        obs['year'] = self.stnda.days[YEAR]
        obs['month'] = self.stnda.days[MONTH]
        obs['day'] = self.stnda.days[DAY]
        obs['ymd'] = self.stnda.days[YMD]
        obs['tmin'] = tmin
        obs['tmax'] = tmax
        obs['prcp'] = self.empty_obs
        obs['swe'] = self.empty_obs
        obs['qflag_tmin'] = self.empty_qa
        obs['qflag_tmax'] = self.empty_qa
        obs['qflag_prcp'] = self.empty_qa
                
        return obs
    
def _tobs_shift_tmax(tmax, tobs):
    '''
    Shift morning observations of Tmax back a calendar day
    '''    
    #any tobs before 1100 is considered morning
    tobs_mask_am = np.logical_and(tobs > 0, tobs < 1100)
            
    if np.sum(tobs_mask_am) > 0:
                        
        tobs_mask_ok = np.logical_and(~tobs_mask_am, np.isfinite(tmax))
        idx_shift = np.nonzero(tobs_mask_am)[0]
        idx_shift = idx_shift[idx_shift > 0]
        idx_shift = idx_shift[~tobs_mask_ok[idx_shift-1]]
        
        if idx_shift.size > 1:
        
            tmax_shift = np.ones(tmax.size)*np.nan
            tmax_shift[tobs_mask_ok] = tmax[tobs_mask_ok]
            tmax_shift[idx_shift-1] = tmax[idx_shift]
            tmax = tmax_shift
    
    return tmax

