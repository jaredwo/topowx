'''
Functions for creating a time-of-observation database for GHCN-D
observations.
'''

import subprocess
import os
import datetime
from netCDF4 import Dataset, date2num
from twx.utils import status_check, get_days_metadata, DATE, YMD
import numpy as np
from create_db_all_stations import NCDF_CHK_COLS

def create_tobs_file(path_ghcn_yrly, fpath_out, yrs, element='TMAX', fips_codes=('US','CA','MX')):
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

    elem_str = "'"+element+"'"
    fips_codes_str = "'^"+"|^".join(fips_codes)+"'"
    
    for yr in yrs:
                
        cmd = "".join(["zgrep ",elem_str," ",os.path.join(path_ghcn_yrly,str(yr)),".csv.gz | zgrep -E ",
                      fips_codes_str," | zgrep '[0-9]$' | zgrep -v '2400$' >> ",fpath_out])
        print "Running cmd: "+cmd
        subprocess.call(cmd,shell=True)

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

    n_obs = int(subprocess.check_output(["wc","-l",fpath_tobs_file]).split()[0])

    stchk = status_check(n_obs, 1000000)

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
