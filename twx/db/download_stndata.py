'''
Utility functions to download weather station data. Current main weather
station datasources are:

1.) Global Historical Climate Network Daily (GHCN-Daily)
https://www.ncdc.noaa.gov/oa/climate/ghcn-daily/

2.) NRCS SNOTEL and SCAN
http://www.wcc.nrcs.usda.gov/snow/

3.) Remote Automated Weather Stations (RAWS) at the WRCC website.

Copyright 2014,2015, Jared Oyler.

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
from time import sleep

__all__ = ['ghcnd_download_byyr_data', 'ghcnd_download_data',
           'raws_build_stn_metadata', 'raws_save_all_dly_series',
           'raws_save_stnid_list', 'raws_to_ghcn_subset',
           'load_snotel_stn_inventory', 'SnotelDataService']
import os
import tarfile
import subprocess
from urlparse import urljoin
import numpy as np
import urllib
import urllib2
from datetime import datetime
from calendar import monthrange
from twx.utils import dms2decimal
import pandas as pd
from StringIO import StringIO
import pycurl

RPATH_GHCN = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/'
RPATH_GHCN_BYYEAR = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'

RAWS_STN_TIME_SERIES_URL = "http://www.raws.dri.edu/cgi-bin/wea_dysimts.pl?"
STN_TIME_SERIES2_URL = 'http://www.raws.dri.edu/cgi-bin/wea_dysimts2.pl'
RAWS_STN_METADATA_URL = "http://www.raws.dri.edu/cgi-bin/wea_info.pl?"

URL_SNOTEL = 'http://www.wcc.nrcs.usda.gov/nwcc/view'

RPATH_USHCN = 'ftp://ftp.ncdc.noaa.gov/pub/data/ushcn/v2.5/'

def load_snotel_stn_inventory(fpath_stn_inventory=None):
    '''
    Load inventory of SNOTEL/SCAN stations
    
    Parameters
    ----------
    fpath_stn_inventory : str, optional
        File path to a station inventory file downloaded
        from http://www.wcc.nrcs.usda.gov/nwcc/inventory.
        If not passed, a default pre-downloaded station inventory file
        will be used. 
    
    Returns
    -------
    stns_df : pandas.DataFrame
        Dataframe of SNOTEL/SCAN stations and associated metadata
    
    '''

    if fpath_stn_inventory is None:
    
        # if file path for station inventory not provided
        # load default air temperature station inventory file downloaded
        # from http://www.wcc.nrcs.usda.gov/nwcc/inventory 
        path_root = os.path.dirname(__file__)
        fpath_stn_inventory = os.path.join(path_root, 'data', 'snotel_scan_stns_20150123.csv')
        
    stns_df = pd.read_csv(fpath_stn_inventory)
    stns_df.columns = [a_col.strip() for a_col in stns_df.columns]
    
    # Strip whitespace from all string columns
    # and change to uppercase
    for i, a_col in enumerate(stns_df.columns):
        
        if stns_df.dtypes[i].name == 'object':
            
            stns_df[a_col] = stns_df[a_col].str.upper().str.strip()
    
    # convert elevation from feet to meters
    stns_df['elev'] = stns_df['elev'] * 0.3048  
    
    return stns_df 

class SnotelDataService():
    '''
    Class for downloading historic observations from NRCS 
    SNOTEL and SCAN station networks. See: 
    http://www.wcc.nrcs.usda.gov/report_generator/WebReportScripting.htm.
    '''
    
    def __init__(self, fpath_stn_inventory=None):
        '''
        Parameters
        ----------
        fpath_stn_inventory : str, optional
            File path to a station inventory file downloaded
            from http://www.wcc.nrcs.usda.gov/nwcc/inventory.
            If not passed, a default station inventory file
            will be used. 
        '''
                
        self.stns_df = load_snotel_stn_inventory(fpath_stn_inventory)
        
    def get_stn_obs(self, stnid, year_start_end=None, max_tries=5, nsecs_sleep_tries=15, **kwargs):
        
        '''
        Retrieve observations in CSV format for single station via 
        URL-to-file method using pycurl.
        
        Parameters
        ----------
        stnid : int
            The station/site number (2095, 302, etc). Station numbers
            can be found in the station inventory file from:
            http://www.wcc.nrcs.usda.gov/nwcc/inventory. 
        year_start_end : tuple of two ints, optional
            The start and end year for which to download
            observations. If not passed, will download station's
            entire period-of-record.
        max_tries : int, optional
            Maximum  number of NRCS connection tries when downloading data
            for each year. If max_tries tries is reached, an error will be raised.
            Default: 5.
        nsecs_sleep_tries : int, optional
            The number of seconds to sleep between connection tries if there
            is a connection error. Default: 15 seconds.
        **kwargs
           Any valid arguments for the historic URL-to-file service
           See: http://www.wcc.nrcs.usda.gov/report_generator/WebReportScripting.htm
           Some current valid arguments:
           report: str
               STAND (standard snotel), SOIL (soil temp and moisture)
               SCAN (standard scan), ALL (all), or WEATHER (atmospheric).
               Default: WEATHER
           timeseries : str
               Daily, Hourly, or Hour:HH
               Default: Daily
           month: str 
               MM, CY (calendar year), or WY (water year)
               Default: CY
           day: str
               DD
               Default: ''
        
        Returns
        -------
        yrs_obs, yrs : tuple
            yrs_obs is a list of ndarrays (dtype=str). Each separate ndarray contains
            the csv observation lines for a specific year. yrs is an ndarray with the
            corresponding years.
        '''

        # SNOTEL URL query string parameters  
        url_args = {'intervalType':'Historic',
                   'report':'WEATHER',
                   'timeseries':'Daily',
                   'format':'copy',
                   'sitenum':str(stnid),
                   'year':'',
                   'month':'CY',
                   'day':''}
        # update url args with any custom settings passed in kwargs
        url_args.update(kwargs)
        
        if year_start_end is None:
            
            # year start end not provided
            # download entire period-of-record of station
            a_stn = self.stns_df[self.stns_df['station id'] == stnid].iloc[0]
            year_start = int(a_stn['start_date'][0:4])
            year_end = int(a_stn['end_date'][0:4])
            
            # If no end year, use current year
            if year_end == 2100:
                
                year_end = datetime.now().year
            
            year_start_end = (year_start, year_end)
                    
        yrs = np.arange(year_start_end[0], year_start_end[1] + 1)
        yrs_obs = []
        yrs_success = np.ones(yrs.size, dtype=np.bool)
        
        for i, yr in enumerate(yrs):
            
            ntries = 0
            
            buf = StringIO()
            c = pycurl.Curl()
            c.setopt(c.WRITEDATA, buf)
            
            url_args['year'] = yr
            url = "?".join([URL_SNOTEL, urllib.urlencode(url_args)])  
            c.setopt(c.URL, url)
            
            while 1:
        
                try:
                    
                    print url  
                    c.perform()
                    c.close()
                    break
        
                except pycurl.error as e:
                
                    print "Error downloading station obs for station %d for year %d: %s" % (stnid, yr, str(e))
        
                    ntries += 1
                    
                    if ntries == max_tries:
                    
                        print "Max tries reached for station %d for year %d. Raising error..." % (stnid, yr)
                        raise
                    
                    else:
                    
                        print "Sleeping %d secs and trying year again..." % nsecs_sleep_tries
                        sleep(nsecs_sleep_tries)
            
            try:
                # Change string buffer to array of lines
                lines = np.array(buf.getvalue().splitlines(True))
                
                # Get rid of extra end-of-water year lines
                mask = np.char.find(lines, '-09-30,23:59') == -1
                             
                lines = np.extract(mask, lines)
                
            except Exception as e:
                
                print "Error: could not parse obs for station %d for year %d: %s" % (stnid, yr, str(e))
                yrs_success[i] = False
            
            if yrs_success[i]:
                yrs_obs.append(lines)
        
        
        yrs = yrs[yrs_success]
        
        return yrs_obs, yrs
    
    def write_stn_obs(self, stnid, path_out, year_start_end=None, output_prefix_field='cdbs_id', max_tries=5, nsecs_sleep_tries=15, **kwargs):
        '''
        Retrieve and output SNOTEL data to CSV files. A station directory
        will be created in path_out and a separate CSV file written for each year. 
        CSV filename: [output_prefix_field]_[year].csv
        
        Parameters
        ----------
        stnid : int
            The station/site number (2095, 302, etc). Station numbers
            can be found in the station inventory file from:
            http://www.wcc.nrcs.usda.gov/nwcc/inventory.
        path_out : str
            The output path for the CSV file  
        year_start_end : tuple of two ints, optional
            The start and end year for which to download
            observations. If not passed, will download station's
            entire period-of-record.
        output_prefix_field : str
            The field from the station inventory file that will form
            the prefix of the output filename. Default: the station's cdbs_id.
        max_tries : int, optional
            Maximum  number of NRCS connection tries when downloading data
            for each year. If max_tries tries is reached, an error will be raised.
            Default: 5.
        nsecs_sleep_tries : int, optional
            The number of seconds to sleep between connection tries if there
            is an connection error. Default: 15 seconds.
        **kwargs
           Any valid arguments for the historic URL-to-file service
           See: http://www.wcc.nrcs.usda.gov/report_generator/WebReportScripting.htm
           Some current valid arguments:
           report: str
               STAND (standard snotel), SOIL (soil temp and moisture)
               SCAN (standard scan), ALL (all), or WEATHER (atmospheric).
               Default: WEATHER
           timeseries : str
               Daily, Hourly, or Hour:HH
               Default: Daily
           month: str 
               MM, CY (calendar year), or WY (water year)
               Default: CY
           day: str
               DD
               Default: ''
        '''
        
        yrs_obs, yrs = self.get_stn_obs(stnid, year_start_end, **kwargs)
        
        a_stn = self.stns_df[self.stns_df['station id'] == stnid].iloc[0]
        
        fname_prefix = str(a_stn[output_prefix_field]).strip()
        
        if fname_prefix == '':
            print "Warning: %s field was blank for stnid %d. Will prefix output with stnid instead" % (output_prefix_field, stnid)
            fname_prefix = str(stnid)
            
        stn_dir_out = os.path.join(path_out, fname_prefix)
        
        if not os.path.isdir(stn_dir_out):
            os.mkdir(stn_dir_out)
        
        for obs_lines, yr in zip(yrs_obs, yrs):
        
            fname = "%s_%d.csv" % (fname_prefix, yr)
        
            print "Writing output file %s..." % fname
        
            fout = open(os.path.join(stn_dir_out, fname), 'w')
            fout.writelines(obs_lines)
            fout.close()

def ghcnd_download_data(local_path, remote_path=RPATH_GHCN, extract_tar=True):
    '''
    Use wget to download and unzip the latest GHCN-Daily data

    Parameters
    ----------
    local_path : str
        The local path to download to.
    remote_path : str, optional
        The remote path to download from.
    extract_tar : bool, optional
        Extract the ghcnd_all.tar.gz file
    '''

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-version.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'status.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'readme.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-inventory.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-stations.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd_all.tar.gz')])

    if extract_tar:
        print "Extracting GHCN-Daily tar.gz file..."
        ghcn_tar = tarfile.open(os.path.join(local_path, 'ghcnd_all.tar.gz'))
        ghcn_tar.extractall(local_path)


def ghcnd_download_byyr_data(local_path, yrs, remote_path=RPATH_GHCN_BYYEAR):
    '''
    Use wget to download the latest GHCN-Daily data in "by year" format

    Parameters
    ----------
    local_path : str
        The local path to download to.
    yrs: sequence of int
        A list of years for which to download
    remote_path : str, optional
        The remote path to download from.
    '''

    for yr in yrs:

        subprocess.call(['wget', '--directory-prefix=' + local_path,
                         urljoin(remote_path, "%s.csv.gz" % (yr,))])
        

def ushcn_download_data(local_path, remote_path=RPATH_USHCN, extract_tar=True):
    '''
    Use wget to mirror and unzip the latest USHCN data

    Parameters
    ----------
    local_path : str
        The local path to download to.
    remote_path : str, optional
        The remote path to download from.
    extract_tar : bool, optional
        Extract all *.tar.gz files downloaded
        to local_path
    '''
    
    subprocess.call(['wget','--mirror',
                     '--directory-prefix=' + local_path,
                     '-nd',remote_path])
    
    if extract_tar:
        
        fnames = np.array(os.listdir(local_path))
        fnames = fnames[np.char.endswith(fnames,'tar.gz')]
        
        for a_name in fnames:
            
            print "Extracting "+a_name+"..."
            a_tar = tarfile.open(os.path.join(local_path, a_name))
            a_tar.extractall(local_path)
    

def raws_save_stnid_list(out_fpath):
    '''
    Build and save a list of RAWS station ids available from WRCC.

    Parameters
    ----------
    out_fpath : str
        The filename to which to save the station id list.
    '''

    path_root = os.path.dirname(__file__)

    # raws_stnlist_pages.txt has URLs for HTML files that list RAWS stations
    afile = open(os.path.join(path_root, 'data', 'raws_stnlst_pages.txt'))
    stn_ids = []

    for line in afile.readlines():

        req = urllib2.Request(line.strip())
        response = urllib2.urlopen(req)
        plines = response.readlines()
        for pline in plines:

            if "rawMAIN.pl" in pline:

                stn_id = pline.split("?")[1][0:6]
                print stn_id
                stn_ids.append(stn_id)

    stn_ids = np.unique(stn_ids)
    print "Total # of stn_ids: " + str(stn_ids.size)

    fo = open(out_fpath, "w")
    for stn_id in stn_ids:
        fo.write("".join([stn_id, "\n"]))


def raws_to_ghcn_subset(fpath_raws_ids, fpath_ghcnd_stns, fpath_out):
    '''
    Subset raws stations to only those that are in GHCN-D

    Parameters
    ----------
    fpath_raws_ids : str
        The filename of a RAWS station id file produced by 'raws_save_stnid_list'.
    fpath_ghcnd_stns : str
        The filename of the ghcnd-stations.txt file provided in the GHCN-D download.
    fpath_out : str
        The filename to which to save the subset station id list
    '''

    raws_ids_orig = np.loadtxt(fpath_raws_ids, dtype="<S6")
    raws_ids = np.array([x[2:] for x in raws_ids_orig], dtype="<S4")

    ghcn_stns = open(fpath_ghcnd_stns)

    ghcn_raws = []
    for aline in ghcn_stns.readlines():

        stn_id = aline[0:11].strip()

        # prefix for a raws station
        if stn_id[0:3] == "USR":
            ghcn_raws.append(stn_id[-4:])

    ghcn_raws = np.array(ghcn_raws, dtype="<S4")

    fnl_ids = raws_ids_orig[np.in1d(raws_ids, ghcn_raws, True)]
    np.savetxt(fpath_out, fnl_ids, "%s")


def raws_build_stn_metadata(fpath_stnids, fpath_out):
    '''
    Build a RAWS station metadata file

    Parameters
    ----------
    fpath_raws_ids : str
        The filename of a RAWS station id file.
    fpath_out : str
        The filename to which to save metadata
    '''

    afile = open(fpath_stnids)
    fo = open(fpath_out, "w")

    for stn_id in afile.readlines():

        try:
            stn_id = stn_id.strip()

            response = urllib2.urlopen("".join([RAWS_STN_TIME_SERIES_URL, stn_id]))
            plines = response.readlines()

            start_date = None
            end_date = None
            stn_name = None
            read_stn_name = False

            for pline in plines:

                if start_date is None or end_date is None or stn_name is None:

                    if read_stn_name:
                        stn_name = pline.split(">")[1][0:-4]
                        read_stn_name = False

                    if "Station:" in pline:
                        read_stn_name = True

                    if "Earliest available data" in pline:
                        start_date = datetime.strptime(pline.split(":")[1][1:-2],
                                                       "%B %Y")

                    if "Latest available data" in pline:
                        end_date = datetime.strptime(pline.split(":")[1][1:-2],
                                                     "%B %Y")
                        # set to last day of month
                        end_date = end_date.replace(day=monthrange(end_date.year,
                                                                   end_date.month)[1])

            response = urllib2.urlopen("".join([RAWS_STN_METADATA_URL, stn_id]))
            plines = response.readlines()
            lon, lat, elev = [None] * 3

            for x in np.arange(len(plines)):

                pline = plines[x]

                if "Latitude" in pline:
                    lat = _parse_decdegrees(plines[x + 2])

                if "Longitude" in pline:
                    lon = _parse_decdegrees(plines[x + 2])

                if "Elevation" in pline:
                    elev = float(plines[x + 2].split()[0]) * 0.3048  # convert from feet to meters

            stn_ln = " ".join([stn_id[2:], start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"),
                               "%.5f" % lat, "%.5f" % lon, "%.2f" % elev, stn_name, "\n"])

            print stn_ln
            fo.write(stn_ln)
        except:
            print "COULD NOT LOAD METADATA FOR: ", stn_id
    fo.close()


def raws_save_all_dly_series(fpath_meta, path_out):
    '''
    Webcrawl and download RAWS daily station from WRCC

    Parameters
    ----------
    fpath_meta : str
        The filename of RAWS station metadata from 'raws_build_stn_metadata'.
    path_out : str
        The directory to which to save the data files
    '''

    fmeta = open(fpath_meta)

    for line in fmeta.readlines():

        stn_id = line[0:4]
        start_date = datetime.strptime(line[5:15], "%Y-%m-%d")
        end_date = datetime.strptime(line[16:26], "%Y-%m-%d")

        if not os.path.exists(os.path.join(path_out, "%s.txt" % (stn_id,))):

            print stn_id, start_date, end_date
            _save_dly_series(stn_id, start_date, end_date, path_out)


def _save_dly_series(stn_id, start_date, end_date, out_path):

    values = {'smon': start_date.strftime("%m"),
              'sday': start_date.strftime("%d"),
              'syea': start_date.strftime("%y"),
              'emon': end_date.strftime("%m"),
              'eday': end_date.strftime("%d"),
              'eyea': end_date.strftime("%y"),
              'qBasic': "ON",
              'unit': 'M',
              'Ofor': 'A',
              'Datareq': 'A',
              'qc': 'Y',
              'miss': '08',  # -9999
              "Submit Info": "Submit Info",
              'stn': stn_id,
              'WsMon': '01',
              'WsDay': '01',
              'WeMon': '12',
              'WeDay': '31'}

    data = urllib.urlencode(values)
    req = urllib2.Request(STN_TIME_SERIES2_URL, data)
    response = urllib2.urlopen(req)

    fo = open(os.path.join(out_path, "%s.txt" % (stn_id,)), "w")

    for pline in response.readlines():
        fo.write(pline)
    fo.close()


def _parse_decdegrees(a_str):

    vals = a_str.split("&")

    deg = float(vals[0])

    vals = vals[1].split()

    minute = float(vals[1][0:-1])
    sec = float(vals[2])

    return dms2decimal(deg, minute, sec)
