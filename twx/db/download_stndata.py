'''
Utility functions to download weather station data.  Current main weather
station datasources are:

1.) Global Historical Climate Network Daily (GHCN-Daily)

2.) SNOTEL Current Tab Card Files (tab delimited files)

3.) SNOTEL Historical. No download function written as this does not appear to
    be updated regularly. Downloaded manually from
    ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/snothist/.

4.) Remote Automated Weather Stations (RAWS) at the WRCC website.
'''
__all__ = ['ghcnd_download_byyr_data','ghcnd_download_data',
           'raws_build_stn_metadata','raws_save_all_dly_series',
           'raws_save_stnid_list','raws_to_ghcn_subset',
           'snotel_mirror_tabdata']
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

RPATH_SNOTEL_TABDATA = 'ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/cards/'
RPATH_GHCN = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/'
RPATH_GHCN_BYYEAR = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'

RAWS_STN_TIME_SERIES_URL = "http://www.raws.dri.edu/cgi-bin/wea_dysimts.pl?"
STN_TIME_SERIES2_URL = 'http://www.raws.dri.edu/cgi-bin/wea_dysimts2.pl'
RAWS_STN_METADATA_URL = "http://www.raws.dri.edu/cgi-bin/wea_info.pl?"


def snotel_mirror_tabdata(local_path, remote_path=RPATH_SNOTEL_TABDATA):
    '''
    Use wget to mirror SNOTEL current tab card files. This takes a while
    since there are many individual files.

    Parameters
    ----------
    local_path : str
        The local mirror path to mirror to.
    remote_path : str, optional
        The remote path to mirror from.
    '''

    subprocess.__call(['/usr/local/bin/wget', '--no-verbose', '--mirror',
                     '--directory-prefix=' + local_path, remote_path])


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

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-version.txt')])

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'status.txt')])

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'readme.txt')])

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-inventory.txt')])

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-stations.txt')])

    subprocess.__call(['wget', '--directory-prefix=' + local_path,
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

        subprocess.__call(['wget', '--directory-prefix=' + local_path,
                         urljoin(remote_path, "%s.csv.gz" % (yr, ))])


def raws_save_stnid_list(out_fpath):
    '''
    Build and save a list of RAWS station ids available from WRCC.

    Parameters
    ----------
    out_fpath : str
        The filename to which to save the station id list.
    remote_path : str, optional
        The remote path to mirror from.
    '''

    path_root = os.path.dirname(__file__)

    #raws_stnlist_pages.txt has URLs for HTML files that list RAWS stations
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
    Webcraw and download RAWS daily station from WRCC

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
