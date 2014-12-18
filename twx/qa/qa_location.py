'''
Functions for quality assurance of station location metadata (longitude, latitude, elevation). 
In most cases, elevation is correct, but longitude and/or latitude are either imprecise or incorrect.
The functions can be used to compare the provided elevation of a station to that of a high resolution
DEM. Those stations that have an elevation that significantly differs from the corresponding DEM elevation
likely have issues with their metadata. 

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

__all__ = ["combine_locqa","qa_stn_locs","set_usrname_geonames","update_stn_locs"]

import urllib, urllib2
from xml.dom import minidom
from twx.db.station_data import STN_ID, STATE, LON, LAT, ELEV
import numpy as np
import time
from netCDF4 import Dataset

# DEM service URLs
URL_USGS_NED = 'http://gisdata.usgs.gov/XMLWebServices/TNM_Elevation_Service.asmx/getElevation'
URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'

def _load_locqa_lines(path_locqa):
    '''
    Load a location quality assurance csv file. 
    A location quality assurance csv file has the following headers: 
    STN_ID: The station id of the station
    ST: That state of the state
    LON: The original longitude of the station
    LAT: The original latitude of the station
    ELEV: The elevation provided for the station from the original data source (meters)
    DEM: The high-resolution DEM elevation for the station's original lat, lon (meters)
    DIF: The difference between the DEM elevation and the original listed elevation (meters)
    LON_NEW: A new, fixed longitude for the station (if applicable, manually added by the user)
    LAT_NEW: A new, fixed latitude for the station (if applicable, manually added by the user)
    ELEV_NEW: A new, fixed elevation for the station (if applicable, manually added by the user)
     
    Parameters
    ----------
    path_locqa : str
        The file path to the location quality assurance csv file
            
    Returns
    -------
    locs : dict
        A dict of line strings. The dict keys are station ids.
    '''

    afile = open(path_locqa)
    afile.readline()

    locs = {}
    for line in afile.readlines():

        # STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        stn_id = line.split(",")[0].strip()
        locs[stn_id] = line

    return locs

def _load_locs_fixed_all(path_locqa):
    '''
    Load a location quality assurance csv file. A location quality assurance csv file
    has the following headers: 
    STN_ID: The station id of the station
    ST: That state of the state
    LON: The original longitude of the station
    LAT: The original latitude of the station
    ELEV: The elevation provided for the station from the original data source (meters)
    DEM: The high-resolution DEM elevation for the station's original lat, lon (meters)
    DIF: The difference between the DEM elevation and the original listed elevation (meters)
    LON_NEW: A new, fixed longitude for the station (if applicable, manually added by the user)
    LAT_NEW: A new, fixed latitude for the station (if applicable, manually added by the user)
    ELEV_NEW: A new, fixed elevation for the station (if applicable, manually added by the user)
     
    Parameters
    ----------
    path_locqa : str
        The file path to the location quality assurance csv file.
            
    Returns
    -------
    locs : dict
        A dict of tuples containing longitude, latitude, and elevation values for all
        stations in the location quality assurance csv file. If a station has values
        for the LON_NEW, LAT_NEW fields, the LON_NEW, LAT_NEW, and ELEV_NEW fields are
        returned instead of the original. The dict keys are station ids
    '''

    afile = open(path_locqa)
    afile.readline()

    locs = {}
    for line in afile.readlines():

        # STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        vals = line.split(",")
        vals = [x.strip() for x in vals]

        # If fixed/new values are not empty, use these instead of originals
        if vals[7] != '' and vals[8] != '':

            locs[vals[0]] = [float(vals[7]), float(vals[8]), float(vals[9])]

        else:

            locs[vals[0]] = [float(vals[2]), float(vals[3]), float(vals[4])]

    return locs

def _load_locs_fixed(path_locqa):
    '''
    Load only manually fixed station locations in a location quality assurance csv file. 
    A location quality assurance csv file has the following headers: 
    STN_ID: The station id of the station
    ST: That state of the state
    LON: The original longitude of the station
    LAT: The original latitude of the station
    ELEV: The elevation provided for the station from the original data source (meters)
    DEM: The high-resolution DEM elevation for the station's original lat, lon (meters)
    DIF: The difference between the DEM elevation and the original listed elevation (meters)
    LON_NEW: A new, fixed longitude for the station (if applicable, manually added by the user)
    LAT_NEW: A new, fixed latitude for the station (if applicable, manually added by the user)
    ELEV_NEW: A new, fixed elevation for the station (if applicable, manually added by the user)
     
    Parameters
    ----------
    path_locqa : str
        The file path to the location quality assurance csv file
            
    Returns
    -------
    locs : dict
        A dict of tuples containing LON_NEW,LAT_NEW,ELEV_NEW values for only stations
        that had their locations manually fixed. The dict keys are station ids
    '''

    afile = open(path_locqa)
    afile.readline()

    locs = {}
    for line in afile.readlines():

        # STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        vals = line.split(",")

        # If fixed/new values are not empty, load station
        if vals[7] != '' and vals[8] != '':

            locs[vals[0]] = [float(vals[7]), float(vals[8]), float(vals[9])]

    return locs

def combine_locqa(old_locqa, new_locqa, path_fout):
    '''
    Combines two location quality assurance csv files into a single
    location quality assurance csv. If the 2 files have overlapping stations,
    the new file takes precedence over the old file.
    
    Parameters
    ----------
    old_locqa : str
        A file path to a location quality assurance csv file
    new_locqa : str
        A file path to a location quality assurance csv file
    path_fout: str
        The file path for the combined location quality assurance csv file
    '''

    locs_old = _load_locqa_lines(old_locqa)
    locs_new = _load_locqa_lines(new_locqa)

    stnids_old = np.array(locs_old.keys())
    stnids_new = np.array(locs_new.keys())

    stnids_old = stnids_old[np.logical_not(np.in1d(stnids_old, stnids_new, assume_unique=True))]

    fout = open(path_fout, "w")
    fout.write(",".join(["STN_ID", "ST", "LON", "LAT", "ELEV", "DEM", "DIF", "LON_NEW", "LAT_NEW", "ELEV_NEW", "\n"]))

    for stn_id in stnids_old:

        fout.write(locs_old[stn_id])

    for stn_id in stnids_new:

        fout.write(locs_new[stn_id])

    fout.close()


def _get_elev_geonames(lon, lat, usrname=None, url=URL_GEONAMES_SRTM):
    '''
    Get elev value from geonames web sevice (SRTM or ASTER). If usrname is None,
    the existence of global variable USRNAME_GEONAMES will be checked to try to
    retrieve a geonames username. USRNAME_GEONAMES can be set with set_usrname_geonames
    '''

    global USRNAME_GEONAMES

    if usrname is None:

        try:

            usrname = USRNAME_GEONAMES

        except NameError:

            raise Exception("_get_elev_geonames: usrname is None and  USRNAME_GEONAMES variable doesn't exist!")

    while 1:
        # ?lat=50.01&lng=10.2&username=demo
        # url GET args
        values = {'lat' : lat,
        'lng' : lon,
        'username' : usrname}

        # encode the GET arguments
        data = urllib.urlencode(values)

        # make the URL into a qualified GET statement
        get_url = "".join([url, "?", data])

        req = urllib2.Request(url=get_url)
        response = urllib2.urlopen(req)
        elev = float(response.read().strip())

        if elev == -32768.0 and url == URL_GEONAMES_SRTM:
            # Try ASTER instead
            url = URL_GEONAMES_ASTER
        else:
            break

    # print "".join(["Geonames Elev: ",str(elev)])
    return elev

def _get_elev(stn):
    '''
    Get the elevation of a station's lon/lat from a DEM online datasource
    
    Parameters
    ----------
    stn : structured ndarray
        A station record from a structured ndarray containing at least the
        following fields: STN_ID,STATE,LON,LAT,ELEV
            
    Returns
    -------
    elev_dem : float
        The DEM elevation corresponding to the station's lon/lat 
    '''

    # determine if a us station or not. currently, only stations outside us are in mexico or canada
    us_stn = not (stn[STN_ID].find('CA') != -1 or stn[STN_ID].find('MX') != -1 or
                  (stn[STATE] == '' or stn[STATE] == 'AK' or stn[STATE] == 'HI'))

    while 1:

        try:
            # If the station is within the US, use the USGS data service
            # If not in the US, use the geonames data service
            if us_stn:
                elev_dem = _get_elev_usgs(stn[LON], stn[LAT])
            else:
                elev_dem = _get_elev_geonames(stn[LON], stn[LAT])

            return elev_dem

        except urllib2.URLError:
            print "Error in connection. sleep and try again..."
            time.sleep(5)


def _get_elev_usgs(lon, lat):
    '''
    Get elev value from USGS NED 1/3 arc-sec DEM.  Code directly modeled from:
    http://casoilresource.lawr.ucdavis.edu/drupal/node/610
    '''

    #   NED 1/3rd arc-second: Eastern United States    NED.CONUS_NED_13E    -99.0006,24.9994,
    # -65.9994,49.0006
    # NED 1/3rd arc-second: Western United States    NED.CONUS_NED_13W    -125.0006,25.9994,
    # -98.9994,49.0006

    if lon <= -98.9994:
        src_layer = 'NED.CONUS_NED_13W'
    else:
        src_layer = 'NED.CONUS_NED_13E'


    # url GET args
    values = {'X_Value' : lon,
    'Y_Value' : lat,
    'Elevation_Units' : 'meters',
    'Source_Layer' : src_layer,
    'Elevation_Only' : '1', }

    # make some fake headers, with a user-agent that will
    # not be rejected by bone-headed servers
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent' : user_agent}

    # encode the GET arguments
    data = urllib.urlencode(values)

    # make the URL into a qualified GET statement:
    get_url = URL_USGS_NED + '?' + data

    # make the request: note that by ommitting the url arguments
    # we force a GET request, instead of a POST
    req = urllib2.Request(url=get_url, headers=headers)
    response = urllib2.urlopen(req)
    the_page = response.read()

    try:

        # convert the HTML back into plain XML
        for entity, char in (('lt', '<'), ('gt', '>'), ('amp', '&')):
            the_page = the_page.replace('&%s;' % entity, char)

        # clean some cruft... XML won't parse with this stuff in there...
        the_page = the_page.replace('<string xmlns="http://gisdata.usgs.gov/XMLWebServices/">', '')
        the_page = the_page.replace('<?xml version="1.0" encoding="utf-8"?>\r\n', '')
        the_page = the_page.replace('</string>', '')
        the_page = the_page.replace('<!-- Elevation Values of -1.79769313486231E+308 (Negative Exponential Value) may mean the data source does not have values at that point.  --> <USGS_Elevation_Web_Service_Query>', '')

        # parse the cleaned XML
        dom = minidom.parseString(the_page)
        children = dom.getElementsByTagName('Elevation_Query')[0]

        # extract the interesting parts
        elev = float(children.getElementsByTagName('Elevation')[0].firstChild.data)
        data_source = children.getElementsByTagName('Data_Source')[0].firstChild.data

        # print to stdout
        # print "%f,%f,%f,%s" % (lon, lat, elev, data_source)

        return elev

    except:

        print "".join(["ERROR: ", str(lon), ",", str(lat)])
        return np.nan

def update_stn_locs(path_db, path_locqa):
    '''
    Update the longitude, latitude, and elevation of stations in a netCDF database 
    based on corrected locations in a location quality assurance csv file.
    A location quality assurance csv file has the following headers: 
    STN_ID: The station id of the station
    ST: That state of the state
    LON: The original longitude of the station
    LAT: The original latitude of the station
    ELEV: The elevation provided for the station from the original data source (meters)
    DEM: The high-resolution DEM elevation for the station's original lat, lon (meters)
    DIF: The difference between the DEM elevation and the original listed elevation (meters)
    LON_NEW: A new, fixed longitude for the station (if applicable, manually added by the user)
    LAT_NEW: A new, fixed latitude for the station (if applicable, manually added by the user)
    ELEV_NEW: A new, fixed elevation for the station (if applicable, manually added by the user)
     
    Parameters
    ----------
    path_db : str
        The file path to the netCDF4 database
    path_locqa : str
        The file path to the location quality assurance csv file
    '''

    locs_fixed = _load_locs_fixed(path_locqa)

    ds = Dataset(path_db, 'r+')
    var_lon = ds.variables['lon']
    var_lat = ds.variables['lat']
    var_elev = ds.variables['elev']
    stn_ids = ds.variables['stn_id'][:]

    cnt = 0
    print "Total # of station locs to fix: " + str(len(locs_fixed.keys()))
    for stn_id in locs_fixed.keys():

        try:
            x = np.nonzero(stn_ids == stn_id)[0][0]
        except IndexError:
            print "Could not update, since station not in database: " + stn_id
            continue

        lon, lat, elev = locs_fixed[stn_id]
        var_lon[x] = lon
        var_lat[x] = lat
        var_elev[x] = elev
        cnt += 1
        ds.sync()
    print "Total # of stations locs updated: " + str(cnt)

def qa_stn_locs(stns, path_out, path_locqa_prev=None):
    '''
    Perform quality assurance of station locations by retrieving the DEM elevation for 
    a station's lon/lat from a DEM online datasource and then comparing the DEM elevation
    with that listed for the station. The results are written to an output location 
    quality assurance csv file. Inspecting the output file, the user must decide if 
    there is a location issue that warrants manually relocating a station and then update the
    LON_NEW, LAT_NEW, and ELEV_NEW of the output file and run update_stn_locs. The output
    file headers are:
    STN_ID: The station id of the station
    ST: That state of the state
    LON: The original longitude of the station
    LAT: The original latitude of the station
    ELEV: The elevation provided for the station from the original data source (meters)
    DEM: The high-resolution DEM elevation for the station's original lat, lon (meters)
    DIF: The difference between the DEM elevation and the original listed elevation (meters)
    LON_NEW: A new, fixed longitude for the station (if applicable, manually added by the user)
    LAT_NEW: A new, fixed latitude for the station (if applicable, manually added by the user)
    ELEV_NEW: A new, fixed elevation for the station (if applicable, manually added by the user)
     
    Parameters
    ----------
    stns : structured ndarray
        A structured station array containing the longitude, latitude, and elevation of
        the stations to be checked. A structured station array is an attribute
        of twx.db.StationDataDb and can be subsetted.
    path_out : str
        The file path for the output location quality assurance csv file
    path_locqa_prev : str, optional
        A file path to a previous location quality assurance csv file. If a station location 
        (lon, lat, elev) of the provided stns structured ndarray matches that in the
        previous location quality assurance csv file, the station location will not be
        QA'd and written to the output file  
    '''

    if path_locqa_prev is not None:

        locs_fixed = _load_locs_fixed_all(path_locqa_prev)

    else:

        locs_fixed = {}

    print "Total num of stations for location qa: " + str(stns.size)

    fout = open(path_out, "w")
    fout.write(",".join(["STN_ID", "ST", "LON", "LAT", "ELEV", "DEM", "DIF", "LON_NEW", "LAT_NEW", "ELEV_NEW", "\n"]))

    for stn in stns:

        chk_stn = False

        if locs_fixed.has_key(stn[STN_ID]):

            lon, lat, elev = locs_fixed[stn[STN_ID]]
            if np.round(stn[LON], 4) != np.round(lon, 4) or np.round(stn[LAT], 4) != np.round(lat, 4) or np.round(stn[ELEV], 0) != np.round(elev, 0):
                chk_stn = True

        else:
            chk_stn = True

        if chk_stn:

            elev_dem = _get_elev(stn)
            dif = stn[ELEV] - elev_dem
            print "Station %s elevation difference from DEM: %.2f meters." % (stn[STN_ID], dif)
            fout.write(",".join([stn[STN_ID], stn[STATE], str(stn[LON]), str(stn[LAT]), str(stn[ELEV]), str(elev_dem), str(dif), "", "", "", "\n"]))

    fout.close()

def set_usrname_geonames(usr_name):
    '''
    Set a global USRNAME_GEONAMES username variable for _get_elev_geonames
    
    Parameters
    ----------
    usr_name : str
        A username for geonames
    '''

    global USRNAME_GEONAMES
    USRNAME_GEONAMES = usr_name
