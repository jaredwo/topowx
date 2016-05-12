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

__all__ = ['get_elevation', 'LocQA']

from time import sleep
from urllib2 import HTTPError
import json
import numpy as np
import urllib
import urllib2
import pandas as pd

def _get_elev_usgs(lon, lat, maxtries):
    """Get elev value from USGS NED 1/3 arc-sec DEM.

    http://ned.usgs.gov/epqs/
    """

    URL_USGS_NED = 'http://ned.usgs.gov/epqs/pqs.php'
    USGS_NED_NODATA = -1000000

    # url GET args
    values = {'x': lon,
              'y': lat,
              'units': 'Meters',
              'output': 'json'}

    data = urllib.urlencode(values)

    req = urllib2.Request(URL_USGS_NED, data)
                
    ntries = 0
    
    while 1:
        
        try:
            
            response = urllib2.urlopen(req)
            break

        except HTTPError:
            
            ntries += 1
        
            if ntries >= maxtries:
        
                raise
        
            sleep(1)
            
    json_response = json.loads(response.read())
    elev = np.float(json_response['USGS_Elevation_Point_Query_Service']
                    ['Elevation_Query']['Elevation'])

    if elev == USGS_NED_NODATA:

        elev = np.nan

    return elev

def _get_elev_geonames(lon, lat, usrname_geonames, maxtries):
    """Get elev value from geonames web sevice (SRTM or ASTER)
    """

    URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
    URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'

    url = URL_GEONAMES_SRTM

    while 1:
        # ?lat=50.01&lng=10.2&username=demo
        # url GET args
        values = {'lat': lat, 'lng': lon, 'username': usrname_geonames}

        # encode the GET arguments
        data = urllib.urlencode(values)

        # make the URL into a qualified GET statement
        get_url = "".join([url, "?", data])

        req = urllib2.Request(url=get_url)
        
        ntries = 0
        
        while 1:
            
            try:
                
                response = urllib2.urlopen(req)
                break

            except HTTPError:
                
                ntries += 1
            
                if ntries >= maxtries:
            
                    raise
            
                sleep(1)
        
        elev = float(response.read().strip())

        if elev == -32768.0 and url == URL_GEONAMES_SRTM:
            # Try ASTER instead
            url = URL_GEONAMES_ASTER
        else:
            break

    if elev == -32768.0 or elev == -9999.0:
        elev = np.nan

    return elev

def get_elevation(lon, lat, usrname_geonames=None, maxtries=3):

    elev = _get_elev_usgs(lon, lat, maxtries)

    if np.isnan(elev) and usrname_geonames is not None:

        elev = _get_elev_geonames(lon, lat, usrname_geonames, maxtries)

    return elev

class LocQA(object):
    '''Class for managing location quality assurance HDF database
    '''
    
    _cols_locqa = ['station_id','station_name', 'longitude', 'latitude',
                   'elevation','elevation_dem','longitude_qa','latitude_qa',
                   'elevation_qa']

    def __init__(self, fpath_locqa_hdf, mode='a', usrname_geonames=None):
        
        self.fpath_locqa_hdf = fpath_locqa_hdf
        self.usrname_geonames = usrname_geonames
        
        self._store = pd.HDFStore(fpath_locqa_hdf, mode)
        self.reload_stns_locqa()
                    
    def reload_stns_locqa(self):
        
        try:
            
            self._stns_locqa = self._store.select('stns')
            
        except KeyError:

            self._stns_locqa = pd.DataFrame(columns=self._cols_locqa)
            
            # Make sure numeric columns are float
            cols_flt = ['longitude','latitude','elevation','elevation_dem',
                        'longitude_qa','latitude_qa','elevation_qa']
            
            self._stns_locqa[cols_flt] = self._stns_locqa[cols_flt].astype(np.float)
        
        self._stns_locqa = self._stns_locqa[~self._stns_locqa.index.duplicated(keep='last')].copy()
        
    
    def get_elevation_dem(self, lon, lat):
        
        return get_elevation(lon, lat, self.usrname_geonames)
        
    def update_locqa_hdf(self, stns, reload_locqa=True):
        
        self._store.append('stns', stns[self._cols_locqa], min_itemsize={'station_id':50,
                                                                         'station_name':50,
                                                                         'index':50})
        self._store.flush()
        
        if reload_locqa:
        
            self.reload_stns_locqa()
    
    def add_locqa_cols(self, stns):
        
        stns = stns.copy()
        
        locqa_cols = ['elevation_dem', 'longitude_qa', 'latitude_qa', 'elevation_qa']
        
        isclose_cols = ['longitude', 'latitude', 'elevation']
#        isclose_cols_ = (pd.Series(isclose_cols) + '_').values
#        rtols = [0]*len(isclose_cols)
#        atols = [1e-02,1e-02,1]
        
        a_stns = stns[isclose_cols].join(self._stns_locqa, how='left', rsuffix='_')
        
        # Check if lon, lat, elev of a station is close to lon, lat, elev of
        # the station in the loc qa store
#        mask_isclose = self._isclose(a_stns, isclose_cols, isclose_cols_, rtols, atols)
        # Check if lon, lat, elev of a station is close to the qa lon, lat, elev
        # of the station
#         mask_isclose2 = self._isclose(a_stns, isclose_cols,
#                                       ['longitude_qa', 'latitude_qa',
#                                        'elevation_qa'], rtols, atols) 
#         
#         # If 
#         a_stns.loc[~np.logical_or(mask_isclose1, mask_isclose2), locqa_cols] = np.nan
        
        for c in locqa_cols:
            stns[c] = a_stns[c]
                                        
        return stns
        
    def get_locqa_fail_stns(self, stns, elev_dif_thres=200):
        
        stns = stns.copy()
        stns['elevation_dif'] = stns.elevation - stns.elevation_dem 
        
        mask_fail = ((stns.elevation_dif.abs() > elev_dif_thres) |
                     (stns.elevation_dif.isnull())).values
        mask_noqa = ((stns.longitude_qa.isnull()) | (stns.latitude_qa.isnull())
                     | (stns.elevation_qa.isnull())).values
        mask_fnl = np.logical_and(mask_fail, mask_noqa)
                     
        return stns[mask_fnl].copy()
        
        
    def _isclose(self, stns, cols1, cols2, rtols, atols):
    
        mask_isclose = np.ones(len(stns), dtype=np.bool)
        
        for c1,c2,rtol,atol in zip(cols1,cols2,rtols,atols):
            
            a_mask = np.isclose(stns[c1].values, stns[c2].values,
                                rtol=rtol, atol=atol, equal_nan=True)
            mask_isclose = np.logical_and(a_mask, mask_isclose)
        
        return mask_isclose
    
    def close(self):
        
        self._store.close()
        self._store = None
    