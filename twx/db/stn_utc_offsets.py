'''
Utilities for retrieving time zone information for 
a specific lon,lat.

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

__all__ = ['UtcOffset', 'GeonamesError', 'TZGeonamesClient']

import numpy as np
import urllib
import urllib2
import json
import fiona
from shapely.geometry import shape, MultiPolygon, Point, box
from functools import partial
import pyproj
from shapely.ops import transform

class UtcOffset():
    '''
    Class for retrieving time zone offset from
    Coordinated Universial Time (UTC) for a specific point.
    '''

    def __init__(self, fpath_timezone_shp, ndata=-32767, geonames_usrname=None, tz_shp_bnds=None):
        '''
        Parameters
        ----------
        fpath_timezone_shp : str
            Path to world_timezones.shp shapefile that defines
            UTC offsets. Downloaded from http://www.sharegeo.ac.uk/handle/10672/285.
        ndata : int, optional
            The value that should be returned if no time zone
            information can be found for the point of interest.
        geonames_usrname : str, optional
            A geonames username. If a geonames username is provided,
            the Geonames web service will be checked for time zone
            information if no information on a point's time zone can
            be found via the local shapefile.
        tz_shp_bnds : tuple, optional
            Lon/lat bounding box (min lon, min lat, max lon, max lat).
            Only timezone polygons from world_timezones.shp that intersect 
            this bounding box will be considered when searching for a point's
            UTC offset. If you know all points will only be in a certain region
            of the globe, this can significantly improve search time. By default,
            all polygons are searched (tz_shp_bnds=None).
        '''

        fpath_utc_shpfile = fpath_timezone_shp

        tz_shp = fiona.open(fpath_utc_shpfile)

        print "Loading timezone UTC offsets..."
        self.offsets = [p['properties']['UTC_OFFSET'] for p in tz_shp]
        self.offsets = np.array([_utc_int(utc_str, ndata) for utc_str in self.offsets])

        print "Loading timezone polygons..."
        tz_polys = MultiPolygon([shape(poly['geometry']) for poly in tz_shp])
        
        # Need to transform from geographic lon/lat to cartesian (spherical Mercator in this case)
        # so that shapely "contains" function is accurate.
        # http://stackoverflow.com/questions/21328854/shapely-and-matplotlib-point-in-polygon-not-accurate-with-geolocation
        self._project = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                                pyproj.Proj('+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +no_defs'))
        
        print "Projecting timezone polygons..."
        tz_polys = transform(self._project, tz_polys)
        
        self.tz_geoms = np.array([g for g in tz_polys.geoms])
        
        if tz_shp_bnds is not None:
            
            # Remove timezone polygons that are not in specified bounding box
            b = box(tz_shp_bnds[0], tz_shp_bnds[1], tz_shp_bnds[2], tz_shp_bnds[3])
            b = transform(self._project, b)
            mask_b = np.array([b.intersects(g) for g in self.tz_geoms])
            
            self.tz_geoms = self.tz_geoms[mask_b]
            self.offsets = self.offsets[mask_b]
        
        if geonames_usrname is None:
            self.tz_geon = None
        else:
            self.tz_geon = TZGeonamesClient(geonames_usrname)

        self.ndata = ndata

    def get_utc_offset(self, lon, lat):
        '''
        Retrieve the UTC offset for a specific point. First checks
        a local shapefile of time zones. If the time zone of the point
        cannot be determined from the shapefile, the Geonames data web
        service will be checked if a Geonames username was provided on
        UtcOffset object creation.
        
        Parameters
        ----------
        lon : double
            The longitude of the point
        lat : double
            The latitude of the point
            
        Returns
        ----------
        offset : int
            The UTC offset for the point. If the offset cannot be
            determined, the ndata value is returned.
        '''

        offset = self.ndata
        
        pt = Point(lon, lat)
        # reproject point
        pt = transform(self._project, pt)
        
        i = np.nonzero(np.array([g.contains(pt) for g in self.tz_geoms]))[0]
        
        if i.size == 1:

            offset = self.offsets[i[0]]
        
        if offset == self.ndata and self.tz_geon is not None:
             
            print "UtcOffset: Could not find UTC polygon for point %.4f,%.4f. Trying geonames..." % (lon, lat)
            offset = self.tz_geon.get_utc_offset(lon, lat)

        return offset


def _utc_int(utc_str, ndata=-32767):

    try:
        int_utc = int(utc_str[3:6])
    except (ValueError, TypeError):
        int_utc = ndata

    return int_utc


class GeonamesError(Exception):
    '''
    Represents an error when retrieving time zone
    information from the Geonames data web service
    Written by: https://gist.github.com/pamelafox/2288222.
    '''

    def __init__(self, status):
        Exception.__init__(self, status)
        self.status = status

    def __str__(self):
        return self.status

    def __unicode__(self):
        return unicode(self.__str__())


class TZGeonamesClient(object):
    '''
    Class for retrieving time zone information for a specific
    point from the Geonames data web service.
    Written by: https://gist.github.com/pamelafox/2288222.
    '''

    BASE_URL = 'http://api.geonames.org/'

    def __init__(self, username):
        '''
        Parameters
        ----------
        username : str
            A geonames username
        '''

        self.username = username

    def __call(self, service, params=None):

        url = self.__build_url(service, params)

        try:
            response = urllib2.urlopen(urllib2.Request(url))
            json_response = json.loads(response.read())
        except urllib2.URLError:
            raise GeonamesError('API didnt return 200 response.')
        except ValueError:
            raise GeonamesError('API did not return valid json response.')
        else:
            if 'status' in json_response:
                raise GeonamesError(json_response['status']['message'])
        return json_response

    def __build_url(self, service, params=None):
        url = '%s%s?username=%s' % (TZGeonamesClient.BASE_URL, service, self.username)
        if params:
            if isinstance(params, dict):
                params = dict((k, v) for k, v in params.items() if v is not None)
                params = urllib.urlencode(params)
            url = '%s&%s' % (url, params)
        return url

    def find_timezone(self, lon, lat):
        # http://api.geonames.org/timezoneJSON?lat=47.01&lng=10.2&username=demo
        return self.__call('timezoneJSON', {'lat':lat, 'lng':lon})

    def get_utc_offset(self, lon, lat):

        tz = self.find_timezone(lon, lat)
        return tz['rawOffset']
