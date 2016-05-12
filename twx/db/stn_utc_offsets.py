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

from datetime import datetime
import json
import numpy as np
import pytz
import urllib
import urllib2

# tzwhere currently sets log level to debug when imported
# get log level before import and then reset log level to this
# value after tzwhere import
import logging
logger = logging.getLogger()
log_level = logger.level

from tzwhere.tzwhere import tzwhere

logger.setLevel(log_level)

class UtcOffset():
    '''
    Class for retrieving time zone offset from
    Coordinated Universial Time (UTC) for a specific point.
    '''

    def __init__(self, ndata=-32767, geonames_usrname=None):
        '''
        Parameters
        ----------
        ndata : int, optional
            The value that should be returned if no time zone
            information can be found for the point of interest.
        geonames_usrname : str, optional
            A geonames username. If a geonames username is provided,
            the Geonames web service will be checked for time zone
            information if no information on a point's time zone can
            be found via the local shapefile.
        '''
        
        print "Initializing tzwhere..."
        self.tzw = tzwhere(shapely=True, forceTZ=True)
        
        self.tz_offsets = {}
        tz_names = np.array(self.tzw.timezoneNamesToPolygons.keys())
        tz_names = tz_names[tz_names != 'uninhabited']
        
        a_date = datetime(2009,1,1)
        
        for a_name in tz_names:
            a_tz = pytz.timezone(a_name)
            self.tz_offsets[a_name] = a_tz.utcoffset(a_date).total_seconds()/3600.0
                
        if geonames_usrname is None:
            self.tz_geon = None
        else:
            self.tz_geon = TZGeonamesClient(geonames_usrname)

        self.ndata = ndata

    def get_utc_offset(self, lon, lat):
        '''
        Retrieve the UTC offset for a specific point. First checks
        local polygon file of time zones. If the time zone of the point
        cannot be determined locally, the Geonames data web
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
        
        tz_name = self.tzw.tzNameAt(lat, lon, forceTZ=True)
        offset = self.ndata
        
        if tz_name is not None:
            offset = self.tz_offsets[tz_name]
        
        
        if offset == self.ndata and self.tz_geon is not None:
             
            print "UtcOffset: Could not find UTC polygon for point %.4f,%.4f. Trying geonames..." % (lon, lat)
            offset = self.tz_geon.get_utc_offset(lon, lat)

        return offset

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
