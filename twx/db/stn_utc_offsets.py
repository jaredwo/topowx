'''
Utilities for retrieving time zone information for 
a specific lon,lat. 
'''

__all__ = ['UtcOffset', 'GeonamesError', 'TZGeonamesClient']

import numpy as np
import urllib
import urllib2
import json
import fiona
from shapely.geometry import shape, MultiPolygon, Point
import os

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

        path_root = os.path.dirname(__file__)
        fpath_utc_shpfile = os.path.join(path_root, 'data', 'timezones', 'world_timezones.shp')

        tz_shp = fiona.open(fpath_utc_shpfile)

        print "Loading timezone UTC offsets..."
        self.offsets = [p['properties']['UTC_OFFSET'] for p in tz_shp]
        self.offsets = np.array([_utc_int(utc_str, ndata) for utc_str in self.offsets])

        print "Loading timezone polygons..."
        self.tz_polys = MultiPolygon([shape(poly['geometry']) for poly in tz_shp])

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

        pt = Point(lon, lat)
        i = np.nonzero(np.array([g.contains(pt) for g in self.tz_polys.geoms]))[0]

        if i.size == 1:

            return self.offsets[i[0]]

        elif self.tz_geon is not None:

            print "UtcOffset: Could not find UTC polygon for point %.4f,%.4f. Trying geonames..." % (lon, lat)
            return self.tz_geon.get_utc_offset(lon, lat)

        else:

            return self.ndata


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

# if __name__ == '__main__':
#
#     db = StationDataDb('/projects/daymet2/station_data/all/all_1948_2012.nc',mode='r+')
#     varutc = db.add_stn_variable("utc_offset","utc offset","hours","i2")
#     utc_off = np.ma.filled(varutc[:].astype(np.float),np.nan)
#
#     tz = TZGeonamesClient('jaredntsg')
#
#     #Load the period-of-record datafile
#     por = load_por_csv('/projects/daymet2/station_data/all/all_por_1948_2012.csv')
#     mask_por_tmin,mask_por_tmax = build_valid_por_masks(por)[0:2]
#
#     mask_stns = np.logical_and(np.logical_or(mask_por_tmin,mask_por_tmax),np.isnan(utc_off))
#     idxs = np.nonzero(mask_stns)[0]
#
#     print idxs.size
#     #sys.exit()
#
#     statchk = status_check(idxs.size,50)
#
#     for x in idxs:
#
#         try:
#             varutc[x] = tz.get_utc_offset(db.stns[LON][x], db.stns[LAT][x])
#         except GeonamesError as e:
#             if 'no timezone information' in e.status:
#                 print 'No timezone info for ' + str(db.stns[x])
#             else:
#                 raise e
#
#         db.ds.sync()
#         statchk.increment()
#         time.sleep(1.0)





