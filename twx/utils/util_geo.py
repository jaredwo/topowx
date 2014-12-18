'''
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
import numpy

RADIAN_CONVERSION_FACTOR = 0.017453292519943295 #pi/180
AVG_EARTH_RADIUS_KM = 6371.009 #Mean earth radius as defined by IUGG

def grt_circle_dist(lon1,lat1,lon2,lat2):
        '''
        Calculate great circle distance according to the haversine formula
        see http://en.wikipedia.org/wiki/Great-circle_distance
        '''
        #convert to radians
        lat1rad = lat1 * RADIAN_CONVERSION_FACTOR
        lat2rad = lat2 * RADIAN_CONVERSION_FACTOR
        lon1rad = lon1 * RADIAN_CONVERSION_FACTOR
        lon2rad = lon2 * RADIAN_CONVERSION_FACTOR
        deltaLat = lat1rad - lat2rad
        deltaLon = lon1rad - lon2rad
        centralangle = 2 * numpy.arcsin(numpy.sqrt((numpy.sin (deltaLat/2))**2 + numpy.cos(lat1rad) * numpy.cos(lat2rad) * (numpy.sin(deltaLon/2))**2))
        #average radius of earth times central angle, result in kilometers
        #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
        distKm = AVG_EARTH_RADIUS_KM * centralangle 
        return distKm


def dist_ca(lon1,lat1,lon2,lat2):
        '''
        Calculate great circle distance according to the haversine formula
        see http://en.wikipedia.org/wiki/Great-circle_distance
        Also return central angle 
        '''
        #convert to radians
        lat1rad = lat1 * RADIAN_CONVERSION_FACTOR
        lat2rad = lat2 * RADIAN_CONVERSION_FACTOR
        lon1rad = lon1 * RADIAN_CONVERSION_FACTOR
        lon2rad = lon2 * RADIAN_CONVERSION_FACTOR
        deltaLat = lat1rad - lat2rad
        deltaLon = lon1rad - lon2rad
        centralangle = 2 * numpy.arcsin(numpy.sqrt((numpy.sin (deltaLat/2))**2 + numpy.cos(lat1rad) * numpy.cos(lat2rad) * (numpy.sin(deltaLon/2))**2))
        #average radius of earth times central angle, result in kilometers
        #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
        distKm = AVG_EARTH_RADIUS_KM * centralangle 
        return distKm,centralangle


def dist_ca_slc(lon1,lat1,lon2,lat2):
        '''
        Calculate great circle distance according to the haversine formula
        see http://en.wikipedia.org/wiki/Great-circle_distance
        Also return central angle 
        '''
        #convert to radians*numpy.cos(lat2rad)*cos(deltaLon)
        lat1rad = lat1 * RADIAN_CONVERSION_FACTOR
        lat2rad = lat2 * RADIAN_CONVERSION_FACTOR
        lon1rad = lon1 * RADIAN_CONVERSION_FACTOR
        lon2rad = lon2 * RADIAN_CONVERSION_FACTOR
        deltaLon = lon1rad - lon2rad

        centralangle = numpy.arccos((numpy.sin(lat1rad)*numpy.sin(lat2rad))+((numpy.cos(lat1rad)*numpy.cos(lat2rad))*numpy.cos(deltaLon)))
        #average radius of earth times central angle, result in kilometers
        #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
        distKm = AVG_EARTH_RADIUS_KM * centralangle 
        return distKm,centralangle