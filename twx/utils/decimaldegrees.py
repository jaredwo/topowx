# -*- coding: utf-8 -*-
"""
PyDecimalDegrees - geographic coordinates conversion utility.

Copyright (C) 2006 by Mateusz Łoskot <mateusz@loskot.net>

This file is part of PyDecimalDegrees module.

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from
the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:
1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

DESCRIPTION

DecimalDegrees module provides functions to convert between
degrees/minutes/seconds and decimal degrees.

Inspired by Walter Mankowski's Geo::Coordinates::DecimalDegrees module
for Perl, originally located in CPAN Archives:
http://search.cpan.org/~waltman/Geo-Coordinates-DecimalDegrees-0.05/

doctest examples are based following coordinates:
DMS: 121 8' 6"
DM: 121 8.1'
DD: 121.135

To run doctest units just execut this module script as follows
(-v instructs Python to run script in verbose mode):

$ python decimaldegrees.py [-v]

"""
__revision__ = "$Revision: 1.0 $"


def decimal2dms(decimal_degrees):
    """ Converts a floating point number of degrees to the equivalent
    number of degrees, minutes, and seconds, which are returned
    as a 3-element list. If 'decimal_degrees' is negative,
    only degrees (1st element of returned list) will be negative,
    minutes (2nd element) and seconds (3rd element) will always be positive.

    Example:
    >>> decimal2dms(121.135)
    [121, 8, 6.0000000000184173]
    >>> decimal2dms(-121.135)
    [-121, 8, 6.0000000000184173]
    
    """

    degrees = int(decimal_degrees)
    decimal_minutes = abs(decimal_degrees - degrees) * 60
    minutes = int(decimal_minutes)
    seconds = (decimal_minutes - minutes) * 60
    return [degrees, minutes, seconds]


def decimal2dm(decimal_degrees):
    """ Converts a floating point number of degrees to the equivalent
    number of degrees and minutes, which are returned as a 2-element list.
    If 'decimal_degrees' is negative, only degrees (1st element of returned list)
    will be negative, minutes (2nd element) will always be positive.

    Example:
    >>> decimal2dm(121.135)
    [121, 8.100000000000307]
    >>> decimal2dm(-121.135)
    [-121, 8.100000000000307]
    
    """

    degrees = int(decimal_degrees) 
    minutes = abs(decimal_degrees - degrees) * 60
    return [degrees, minutes]


def dms2decimal(degrees, minutes, seconds):
    """ Converts degrees, minutes, and seconds to the equivalent
    number of decimal degrees. If parameter 'degrees' is negative,
    then returned decimal-degrees will also be negative.
    
    Example:
    >>> dms2decimal(121, 8, 6)
    121.13500000000001
    >>> dms2decimal(-121, 8, 6)
    -121.13500000000001
    
    """
    
    decimal = 0.0
    if (degrees >= 0):
        decimal = degrees + float(minutes)/60 + float(seconds)/3600
    else:
        decimal = degrees - float(minutes)/60 - float(seconds)/3600
        
    return decimal


def dm2decimal(degrees, minutes):
    """ Converts degrees and minutes to the equivalent number of decimal
    degrees. If parameter 'degrees' is negative, then returned decimal-degrees
    will also be negative.
    
    Example:
    >>> dm2decimal(121, 8.1)
    121.13500000000001
    >>> dm2decimal(-121, 8.1)
    -121.13500000000001
    
    """
    return dms2decimal(degrees, minutes, 0)


