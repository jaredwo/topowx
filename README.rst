##########
TopoWx
##########

TopoWx ("Topography Weather") is an open source framework written in Python and
R for interpolating temperature observations at a "topoclimatic" spatial scale
(<= 10 km). Using digital elevation model (DEM) variables and remotely sensed
observations of land skin temperature, TopoWx empirically models the effect
of various topoclimatic factors (eg. elevation, cold air drainge potential,
slope/aspect, coastal proximity) on air temperature. The current interpolation
procedures include moving window regression kriging and geographically
weighted regression.  To avoid artificial climate trends, TopoWx homogenizes
all input station data using the `GHCN/USHCN Pairwise Homogenization
Algorithm <http://www.ncdc.noaa.gov/oa/climate/research/ushcn/#phas>`_. TopoWx
was developed at University of Montana within the `Numerical Terradynamic
Simulation Group <http://www.ntsg.umt.edu>`_ and the `Montana Climate
Office <http://www.climate.umt.edu>`_. TopoWx has been used to
produce a 1948-present 30-arcsec resolution (~800-m) `gridded dataset
<http://www.ntsg.umt.edu/project/TopoWx>`_ of daily minimum and maximum 
topoclimatic air temperature for the conterminous U.S.

Code and Installation
=============
TopoWx is provided as a Python package (twx), but also uses several R
modules. The target audience includes the climate science and climate impacts
research communities, developers, and statisticians. Step-by-step example
python scripts for using twx to build a 1948-present gridded temperature
dataset for the conterminous U.S. are provided in the bin directory. For
twx installation instructions and requirements, see the INSTALL file.

Homepage
=============
http://www.ntsg.umt.edu/project/TopoWx

References
=============
Oyler, J. W., Ballantyne, A., Jencso, K., Sweet, M. and Running, S. W. (2014),
Creating a topoclimatic daily air temperature dataset for the conterminous
United States using homogenized station data and remotely sensed land skin
temperature. Int. J. Climatol. http://dx.doi.org/10.1002/joc.4127.

Oyler, J.W., Dobrowski, S.Z., Ballantyne, A.P., Klene, A.E., Running, S.W. (In
Press). Artificial Amplification of Warming Trends Across the Mountains of
the Western United States. Geophysical Research Letters.