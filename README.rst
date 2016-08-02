##########
TopoWx
##########

TopoWx ("Topography Weather") is an open source framework written in Python and
R for interpolating temperature observations at a "topoclimatic" spatial scale
(<= 10 km). Using digital elevation model (DEM) variables and remotely sensed
observations of land skin temperature, TopoWx empirically models the effect
of various topoclimatic factors (eg. elevation, cold air drainage potential,
slope/aspect, coastal proximity) on air temperature. The current interpolation
procedures include moving window regression kriging and geographically
weighted regression. To better ensure temporal consistency, TopoWx homogenizes
all input station data using the `GHCN/USHCN Pairwise Homogenization
Algorithm <http://www.ncdc.noaa.gov/oa/climate/research/ushcn/#phas>`_. TopoWx
was developed at University of Montana within the `Numerical Terradynamic
Simulation Group <http://www.ntsg.umt.edu>`_ and the `Montana Climate
Office <http://www.climate.umt.edu>`_, and is currently maintained through
the `Network for Sustainable Climate Risk Management (SCRiM) at Penn State
<http://www.scrimhub.org/resources/topowx/>`_. TopoWx has been used to
produce a 1948-present 30-arcsec resolution (~800-m) `gridded dataset
<http://www.scrimhub.org/resources/topowx/>`_ of daily minimum and maximum 
topoclimatic air temperature for the conterminous U.S.

Code and Installation
=============
TopoWx is provided as a Python package (twx), but also uses several R
modules. The target audience includes the climate science and climate impacts
research communities, developers, and statisticians. Step-by-step example
python scripts for using twx to build a 1948-present gridded temperature
dataset for the conterminous U.S. are provided in the scripts directory. For
twx installation instructions and requirements, see the INSTALL file.

Homepage
=============
http://www.scrimhub.org/resources/topowx/

References
=============
Oyler, J.W., Ballantyne, A., Jencso, K., Sweet, M. and Running, S. W. (2014).
Creating a topoclimatic daily air temperature dataset for the conterminous
United States using homogenized station data and remotely sensed land skin
temperature. International Journal of Climatology. http://dx.doi.org/10.1002/joc.4127.

Oyler, J.W., Dobrowski, S.Z., Ballantyne, A.P., Klene, A.E., Running, S.W.
(2015). Artificial amplification of warming trends across the mountains of
the western United States. Geophysical Research Letters.
http://dx.doi.org/10.1002/2014GL062803.

Oyler, J.W., S.Z. Dobrowski, Z.A. Holden, and S.W. Running (2016), Remotely
sensed land skin temperature as a spatial predictor of air temperature across
the conterminous United States. Journal of Applied Meteorology and Climatology.
http://dx.doi.org/10.1175/JAMC-D-15-0276.1.

Acknowledgements
=============
Development of the current version of TopoWx was supported by the National
Science Foundation through the Network for Sustainable Climate Risk Management
(SCRiM) under NSF cooperative agreement GEO-1240507. Original TopoWx development
was supported by the National Science Foundation under EPSCoR Grant EPS-1101342,
the US Geological Survey North Central Climate Science Center Grant G-0734-2,
and the US Geological Survey Energy Resources Group Grant G11AC20487.