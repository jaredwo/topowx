Installing from Source
=============

The TopoWx code library is provided to facilitate community-driven improvement
of topoclimatic datasets. It is not meant to be a 'user-friendly' end-user
application. TopoWx has been tested on Linux (CentOS 5.x,6.x) and MacOSX.
The main code library is provided in the twx Python package and can be
installed via the setup.py script:
 
  python setup.py install

if you just plan to run TopoWx and not modify the library, or

  python setup.py develop

if you plan to modify the TopoWx library.

Step-by-step Examples
=============

Example python scripts for using twx are provided in the bin directory. These
scripts were used to build the 1948-present gridded temperature
dataset for the conterminous U.S. described by `Oyler et al. (2014)
<http://dx.doi.org/10.1002/joc.4127>`_.

Required Dependencies
=============

Below are the required dependencies for the twx package. Version numbers
represent the version of the library used in producing the 1948-present
gridded temperature dataset for the conterminous U.S. Older or newer versions
may work, but have not been fully tested.

Python Packages
-----------------

* fiona 1.5.1
* gdal 1.11.2
* matplotlib 1.4.3 (basemap toolkit 1.0.7)
* mpi4py 1.3.1
* netCDF4 1.1.8 (netCDF 4.3.2, HDF5 1.8.14)
* numpy 1.9.2
* pandas 0.16.2
* pyproj 1.9.4
* rpy2 2.6.0 (R version 3.2.0)
* scipy 0.15.1
* shapely 1.5.8
* statsmodels 0.6.1
* tzwhere 1.0
* xray 0.5.2

R Packages
-----------------

* changepoint 1.1.5
* gstat 1.0-25
* norm 1.0-9.5
* pcaMethods 1.58.0
* sp 1.1-1

External programs
-----------------

* `Pairwise Homogenization Algorithm (v52i) <http://www.ncdc.noaa.gov/oa/climate/research/ushcn/#phas>`_

