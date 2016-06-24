OVERVIEW
The TopoWx ("Topography Weather") gridded dataset contains %d-%d 30-arcsec resolution (~800-m) 
interpolations of daily minimum and maximum topoclimatic air temperature for the conterminous U.S. 
Using both DEM-based variables and MODIS land skin temperature as predictors of air temperature, 
interpolation procedures include moving window regression kriging and geographically weighted 
regression. To avoid artificial climate trends, all input station data are homogenized using the 
GHCN/USHCN Pairwise Homogenization Algorithm. 

DATA FORMAT
Data are provided in netCDF-4 format (http://www.unidata.ucar.edu/software/netcdf/). Data is 
internally compressed. Chunking is optimized for spatial access 
(http://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_why_it_ matters). A separate 
minimum and maximum temperature netCDF-4 file is provided for each year for both daily and monthly
data. Separate files are also provided for 1981-2010 monthly normals and corresponding interpolation
uncertainty (prediction standard error). 

REFERENCE / HOW TO CITE
Oyler, J. W., Ballantyne, A., Jencso, K., Sweet, M. and Running, S. W. (2015), Creating a 
topoclimatic daily air temperature dataset for the conterminous United States using homogenized 
station data and remotely sensed land skin temperature. Int. J. Climatol. 35 (9): 2258–79. 
http://dx.doi.org/10.1002/joc.4127. 

LICENSE
This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License (http://creativecommons.org/licenses/by-nc-sa/4.0/). 

OPEN SOURCE REPOSITORY
An Open Source code repository for TopoWx is available on GitHub at https://github.com/jaredwo/topowx 
Future enhancements and updates will be coordinated through this repository.

VERSION
Software version: %s
Data version: %s

CONTACT
Jared Oyler (jaredwo@gmail.com) 
