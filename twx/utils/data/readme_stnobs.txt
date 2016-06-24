README FILE FOR TOPOWX STATION OBSERVATION DATA

TopoWx input daily station data (%d-%d) are provided in netCDF4 format. The
main observation variables are structured as 2-dimensional netCDF
variables with shape: (time, station id). Each column is an individual station
time series. The main netCDF variables are also internally compressed
with each data chunk a single station observation time series. Separate files
are provided for both Tmin and Tmax. Each file contains the original raw
observations, the QA'd and homogenized observations, and the final missing
value infilled observations used as input to TopoWx.

A CSV file (homog_adjust.csv) of all homogenization adjustments is also
provided. Each line in the CSV file displays the time period over which
an adjustment was made (YEAR_MONTH_START, YEAR_MONTH_END), the adjustment
magnitude in degrees C (ADJ), the temperature variable (Tmin or Tmax),
and the station metadata.

Note: The netCDF station data files are not fully CF compliant.

Contact: Jared Oyler (jaredwo@gmail.com)

The individual station data files are as follows:

