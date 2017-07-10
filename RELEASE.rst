v1.3.0 (July 10, 2017)
=============

This release updates the TopoWx dataset through 2016. Highlights include:

* Minor code maintenance and technical bug fixes
* No algorithm or science changes with this release

v1.2.0 (August 2, 2016)
=============

This release updates the TopoWx dataset through 2015. Highlights include:

* Added monthly USHCN observations and SNOTEL sensor change metadata to
  the homogenization step to increase homogenization algorithm stability
* Improved script automation and refactored 

v1.1.0 (August 10, 2015)
=============

This release updates the TopoWx dataset through 2014 and includes several
enhancements. Highlights include:

* Added 1151 additional input stations (498 GHCN-D stations, 499 RAWS stations,
  154 SNOTEL/SCAN stations).
* Homogenized observations over 1895-2015 time period instead of 1948-2012.
* Modified missing value infilling procedures to build a separate model for
  each month instead of a single model for the whole year. This avoids seasonal
  infill biases.
* Improved coverage along coastlines by using nearest neighbor resampling
  of MODIS LST where bilinear resampling failed.

v1.0.0 (December 29, 2014)
=============

First public TopoWx release.

