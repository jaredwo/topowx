'''
Utility classes for managing and writing out
interpolation results to netCDF tiles.

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

__all__ = ['Tiler', 'TileWriter', 'TileGridInfo']

import numpy as np
from netCDF4 import Dataset, date2num
import os
import datetime
from datetime import date
from twx.utils import YMD, DATE
import netCDF4

# factor by which daily outputs are scaled.
# everything is stored as int16
SCALE_FACTOR = np.float32(0.01) 

# long name, units, standard name, missing value,cell method
VAR_ATTRS = {'tmin':("minimum air temperature", "C", "air_temperature", netCDF4.default_fillvals['i2'], "minimum"),
             'tmax':("maximum air temperature", "C", "air_temperature", netCDF4.default_fillvals['i2'], "maximum")}

class Tiler():
    '''
    A utility class for breaking up an interpolation grid into tiles
    and separate works chunks for processing
    '''
    
    def __init__(self, ds_mask, ds_attr_ls, tile_size_y, tile_size_x, chk_size_y, chk_size_x, path_out, process_tiles=False):
        '''
        Parameters
        ----------
        ds_mask : netCDF4.Dataset object
            A netCDF4 dataset that defines the interpolation grid
            and mask
        ds_attr_ls : list
            A list of netCDF4.Dataset objects that define
            grids of auxiliary predictors. Each list entry is
            a tuple of size 2 with the predictor name and the
            netCDF4.Dataset object
        tile_size_y : int
            The tile size (# of grid cells) in the y direction.
            The interpolation grid y size must be evenly divisible
            by this number.
        tile_size_x : int
            The tile size (# of grid cells) in the x direction.
            The interpolation grid x size must be evenly divisible
            by this number.
        chk_size_y : int
            The chunk size (# of grid cells) in the y direction.
            Defines the size of a unit of work within a tile.
            tile_size_y size must be evenly divisible by this number.
        chk_size_x : int
            The chunk size (# of grid cells) in the x direction.
            Defines the size of a unit of work within a tile.
            tile_size_x size must be evenly divisible by this number.
        path_out : str
            The path to which the netCDF tiles will be output
        process_tiles : list or boolean, optional
            If list, list of tile numbers to process. If boolean and True, will
            check which tiles already had a directory created in path_out, and remove
            them from the list of tiles to be processed. Note: this only checks
            the existence of the tile directory, not if the tile output is complete. 
            Default: None (all tiles will be processed).
        '''
        
        self.mask = np.array(ds_mask.variables['mask'][:], dtype=np.bool)
        self.lons = ds_mask.variables['lon'][:]
        self.lats = ds_mask.variables['lat'][:]
        self.nrows = self.lats.size
        self.ncols = self.lons.size
        self.tile_size_y = tile_size_y
        self.tile_size_x = tile_size_x
        self.chk_size_y = chk_size_y
        self.chk_size_x = chk_size_x
        
        self.attrs = []
        for varname, ds in ds_attr_ls:
        
            attr = ds.variables[varname]
            attr.set_auto_maskandscale(False)
            self.attrs.append(attr)
            
        self.tile_ids, self.tile_rc = self.__build_tile_dicts(self.nrows, self.ncols, tile_size_y, tile_size_x, self.mask)
        
        self.chk_size_i = 5 + len(self.attrs)
        
        self.wrk_chk = np.zeros((self.chk_size_i, chk_size_y, chk_size_x)) * np.nan
        
        try:
            
            self.process_tiles = list(process_tiles)
            
        except TypeError:
            
            if type(process_tiles) == bool:
                
                if process_tiles:
                    self.process_tiles = self.get_incomplete_tile_nums(path_out)
                else:
                    self.process_tiles = None
                    
            else:
                self.process_tiles = None
        
        
        self.__set_tile_chks()
    
    
    def __set_tile_chks(self):
    
        k = 0
        self.tile_chks = []
        self.ntiles = 0
        
        for i in np.arange(0, self.nrows, self.tile_size_y):
            
            for j in np.arange(0, self.ncols, self.tile_size_x):
                
                msk_tile = self.mask[i:i + self.tile_size_y, j:j + self.tile_size_x]
                
                if msk_tile[msk_tile].size > 0:
                    
                    process_tile = False
                    try:
                        if k in self.process_tiles:
                            process_tile = True
                    except TypeError:
                        process_tile = True
                    
                    if process_tile:
                        
                        for y in np.arange(0, self.tile_size_y, self.chk_size_y):
                            
                            for x in np.arange(0, self.tile_size_x, self.chk_size_x):
                                
                                self.tile_chks.append((k, i, j, y, x))
                    
                        self.ntiles += 1
                         
                    k += 1
            
            self.iter_x = 0
            self.ntile_chks = len(self.tile_chks)
            
    def next(self):
        '''
        Get the next work chunk to process.
        Raises StopIteration when there are no
        more chunks to process
        
        Returns
        -------
        k : int
            The tile # of the work chunk
        wrk_chk : ndarray
            A 3-D array of (5+N)*Y*X where
            N is the number of auxiliary predictors,
            Y is the work chunk size in the y direction,
            and X is the work chunk size in x direction.
            First five entries in the first dimension are:
            tile row, tile col, mask, latitude, longitude 
        '''
        
        if self.iter_x == self.ntile_chks:
            
            raise StopIteration()
        
        else:
            
            k, i, j, y, x = self.tile_chks[self.iter_x]
            
            msk_tile = self.mask[i:i + self.tile_size_y, j:j + self.tile_size_x]
            
            attr_tiles = []
            
            for attr in self.attrs:
                attr_tiles.append(attr[i:i + self.tile_size_y, j:j + self.tile_size_x])
                
            llgrid = np.meshgrid(self.lons[j:j + self.tile_size_x], self.lats[i:i + self.tile_size_y])
            
            rcgrid = np.mgrid[y:y + self.chk_size_y, x:x + self.chk_size_x]

            self.wrk_chk[0, :, :] = rcgrid[0, :, :]  # row
            self.wrk_chk[1, :, :] = rcgrid[1, :, :]  # col
            self.wrk_chk[2, :, :] = msk_tile[y:y + self.chk_size_y, x:x + self.chk_size_x]  # mask
            self.wrk_chk[3, :, :] = llgrid[1][y:y + self.chk_size_y, x:x + self.chk_size_x]  # lat
            self.wrk_chk[4, :, :] = llgrid[0][y:y + self.chk_size_y, x:x + self.chk_size_x]  # lon
            
            for z in np.arange(len(attr_tiles)):
                
                self.wrk_chk[5 + z, :, :] = attr_tiles[z][y:y + self.chk_size_y, x:x + self.chk_size_x]
            
            self.iter_x += 1
            
            return k, self.wrk_chk

    def __build_tile_dicts(self, nrows, ncols, tile_size_y, tile_size_x, mask):
        '''
        Builds dictionaries of tile ids and tile start row,cols 
        '''
        # Build dict of tile ids
        tile_ids = {}
        tile_info = {}
        x = 0
        cnt_y = 0
        cnt_x = 0
        for i in np.arange(0, nrows, tile_size_y):
            
            cnt_x = 0
            for j in np.arange(0, ncols, tile_size_x):
                
                msk_tile = mask[i:i + tile_size_y, j:j + tile_size_x]
                
                if msk_tile[msk_tile].size > 0:
                    
                    atileid = "".join(["h%02d" % (cnt_x,), "v%02d" % (cnt_y,)])
                    tile_ids[x] = atileid
                    tile_info[atileid] = (i, j)
                    x += 1
                
                cnt_x += 1
            
            cnt_y += 1
            
        return tile_ids, tile_info

    def build_tile_grid_info(self):
        '''
        Build a TileGridInfo object containing information on tiles
        and chunks for the interpolation grid.
        '''
        
        return TileGridInfo(self.tile_ids, self.tile_rc, self.ntiles, self.lons, self.lats,
                              self.tile_size_y, self.tile_size_x, self.chk_size_y, self.chk_size_x, self.chk_size_i)
    
    def get_incomplete_tile_nums(self,path_out):
        
        id_to_name = self.tile_ids
        name_to_id = {}
        all_ids = []
        
        for a_id,a_name in id_to_name.items():
            name_to_id[a_name] = a_id
            all_ids.append(a_id)
        
        all_ids = np.unique(all_ids)
        
        names_done = os.listdir(path_out)
        ids_done = np.unique([name_to_id[a_name] for a_name in names_done])
        
        id_mask = np.in1d(all_ids, ids_done, True)
        
        return all_ids[~id_mask]
                
    
class TileGridInfo():
    '''
    A class for holding information on tiles and work chunks
    for an interpolation grid. Normally generated 
    by Tiler.build_tile_grid_info()
    '''           
        
    def __init__(self, tile_ids, tile_rc, ntiles, lons, lats, tile_size_y, tile_size_x, chk_size_y, chk_size_x, chk_size_i):
        
        self.tile_ids = tile_ids
        self.tile_rc = tile_rc
        self.ntiles = ntiles
        self.lons = lons
        self.lats = lats
        self.tile_size_y = tile_size_y
        self.tile_size_x = tile_size_x
        self.chk_size_y = chk_size_y
        self.chk_size_x = chk_size_x
        self.chk_size_i = chk_size_i
        
        self.chks_per_tile = (tile_size_x / chk_size_x) * (tile_size_y / chk_size_y)
        self.nchks = self.chks_per_tile * ntiles
        
    def get_tile_id(self, tile_num):
        
        return self.tile_ids[tile_num]
    
class TileWriter():
    '''
    A utility class for writing out interpolation results
    to netCDF tiles.
    '''
    
    def __init__(self, tile_grid_info, path_out):
        '''
        Parameters
        ----------
        tile_grid_info : TileGridInfo object
            A TileGridInfo object from Tiler.build_tile_grid_info()
            Contains information on tiles and chunks for the interpolation grid.
        path_out : str
            The path to which to output the netCDF tiles
        '''
        
        self.tile_ids = tile_grid_info.tile_ids
        self.tile_rc = tile_grid_info.tile_rc
        self.ntiles = tile_grid_info.ntiles
        self.lons = tile_grid_info.lons
        self.lats = tile_grid_info.lats
        self.path_out = path_out
        self.tile_size_y = tile_grid_info.tile_size_y
        self.tile_size_x = tile_grid_info.tile_size_x
        self.chk_size_y = tile_grid_info.chk_size_y
        self.chk_size_x = tile_grid_info.chk_size_x
        
    def __open_dataset(self, tile_id, varname, days):
    
        fpath = os.path.join(self.path_out, tile_id, "%s_%s.nc" % (tile_id, varname))
        
        try:
            
            ds = Dataset(fpath, 'r+')
            
        except RuntimeError:
            
            if not os.path.exists(os.path.join(self.path_out, tile_id)):
                
                os.mkdir(os.path.join(self.path_out, tile_id))
            
            ds = self.__create_ncdf(fpath, tile_id, varname, days)
        
        return ds
    
    
    def __create_ncdf(self, fpath, tile_id, varname, days):
        
        ds = Dataset(fpath, 'w')
         
        # Set global attributes
        title = "".join(["Daily Interpolated Meteorological Data ", str(days[YMD][0]), "-", str(days[YMD][-1])])
        ds.title = title
        ds.institution = "University of Montana Numerical Terradynamics Simulation Group"
        ds.source = "TopoWx v1.0.0"
        ds.history = "".join(["Created on: ", datetime.datetime.strftime(date.today(), "%Y-%m-%d")]) 
        ds.references = "http://www.ntsg.umt.edu/project/TopoWx"
        ds.comment = "30-arcsec spatial resolution, daily timestep"
        ds.Conventions = "CF-1.6"
        
        str_row, str_col = self.tile_rc[tile_id]
        lons = self.lons[str_col:str_col + self.tile_size_x]
        lats = self.lats[str_row:str_row + self.tile_size_y]
        
        # Create dimensions
        ds.createDimension('time', days.size)
        ds.createDimension('lat', lats.size)
        ds.createDimension('lon', lons.size)
        ds.createDimension('nv', 2)
        ds.createDimension('time_normals', 12)
        
        min_date = days[DATE][0]
        
        # Create dimension variables and fill with values
        times = ds.createVariable('time', 'f8', ('time',), fill_value=False)
        times.long_name = "time"
        times.units = "".join(["days since ", str(min_date.year), "-", str(min_date.month), "-", str(min_date.day), " 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times.bounds = 'time_bnds'
        
        time_bnds = ds.createVariable('time_bnds', 'f8', ('time', 'nv'), fill_value=False)
        
        time_nums = date2num(days[DATE], times.units) + 0.5
        time_nums_bnds = np.empty((days.size, 2))
        time_nums_bnds[:, 0] = time_nums - 0.5
        time_nums_bnds[:, 1] = time_nums + 0.5
        
        times[:] = time_nums
        time_bnds[:] = time_nums_bnds

        time_means = ds.createVariable('time_normals', 'f8', ('time_normals',), fill_value=False)
        time_means.long_name = times.long_name
        time_means.units = times.units
        time_means.standard_name = times.standard_name
        time_means.calendar = times.calendar
        time_means.climatology = "climatology_bounds"
        time_means.comment = "Time dimension for the 1981-2010 monthly normals"
        
        clim_bnds = ds.createVariable('climatology_bounds', 'f8', ('time_normals', 'nv'), fill_value=False)
        
        def get_mid_date(dateStart, dateEnd):
            
            mdate = dateStart + ((dateEnd - dateStart) / 2)
            return datetime.datetime(mdate.year, mdate.month, mdate.day)
        
        for mth in np.arange(1, 13):
            
            mthNext = mth + 1 if mth != 12 else 1
            yrNext = 1981 if mth != 12 else 1982
            time_means[mth - 1] = date2num(get_mid_date(date(1981, mth, 1), date(yrNext, mthNext, 1)), times.units)
        
            yrNext = 2010 if mth != 12 else 2011
            clim_bnds[mth - 1, :] = date2num([datetime.datetime(1981, mth, 1), datetime.datetime(yrNext, mthNext, 1)], times.units)
        
        latitudes = ds.createVariable('lat', 'f8', ('lat',), fill_value=False)
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lats
    
        longitudes = ds.createVariable('lon', 'f8', ('lon',), fill_value=False)
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lons
        
        _add_crs_wgs84_var(ds)
        self.__add_dim_vars(ds, varname, days)
        
        ds.sync()
        
        return ds
        
    def __add_dim_vars(self, ds, varname, days):
        
        long_name, units, standard_name, fill_value, cell_method = VAR_ATTRS[varname]
        mainvar = ds.createVariable(varname, 'i2', ('time', 'lat', 'lon',),
                                    chunksizes=(days[DATE].size, self.chk_size_y, self.chk_size_x),
                                    fill_value=fill_value)
        mainvar.long_name = long_name
        mainvar.units = units
        mainvar.standard_name = standard_name
        mainvar.scale_factor = SCALE_FACTOR
        mainvar.cell_methods = "".join(["area: mean ", "time: ", cell_method])
        _add_grid_mapping_attr(mainvar)
        
        avar = ds.createVariable("".join([varname, "_normal"]), 'f4', ('time_normals', 'lat', 'lon',),
                                 chunksizes=(12, self.chk_size_y, self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar.long_name = "".join(["normal ", mainvar.long_name])
        avar.units = mainvar.units
        avar.standard_name = mainvar.standard_name
        avar.ancillary_variables = "".join([varname, "_se"])
        avar.comment = "The 1981-2010 monthly normals"
        
        if varname == 'tmin':
            avar.cell_methods = "time: minimum within years time: mean over years"
        elif varname == 'tmax':
            avar.cell_methods = "time: maximum within years time: mean over years"
        else:
            raise Exception("Do not recognize varname %s for determining cell methods" % varname)
        _add_grid_mapping_attr(avar)
                
        avar = ds.createVariable("".join([varname, "_se"]), 'f4', ('time_normals', 'lat', 'lon',),
                                 chunksizes=(12, self.chk_size_y, self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar.long_name = "".join([mainvar.long_name, " kriging standard error"])
        avar.standard_name = 'air_temperature standard_error'
        avar.units = mainvar.units
        avar.comment = "The uncertainty in the 1981-2010 monthly normals"
        _add_grid_mapping_attr(avar)
        
        avar = ds.createVariable("inconsist_tair", 'i4', ('lat', 'lon',),
                                 chunksizes=(self.chk_size_y, self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['i4'])
        avar.long_name = "number of days interpolated tmin >= tmax"
        avar.units = "days"
        avar.comment = "The number of days daily tmin/tmax had to be adjusted due to interpolated tmin being >= interpolated tmax"
        _add_grid_mapping_attr(avar)
        
        ds.sync()
        
    def write_tile_chunk(self, tile_id, varname, days, str_row, str_col, daily_vals, mthly_normals, mthly_normals_se, ninvalid):
        '''
        Writes out a work chunk for a netCDF tile. If the file for netCDF tile
        has not yet been created, it will be created.
        
        Parameters
        ----------
        tile_id : str
            The ID of the tile (eg- 'h10v05')
        varname : str
            The name of the variable being written (eg- 'tmin' or 'tmax')
        days : structured array
            A days array produced by twx.utils.get_days_metadata.
            Specifies the days of the time series that will be written.
        str_row : int
            The starting row of the chunk within the tile
        str_col : int
            The ending row of the chunk within the tile
        daily_vals : ndarray (int16)
            A 3-D int16 array of size N*Y*X where N is the number of days,
            Y is chk_size_y and X is chk_size_x. Represents daily
            interpolated values.
        mthly_normals : ndarray
            A 3-D float32/64 array of size 12*Y*X where 
            Y is chk_size_y and X is chk_size_x. Represents interpolated
            monthly normals.
        mthly_normals_se : ndarray
            A 3-D float32/64 array of size 12*Y*X where 
            Y is chk_size_y and X is chk_size_x. Represents the
            kriging standard error of the monthly normals.
        ninvalid : ndarray
            A 2-D int array of size Y*X where 
            Y is chk_size_y and X is chk_size_x. Represents the
            number of days where Tmax < Tmin had to be corrected.
        '''
    
        ds = self.__open_dataset(tile_id, varname, days)
    
        nrows = self.chk_size_y
        ncols = self.chk_size_x
        
        # Data is already scaled and set to int16 so turn autoscale off
        ds.variables[varname].set_auto_maskandscale(False)
        
        ds.variables[varname][:, str_row:str_row + nrows, str_col:str_col + ncols] = daily_vals
        ds.variables["".join([varname, "_normal"])][:, str_row:str_row + nrows, str_col:str_col + ncols] = mthly_normals   
        ds.variables["".join([varname, "_se"])][:, str_row:str_row + nrows, str_col:str_col + ncols] = mthly_normals_se
        ds.variables['inconsist_tair'][str_row:str_row + nrows, str_col:str_col + ncols] = ninvalid
        
        ds.close()
    
def _add_crs_wgs84_var(ds):
    # FROM CF Conventions: 
    # http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html#grid-mappings-and-projections
    crs = ds.createVariable("crs", 'i2')
    crs.grid_mapping_name = "latitude_longitude"
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563

def _add_grid_mapping_attr(avar):
    
    avar.coordinates = "lat lon"
    avar.grid_mapping = "crs"
    
        
