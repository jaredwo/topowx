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

__all__ = ['Tiler', 'TileWriter', 'TileGridInfo', 'TileMosaic', 'write_ds_mthly']

from datetime import date, datetime, timedelta
from netCDF4 import Dataset, date2num, num2date
from twx.utils import MONTH, get_mth_metadata, StatusCheck, YMD, DATE, YEAR, \
    get_days_metadata_dates, get_days_metadata, set_chunk_cache_params
import netCDF4
import numpy as np
import os
import twx

# factor by which daily outputs are scaled.
# everything is stored as int16
SCALE_FACTOR = np.float32(0.01) 

# long name, units, standard name, missing value,cell method
VAR_ATTRS = {'tmin':("minimum air temperature", "C", "air_temperature",
                     netCDF4.default_fillvals['i2'], "minimum"),
             'tmax':("maximum air temperature", "C", "air_temperature",
                     netCDF4.default_fillvals['i2'], "maximum")}

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
    
    def get_incomplete_tile_nums(self, path_out):
        
        id_to_name = self.tile_ids
        name_to_id = {}
        all_ids = []
        
        for a_id, a_name in id_to_name.items():
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
        ds.institution = "University of Montana"
        ds.source = "TopoWx %s" % twx.__version__
        ds.history = "".join(["Created on: ", datetime.strftime(date.today(), "%Y-%m-%d")]) 
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
            return datetime(mdate.year, mdate.month, mdate.day)
        
        for mth in np.arange(1, 13):
            
            mthNext = mth + 1 if mth != 12 else 1
            yrNext = 1981 if mth != 12 else 1982
            time_means[mth - 1] = date2num(get_mid_date(date(1981, mth, 1), date(yrNext, mthNext, 1)), times.units)
        
            yrNext = 2010 if mth != 12 else 2011
            clim_bnds[mth - 1, :] = date2num([datetime(1981, mth, 1), datetime(yrNext, mthNext, 1)], times.units)
        
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

class TileMosaic():
    """Class to create full CONUS netCDF mosaics from the tiled TopoWx output
    """
    
    def __init__(self, fpath_mask, tile_size_y, tile_size_x, chk_size_y,
                 chk_size_x):
        
        ds_mask = Dataset(fpath_mask)
        self.atiler = Tiler(ds_mask, [], tile_size_y, tile_size_x, chk_size_y,
                            chk_size_x, path_out=None)
        self.tinfo = self.atiler.build_tile_grid_info()
        self.lon = ds_mask.variables['lon'][:]
        self.lat = ds_mask.variables['lat'][:]
            
    def create_dly_ann_mosaics(self, tiles, varname, path_in, path_out,
                               start_yr, end_yr, ds_version_str,
                               chunk_cache_size):
        
        tcols = np.array([np.int(tile[1:3]) for tile in tiles])
        trows = np.array([np.int(tile[4:]) for tile in tiles])
        
        tcols = np.arange(np.min(tcols), np.max(tcols) + 1)
        trows = np.arange(np.min(trows), np.max(trows) + 1)
        
        mosaic_tnames = []
        
        for a_col in tcols:
            
            for a_row in trows:
                
                mosaic_tnames.append("h%02dv%02d" % (a_col, a_row))
        
        def get_rows_cols(a_tname):
            try:
                return self.tinfo.tile_rc[a_tname]
            except KeyError:
                return (-1, -1)
        
        rows_cols = np.array([get_rows_cols(a_tname) for a_tname in mosaic_tnames])
        rows_cols = np.ma.masked_equal(rows_cols, -1)
            
        min_row, min_col = np.ma.min(rows_cols, axis=0)
        max_row, max_col = np.ma.max(rows_cols, axis=0)
        
        lon_mosaic = self.lon[min_col:max_col + self.atiler.tile_size_x]
        lat_mosaic = self.lat[min_row:max_row + self.atiler.tile_size_y]
        
        print ("Mosaic datasets will be of shape: %d * %d in size" % 
               (lat_mosaic.size, lon_mosaic.size))
        
        ds_tile = None
        for a_tname in mosaic_tnames:
            try:
                ds_tile = Dataset(os.path.join(path_in, a_tname,
                                               "".join([a_tname,
                                                        "_", varname, ".nc"])))
                break
            except RuntimeError:
                continue
            
        days_all = get_days_metadata_dates(num2date(ds_tile.variables['time'][:],
                                                    units=ds_tile.variables['time'].units))
        
        start_date = datetime(days_all[DATE][0].year,
                              days_all[DATE][0].month,
                              days_all[DATE][0].day)
        end_date = datetime(days_all[DATE][-1].year,
                              days_all[DATE][-1].month,
                              days_all[DATE][-1].day)
        
        days_all = get_days_metadata(start_date, end_date)
        
        mask_all = np.logical_and(days_all[YEAR] >= start_yr,
                                  days_all[YEAR] <= end_yr)
        
        days_all = days_all[mask_all]
        
        uniq_yrs = np.unique(days_all[YEAR])
        
        yr_ds = []
        yr_masks = [np.nonzero(days_all[YEAR] == yr)[0] for yr in uniq_yrs]
        
        
        for yr, yr_mask in zip(uniq_yrs, yr_masks):
            
            fpath_out = os.path.join(path_out, '%s_%d.nc' % (varname, yr))
            
            ###############################################
            # Create output dataset
            ###############################################
            ds_mosaic = Dataset(fpath_out, 'w')
            
            days = days_all[yr_mask]
            
            # Set global attributes
            title = "".join(["Daily Interpolated Topoclimatic Temperature ",
                             str(days[YMD][0]), "-", str(days[YMD][-1])])
            ds_mosaic.title = title
            ds_mosaic.institution = "Pennsylvania State University"
            ds_mosaic.source = "TopoWx software version %s (https://github.com/jaredwo/topowx)" % twx.__version__
            ds_mosaic.history = "".join(["Created on: ", datetime.strftime(date.today(), "%Y-%m-%d"),
                                         " , ", "dataset version %s" % ds_version_str]) 
            ds_mosaic.references = ("http://dx.doi.org/10.1002/joc.4127 , "
                                    "http://dx.doi.org/10.1002/2014GL062803 , "
                                    "http://dx.doi.org/10.1175/JAMC-D-15-0276.1")
            ds_mosaic.comment = ("The TopoWx ('Topography Weather') gridded "
                                 "dataset contains daily 30-arcsec resolution "
                                 "(~800-m resolution; WGS84) interpolations "
                                 "of minimum and maximum topoclimatic air "
                                 "temperature for the conterminous U.S. "
                                 "Using both DEM-based variables and MODIS "
                                 "land skin temperature as predictors of air "
                                 "temperature, interpolation procedures "
                                 "include moving window regression kriging "
                                 "and geographically weighted regression. To "
                                 "avoid artificial climate trends, all input "
                                 "station data are homogenized using the "
                                 "GHCN/USHCN Pairwise Homogenization "
                                 "Algorithm (http://www.ncdc.noaa.gov/oa/"
                                 "climate/research/ushcn/#phas).")
            ds_mosaic.license = ('Creative Commons Attribution-NonCommercial-ShareAlike 4.0 '
                                 'International License (http://creativecommons.org/licenses/by-nc-sa/4.0/)')
            ds_mosaic.Conventions = "CF-1.6"
            
            # Create dimensions
            dim_lon = ds_mosaic.createDimension('lon', lon_mosaic.size)
            dim_lat = ds_mosaic.createDimension('lat', lat_mosaic.size)
            dim_time = ds_mosaic.createDimension('time', days.size)
            ds_mosaic.createDimension('nv', 2)
            
            # Create corresponding dimension variables
            latitudes = ds_mosaic.createVariable('lat', 'f8', ('lat',))
            latitudes.long_name = "latitude"
            latitudes.units = "degrees_north"
            latitudes.standard_name = "latitude"
            latitudes[:] = lat_mosaic
        
            longitudes = ds_mosaic.createVariable('lon', 'f8', ('lon',))
            longitudes.long_name = "longitude"
            longitudes.units = "degrees_east"
            longitudes.standard_name = "longitude"
            longitudes[:] = lon_mosaic
            
            _add_crs_wgs84_var(ds_mosaic)
            
            times = ds_mosaic.createVariable('time', 'f8', ('time',),
                                             fill_value=False)
            times.long_name = "time"
            times.units = "days since 1948-1-1 0:0:0"
            times.standard_name = "time"
            times.calendar = "standard"
            times.bounds = 'time_bnds'
            
            time_bnds = ds_mosaic.createVariable('time_bnds', 'f8', ('time', 'nv'),
                                                 fill_value=False)
            
            time_nums = date2num(days[DATE], times.units) + 0.5
            time_nums_bnds = np.empty((days.size, 2))
            time_nums_bnds[:, 0] = time_nums - 0.5
            time_nums_bnds[:, 1] = time_nums + 0.5
            
            times[:] = time_nums
            time_bnds[:] = time_nums_bnds
            
            long_name, units, standard_name, fill_value, cell_method = VAR_ATTRS[varname]
            var_tair = ds_mosaic.createVariable(varname, 'i2', ('time', 'lat', 'lon',),
                                                chunksizes=(1, 325, 700),
                                                fill_value=fill_value, zlib=True)
            set_chunk_cache_params(chunk_cache_size, var_tair)
            
            var_tair.long_name = long_name
            var_tair.units = units
            var_tair.standard_name = standard_name
            var_tair.scale_factor = SCALE_FACTOR
            var_tair.cell_methods = "".join(["area: mean ", "time: ", cell_method])
            _add_grid_mapping_attr(var_tair)
            var_tair.set_auto_maskandscale(False)
            
            ds_mosaic.sync()
            
            print "Created dataset: " + fpath_out
            
            yr_ds.append(ds_mosaic)
            
            ###############################################
            ###############################################
        
        x = 0
        
        for a_col, i in zip(tcols, np.arange(tcols.size)):
            
            for a_row, j in zip(trows, np.arange(trows.size)):
                
                try:
                
                    ds_tile = Dataset(os.path.join(path_in, mosaic_tnames[x],
                                                   "".join([mosaic_tnames[x],
                                                            "_", varname, ".nc"])))
                    
                    start_row = j * self.atiler.tile_size_y
                    start_col = i * self.atiler.tile_size_x
                    end_row = start_row + self.atiler.tile_size_y
                    end_col = start_col + self.atiler.tile_size_x
                    
                    print "Tile %s: Reading daily data..." % (mosaic_tnames[x],)
                    var_tair_tile = ds_tile.variables[varname]
                    var_tair_tile.set_auto_maskandscale(False)
                    tair = var_tair_tile[mask_all, :, :]
                    
                    print "Tile %s: Writing daily data..." % (mosaic_tnames[x],)
                    
                    for a_ds, a_mask, yr in zip(yr_ds, yr_masks, uniq_yrs):
                        
                        print "Year: %d" % (yr,)
                        a_ds.variables[varname][:, start_row:end_row, start_col:end_col] = np.take(tair, indices=a_mask, axis=0)
                        a_ds.sync()
                        
                except RuntimeError as e:
                    
                    if e.args[0] == 'No such file or directory':
                        print "Tile %s does not exist. Values for tile will be fill values." % (mosaic_tnames[x],)
                    else:
                        raise
                x += 1
        
        for a_ds in yr_ds:
            a_ds.sync()
            a_ds.close()
        
    def create_normals_mosaic(self, tiles, varname, path_in, fpath_out,
                              ds_version_str):
        
        tcols = np.array([np.int(tile[1:3]) for tile in tiles])
        trows = np.array([np.int(tile[4:]) for tile in tiles])
        
        tcols = np.arange(np.min(tcols), np.max(tcols) + 1)
        trows = np.arange(np.min(trows), np.max(trows) + 1)
        
        mosaic_tnames = []
        
        for a_col in tcols:
            
            for a_row in trows:
                
                mosaic_tnames.append("h%02dv%02d" % (a_col, a_row))
                
        def get_rows_cols(a_tname):
            try:
                return self.tinfo.tile_rc[a_tname]
            except KeyError:
                return (-1, -1)
        
        rows_cols = np.array([get_rows_cols(a_tname) for a_tname in mosaic_tnames])
        rows_cols = np.ma.masked_equal(rows_cols, -1)
            
        min_row, min_col = np.ma.min(rows_cols, axis=0)
        max_row, max_col = np.ma.max(rows_cols, axis=0)
        
        lon_mosaic = self.lon[min_col:max_col + self.atiler.tile_size_x]
        lat_mosaic = self.lat[min_row:max_row + self.atiler.tile_size_y]
        
        ds_mosaic = Dataset(fpath_out, 'w')
        
        # Set global attributes
        ds_mosaic.title = "Interpolated 1981-2010 Monthly Normals for Topoclimatic Temperature"
        ds_mosaic.institution = "Pennsylvania State University"
        ds_mosaic.source = "TopoWx software version %s (https://github.com/jaredwo/topowx)" % twx.__version__
        ds_mosaic.history = "".join(["Created on: ", datetime.strftime(date.today(), "%Y-%m-%d"),
                                     " , ", "dataset version %s" % ds_version_str]) 
        ds_mosaic.references = ("http://dx.doi.org/10.1002/joc.4127 , "
                                "http://dx.doi.org/10.1002/2014GL062803 , "
                                "http://dx.doi.org/10.1175/JAMC-D-15-0276.1")
        ds_mosaic.comment = ("1981-2010 monthly normals for the daily TopoWx product. "
                             "The TopoWx ('Topography Weather') gridded "
                             "dataset contains daily 30-arcsec resolution "
                             "(~800-m resolution; WGS84) interpolations "
                             "of minimum and maximum topoclimatic air "
                             "temperature for the conterminous U.S. "
                             "Using both DEM-based variables and MODIS "
                             "land skin temperature as predictors of air "
                             "temperature, interpolation procedures "
                             "include moving window regression kriging "
                             "and geographically weighted regression. To "
                             "avoid artificial climate trends, all input "
                             "station data are homogenized using the "
                             "GHCN/USHCN Pairwise Homogenization "
                             "Algorithm (http://www.ncdc.noaa.gov/oa/"
                             "climate/research/ushcn/#phas).")
        ds_mosaic.license = ('Creative Commons Attribution-NonCommercial-ShareAlike 4.0 '
                             'International License (http://creativecommons.org/licenses/by-nc-sa/4.0/)')
        ds_mosaic.Conventions = "CF-1.6"
        
        dim_lon = ds_mosaic.createDimension('lon', lon_mosaic.size)
        dim_lat = ds_mosaic.createDimension('lat', lat_mosaic.size)
        dim_time = ds_mosaic.createDimension('time', size=12)
        ds_mosaic.createDimension('nv', 2)
        
        print "Mosaic Dataset is %d * %d" % (lat_mosaic.size, lon_mosaic.size)
            
        # Create corresponding dimension variables
        latitudes = ds_mosaic.createVariable('lat', 'f8', ('lat',))
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lat_mosaic
    
        longitudes = ds_mosaic.createVariable('lon', 'f8', ('lon',))
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lon_mosaic
        
        _add_crs_wgs84_var(ds_mosaic)
        
        time_means = ds_mosaic.createVariable('time', 'f8', ('time',), fill_value=False)
        time_means.long_name = "time"
        time_means.units = "days since 1948-1-1 0:0:0"
        time_means.standard_name = "time"
        time_means.calendar = "standard"
        time_means.climatology = "climatology_bounds"
        time_means.comment = "Time dimension for the 1981-2010 monthly normals"
        
        clim_bnds = ds_mosaic.createVariable('climatology_bounds', 'f8',
                                             ('time', 'nv'), fill_value=False)
        
        def get_mid_date(dateStart, dateEnd):
            mdate = dateStart + ((dateEnd - dateStart) / 2)
            return datetime(mdate.year, mdate.month, mdate.day)
        
        for mth in np.arange(1, 13):
            
            mthNext = mth + 1 if mth != 12 else 1
            yrNext = 1981 if mth != 12 else 1982
            time_means[mth - 1] = date2num(get_mid_date(date(1981, mth, 1), date(yrNext, mthNext, 1)), time_means.units)
        
            yrNext = 2010 if mth != 12 else 2011
            clim_bnds[mth - 1, :] = date2num([datetime(1981, mth, 1), datetime(yrNext, mthNext, 1)], time_means.units)
        
        long_name, units, standard_name, fill_value, cell_method = VAR_ATTRS[varname]
        avar = ds_mosaic.createVariable("".join([varname, "_normal"]), 'i2',
                                        ('time', 'lat', 'lon',), chunksizes=(1, 325, 700),
                                        fill_value=netCDF4.default_fillvals['i2'], zlib=True)
        
        avar.long_name = "".join(["normal ", long_name])
        avar.units = units
        avar.standard_name = standard_name
        avar.ancillary_variables = "".join([varname, "_se"])
        avar.comment = "1981-2010 monthly normals"
        avar.scale_factor = SCALE_FACTOR
        
        if varname == 'tmin':
            avar.cell_methods = "time: minimum within years time: mean over years"
        elif varname == 'tmax':
            avar.cell_methods = "time: maximum within years time: mean over years"
        else:
            raise Exception("Do not recognize varname %s for determining cell methods" % varname)
        _add_grid_mapping_attr(avar)
                
        avar = ds_mosaic.createVariable("".join([varname, "_se"]), 'i2', ('time', 'lat', 'lon',),
                                 chunksizes=(1, 325, 700),
                                 fill_value=netCDF4.default_fillvals['i2'], zlib=True)
        avar.long_name = "".join([long_name, " kriging standard error"])
        avar.standard_name = 'air_temperature standard_error'
        avar.units = units
        avar.comment = "Uncertainty in the 1981-2010 monthly normals"
        avar.scale_factor = SCALE_FACTOR
        _add_grid_mapping_attr(avar)
          
        x = 0
        
        varname_se = "".join([varname, "_se"])
        varname_norm = "".join([varname, "_normal"])
        
        var_se = ds_mosaic.variables[varname_se]
        var_norm = ds_mosaic.variables[varname_norm]
        
        var_norm.set_auto_maskandscale(False)
        var_se.set_auto_maskandscale(False)
        
        for a_col, i in zip(tcols, np.arange(tcols.size)):
            
            for a_row, j in zip(trows, np.arange(trows.size)):
                
                try:
                
                    ds_tile = Dataset(os.path.join(path_in, mosaic_tnames[x],
                                                   "".join([mosaic_tnames[x],
                                                            "_", varname, ".nc"])))
                    
                    print "Tile %s: Writing normals data..." % (mosaic_tnames[x],)
                    
                    start_row = j * self.atiler.tile_size_y
                    start_col = i * self.atiler.tile_size_x
                    end_row = start_row + self.atiler.tile_size_y
                    end_col = start_col + self.atiler.tile_size_x
                    
                    tile_tair = ds_tile.variables[varname_norm][:]
                    tile_tair = np.ma.round(tile_tair, 2) / SCALE_FACTOR
                    tile_tair = np.ma.asarray(tile_tair, dtype=np.int16)
                    tile_tair = np.ma.filled(tile_tair, netCDF4.default_fillvals['i2'])
                    
                    tile_se = ds_tile.variables[varname_se][:]
                    tile_se = np.ma.round(tile_se, 2) / SCALE_FACTOR
                    tile_se = np.ma.asarray(tile_se, dtype=np.int16)
                    tile_se = np.ma.filled(tile_se, netCDF4.default_fillvals['i2'])
                    
                    var_norm[:, start_row:end_row, start_col:end_col] = tile_tair
                    var_se[:, start_row:end_row, start_col:end_col] = tile_se
                    ds_mosaic.sync()
                    
                except RuntimeError:
                    
                    print ("Tile %s does not exist. Values for tile will be fill values."
                           % (mosaic_tnames[x],))
                
                x += 1 
        
        ds_mosaic.sync()        
        ds_mosaic.close()

def _copy_nc_dim(ds_out, adim):

    ds_out.createDimension(adim._name, len(adim))
    
def _copy_nc_attrs(var_out, avar, ignore_attrs=['_FillValue']):
    
    names_attrs = avar.ncattrs()
    
    for aname in names_attrs:
        
        if aname not in ignore_attrs:
            
            var_out.setncattr(aname, avar.getncattr(aname))
    
def _copy_nc_var(ds_out, avar, copy_data=False, ignore_attrs=['_FillValue']):

    var_out = ds_out.createVariable(avar._name, avar.dtype, avar.dimensions)
    
    _copy_nc_attrs(var_out, avar, ignore_attrs)
    
    if copy_data:
        var_out[:] = avar[:]

def _create_ds_mthly(ds_dly, fpath_out_ds_mthly, yr, varname, ds_version_str):
    
    var_tair = ds_dly.variables[varname]
    
    ds_out = Dataset(fpath_out_ds_mthly, 'w')
    
    # Setup dimensions
    _copy_nc_dim(ds_out, ds_dly.dimensions['lon'])
    _copy_nc_dim(ds_out, ds_dly.dimensions['lat'])
    ds_out.createDimension('time', 12)
    _copy_nc_dim(ds_out, ds_dly.dimensions['nv'])
    
    # Setup variables
    _copy_nc_var(ds_out, ds_dly.variables['lon'], copy_data=True)
    _copy_nc_var(ds_out, ds_dly.variables['lat'], copy_data=True)
    _copy_nc_var(ds_out, ds_dly.variables['crs'], copy_data=False)
    
    times = ds_out.createVariable('time', 'f8', ('time',), fill_value=False)
    times.long_name = "time"
    times.units = "days since 1948-1-1 0:0:0"
    times.standard_name = "time"
    times.calendar = "standard"
    times.bounds = 'time_bnds'
    time_bnds = ds_out.createVariable('time_bnds', 'f8', ('time', 'nv'), fill_value=False)
    
    dates_mth = [datetime(yr, mth, 1) for mth in np.arange(1, 13)]
    dates_mth.append(datetime(yr + 1, 1, 1))
    
    def get_mid_date(date_start, date_end):
        mdate = date_start + timedelta(days=(((date_end - date_start).days) / 2.0))
        return datetime(mdate.year, mdate.month, mdate.day)
        
    for i in np.arange(12):
        times[i] = date2num(get_mid_date(dates_mth[i], dates_mth[i + 1]), times.units)
        time_bnds[i, :] = date2num([dates_mth[i], dates_mth[i + 1]], times.units)
    
    # Setup main variable
    mainvar_shp = (12, len(ds_dly.dimensions['lat']), len(ds_dly.dimensions['lon']))
    chunk_shp = (1, 325, 700)
    var_mthly_tair = ds_out.createVariable(varname, 'i2', ('time', 'lat', 'lon'), zlib=True,
                                     chunksizes=chunk_shp, fill_value=var_tair._FillValue)
    _copy_nc_attrs(var_mthly_tair, var_tair, ignore_attrs=['_FillValue', 'cell_methods', '_Storage',
                                                           '_ChunkSizes', '_DeflateLevel', '_Shuffle',
                                                           '_Endianness'])
    
    cell_methods = {'tmin':'time: minimum within days time: mean over days area: mean',
                    'tmax':'time: maximum within days time: mean over days area: mean'}
    var_mthly_tair.cell_methods = cell_methods[varname]
    
    # Global Attributes
    ds_out.title = "Monthly Interpolated Topoclimatic Temperature for %d" % yr
    ds_out.institution = "Pennsylvania State University"
    ds_out.source = "TopoWx software version %s (https://github.com/jaredwo/topowx)" % twx.__version__
    ds_out.history = "".join(["Created on: ", datetime.strftime(date.today(), "%Y-%m-%d"),
                              " , ", "dataset version %s" % ds_version_str])  
    ds_out.references = ("http://dx.doi.org/10.1002/joc.4127 , "
                         "http://dx.doi.org/10.1002/2014GL062803 , "
                         "http://dx.doi.org/10.1175/JAMC-D-15-0276.1")
    ds_out.comment = ("Monthly aggregation of the daily TopoWx product."
                      "The TopoWx ('Topography Weather') gridded "
                      "dataset contains daily 30-arcsec resolution "
                      "(~800-m resolution; WGS84) interpolations "
                      "of minimum and maximum topoclimatic air "
                      "temperature for the conterminous U.S. "
                      "Using both DEM-based variables and MODIS "
                      "land skin temperature as predictors of air "
                      "temperature, interpolation procedures "
                      "include moving window regression kriging "
                      "and geographically weighted regression. To "
                      "avoid artificial climate trends, all input "
                      "station data are homogenized using the "
                      "GHCN/USHCN Pairwise Homogenization "
                      "Algorithm (http://www.ncdc.noaa.gov/oa/"
                      "climate/research/ushcn/#phas).")

    ds_out.license = ('Creative Commons Attribution-NonCommercial-ShareAlike 4.0 '
                      'International License (http://creativecommons.org/licenses/by-nc-sa/4.0/)')    

    ds_out.Conventions = "CF-1.6"
    
    ds_out.sync()
    
    return ds_out

class _TairAggregate():
    '''
    A class for aggregating daily data to monthly and annual
    '''
    
    def __init__(self, days):
        '''
        
        Parameters
        ----------
        days : structured numpy array
            a structured array of date information for the time period of interest
            produced from  util_dates.get_days_metadata_* methods
        '''
    
        uYrs = np.unique(days[YEAR])
        uMths = np.unique(days[MONTH])
        
        self.yr_mths_masks = []
        
        for aYr in uYrs:
            
            for aMth in uMths:
                
                self.yr_mths_masks.append(np.nonzero(np.logical_and(days[YEAR] == aYr,
                                                                    days[MONTH] == aMth))[0])
                
        self.days = days
        self.yr_mths = get_mth_metadata(uYrs[0], uYrs[-1])
        self.yr_mths = self.yr_mths[np.in1d(self.yr_mths[MONTH], uMths, False)]
        
        self.yr_masks = []
        
        for aYr in uYrs:
            
            self.yr_masks.append(np.nonzero(self.yr_mths[YEAR] == aYr)[0])
        
        self.u_yrs = uYrs
    
    
    def daily_to_mthly(self, tair):
        '''Aggregate daily data to monthly
        
        Parameters
        ----------
        tair : numpy array
            a numpy array or masked array. Can be of any shape, but first axis
            must be the time dimension
        '''
        
        tair_mthly = np.ma.array([np.ma.mean(np.ma.take(tair, aMask, axis=0),
                                             axis=0, dtype=np.float)
                                  for aMask in self.yr_mths_masks])
        
        return tair_mthly
    
    def daily_to_ann(self, tair):
        '''Aggregate daily data to annual
        
        Parameters
        ----------
        tair : numpy array
            a numpy array or masked array. Can be of any shape, but first axis
            must be the time dimension
        '''
        
        tair_mthly = self.daily_to_mthly(tair)
        tair_ann = self.mthly_to_ann(tair_mthly)
        
        return tair_ann
    
    def mthly_to_ann(self, tair_mthly):
        '''Aggregate daily data to monthly
        
        Parameters
        ----------
        tair_mthly : numpy array
            a numpy array or masked array. Can be of any shape, but first axis
            must be the time dimension
        '''
        
        tair_ann = np.ma.masked_array([np.ma.mean(np.ma.take(tair_mthly,
                                                             aMask, axis=0),
                                                  axis=0, dtype=np.float)
                                       for aMask in self.yr_masks])
        
        return tair_ann

  
def write_ds_mthly(ds_dly, fpath_out, varname, yr, ds_version_str):
    """Write out monthly version of the TopoWx netCDF mosaics
    """
        
    ds_out = _create_ds_mthly(ds_dly, fpath_out, yr, varname, ds_version_str)
    
    dates = num2date(ds_dly.variables['time'][:],
                     units=ds_dly.variables['time'].units)
    days = get_days_metadata_dates(dates)
    yr = days[YEAR][0]
    tagg = _TairAggregate(days)
     
    var_tair = ds_dly.variables[varname]
     
    chking = var_tair.chunking() 
     
    nrow = var_tair.shape[1]
    ncol = var_tair.shape[2]
    rsteps = np.arange(nrow, step=chking[1])
    csteps = np.arange(ncol, step=chking[2])
     
    mthly_tair = np.ma.zeros((12, nrow, ncol))
     
    schk = StatusCheck(rsteps.size * csteps.size, 5)
     
    # Loop through chunks
    for r in rsteps:
         
        for c in csteps:
             
            endr = r + chking[1]
            endc = c + chking[2]
             
            # Just in case nrow/ncols is not 
            # evenly divided by chunksize
            if endr > nrow:
                endr = nrow
            if endc > ncol:
                endc = ncol
     
            a_dly_tair = var_tair[:, r:endr, c:endc]
            a_mthly_tair = tagg.daily_to_mthly(a_dly_tair)
            mthly_tair[:, r:endr, c:endc] = a_mthly_tair
             
            schk.increment()
    
    print "Writing %s monthly data for %d..." % (varname, yr)
    mthly_tair = np.ma.round(mthly_tair, 2)
    ds_out.variables[varname][:] = mthly_tair
    ds_out.sync()
    ds_out.close()
