'''
Created on Oct 15, 2012

@author: jared.oyler
'''
import numpy as np
from netCDF4 import Dataset,date2num
import os
import datetime
from datetime import date
from twx.utils.util_dates import YMD,DATE
import netCDF4
from twx.utils.util_ncdf import to_geotiff,to_geotiffa
from twx.db.create_db_all_stations import dbDataset
import matplotlib.pyplot as plt

SCALE_FACTOR = np.float32(0.01) #factor by which interp outputs are scaled. everything is stored as int16

#long name, units, standard name, missing value,cell method
VAR_ATTRS = {'tmin':("minimum air temperature","C","air_temperature",netCDF4.default_fillvals['i2'],"minimum"),
             'tmax':("maximum air temperature","C","air_temperature",netCDF4.default_fillvals['i2'],"maximum")}

#gdal_polygonize.py tiles.tif -f "ESRI Shapefile" tiles.shp tiles num

class tiler():
    '''
    Breaks up a defined grid into tiles and separate work chunks for processing
    '''
    
    def __init__(self,ds_mask,ds_attr_ls,tile_size_y,tile_size_x,chk_size_y,chk_size_x,process_tiles=None):
        '''
        Constructor
        
        @param ds_mask:
        @param ds_attr_dict:
        @param tile_size_y:
        @param tile_size_x:
        @param chk_size_y:
        @param chk_size_x:
        @param process_tiles:
        '''
        
        self.mask = np.array(ds_mask.variables['mask'][:],dtype=np.bool)
        self.lons = ds_mask.variables['lon'][:]
        self.lats = ds_mask.variables['lat'][:]
        self.nrows = self.lats.size
        self.ncols = self.lons.size
        self.tile_size_y = tile_size_y
        self.tile_size_x = tile_size_x
        self.chk_size_y = chk_size_y
        self.chk_size_x = chk_size_x
        self.process_tiles = process_tiles
        
        self.attrs = []
        for varname,ds in ds_attr_ls:
        
            attr = ds.variables[varname]
            attr.set_auto_maskandscale(False)
            
            self.attrs.append(attr)
            
        self.tile_ids,self.tile_rc = self.build_tile_dicts(self.nrows, self.ncols, tile_size_y, tile_size_x, self.mask)
        
        self.wrk_chk = np.zeros((5+len(self.attrs),chk_size_y,chk_size_x))*np.nan
        
        self.set_tile_chks()
    
    
    def set_tile_chks(self):
    
        k = 0
        self.tile_chks = []
        self.ntiles = 0
        
        for i in np.arange(0,self.nrows,self.tile_size_y):
            
            for j in np.arange(0,self.ncols,self.tile_size_x):
                
                msk_tile = self.mask[i:i+self.tile_size_y,j:j+self.tile_size_x]
                
                if msk_tile[msk_tile].size > 0:
                    
                    process_tile = False
                    try:
                        if k in self.process_tiles:
                            process_tile = True
                    except TypeError:
                        process_tile = True
                    
                    if process_tile:
                        
                        for y in np.arange(0,self.tile_size_y,self.chk_size_y):
                            
                            for x in np.arange(0,self.tile_size_x,self.chk_size_x):
                                
                                self.tile_chks.append((k,i,j,y,x))
                    
                        self.ntiles+=1
                         
                    k+=1
            
            self.iter_x = 0
            self.ntile_chks = len(self.tile_chks)
            
    def next(self):
        
        if self.iter_x == self.ntile_chks:
            
            raise StopIteration()
        
        else:
            
            k,i,j,y,x = self.tile_chks[self.iter_x]
            
            msk_tile = self.mask[i:i+self.tile_size_y,j:j+self.tile_size_x]
            
            attr_tiles = []
            #x = 0
            for attr in self.attrs:
                attr_tiles.append(attr[i:i+self.tile_size_y,j:j+self.tile_size_x])
                
#                if x >= 3 and x <= 14:
#                    plt.imshow(attr_tiles[-1])
#                    plt.colorbar()
#                    plt.show()
#                x+=1
                
            #elev_tile = elev[i:i+params[P_TILESIZE_Y],j:j+params[P_TILESIZE_X]]
            llgrid = np.meshgrid(self.lons[j:j+self.tile_size_x],self.lats[i:i+self.tile_size_y])
            
            rcgrid = np.mgrid[y:y+self.chk_size_y,x:x+self.chk_size_x]

            self.wrk_chk[0,:,:] = rcgrid[0,:,:] #row
            self.wrk_chk[1,:,:] = rcgrid[1,:,:] #col
            self.wrk_chk[2,:,:] = msk_tile[y:y+self.chk_size_y,x:x+self.chk_size_x] #mask
            self.wrk_chk[3,:,:] = llgrid[1][y:y+self.chk_size_y,x:x+self.chk_size_x] #lat
            self.wrk_chk[4,:,:] = llgrid[0][y:y+self.chk_size_y,x:x+self.chk_size_x] #lon
            
            for z in np.arange(len(attr_tiles)):
                
                self.wrk_chk[5+z,:,:] = attr_tiles[z][y:y+self.chk_size_y,x:x+self.chk_size_x] #elev
            
            self.iter_x+=1
            
            return k,self.wrk_chk

    def build_tile_dicts(self,nrows,ncols,tile_size_y,tile_size_x,mask):
        '''
        Builds dictionaries of tile ids and tile start row,cols 
        
        :param nrows: overall # of rows
        :param ncols: overall # of cols
        :param tile_size_y: size of a tile in y direction
        :param tile_size_x: size of a tile in x direction
        :param mask: a mask (1=process, 0=do not process) of study area
        '''
        
        #Build dict of tile ids
        tile_ids = {}
        tile_info = {}
        x = 0
        cnt_y = 0
        cnt_x = 0
        for i in np.arange(0,nrows,tile_size_y):
            
            cnt_x = 0
            for j in np.arange(0,ncols,tile_size_x):
                
                msk_tile = mask[i:i+tile_size_y,j:j+tile_size_x]
                
                if msk_tile[msk_tile].size > 0:
                    
                    atileid = "".join(["h%02d"%(cnt_x,),"v%02d"%(cnt_y,)])
                    tile_ids[x] = atileid
                    tile_info[atileid] = (i,j)
                    x+=1
                
                cnt_x+=1
            
            cnt_y+=1
            
        return tile_ids, tile_info

    def build_tile_grid_info(self):
        
        return tile_grid_info(self.tile_ids, self.tile_rc, self.ntiles, self.lons, self.lats, 
                              self.tile_size_y, self.tile_size_x, self.chk_size_y, self.chk_size_x)
                
class tile_grid_info():           
        
    def __init__(self,tile_ids, tile_rc, ntiles, lons, lats,tile_size_y,tile_size_x,chk_size_y,chk_size_x):
        
        self.tile_ids = tile_ids
        self.tile_rc = tile_rc
        self.ntiles = ntiles
        self.lons = lons
        self.lats = lats
        self.tile_size_y = tile_size_y
        self.tile_size_x = tile_size_x
        self.chk_size_y = chk_size_y
        self.chk_size_x = chk_size_x
        
        self.chks_per_tile = (tile_size_x/chk_size_x)*(tile_size_y/chk_size_y)
        self.nchks = self.chks_per_tile*ntiles
        
    def get_tile_id(self,tile_num):
        return self.tile_ids[tile_num]
    

class tile_writer():
    
    def __init__(self,tile_grid_info,path_out):
        
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
        
    def open_dataset(self,tile_id,varname,days):
    
        fpath = "".join([self.path_out,tile_id,"/",tile_id,"_",varname,".nc"])
    
        try:
            
            ds = Dataset(fpath,'r+')
            
        except RuntimeError:
            
            if not os.path.exists("".join([self.path_out,tile_id])):
                os.mkdir("".join([self.path_out,tile_id]))
            ds = self.create_ncdf(fpath,tile_id,varname,days)
        
        return ds
    
    
    def create_ncdf(self,fpath,tile_id,varname,days):
        
        ds = Dataset(fpath,'w')
         
        #Set global attributes
        title = "".join(["Daily Interpolated Meteorological Data ",str(days[YMD][0]),"-",str(days[YMD][-1])])
        ds.title = title
        ds.institution = "University of Montana Numerical Terradynamics Simulation Group"
#        ds.institution = " | ".join(["University of Montana Numerical Terradynamics Simulation Group",
#                                    "NASA Ames Ecological Forecasting Lab"])
        ds.source = "TopoWx v1.0.0"
        ds.history = "".join(["Created on: ",datetime.datetime.strftime(date.today(),"%Y-%m-%d")]) 
        ds.references = "http://www.ntsg.umt.edu/project/TopoWx"
        ds.comment = "30-arcsec spatial resolution, daily timestep"
        ds.Conventions = "CF-1.6"
        
        str_row,str_col = self.tile_rc[tile_id]
        lons = self.lons[str_col:str_col+self.tile_size_x]
        lats = self.lats[str_row:str_row+self.tile_size_y]
        
        #Create 3-dimensions
        ds.createDimension('time',days.size)
        ds.createDimension('lat',lats.size)
        ds.createDimension('lon',lons.size)
        ds.createDimension('nv',2)
        ds.createDimension('time_normals',12)
        
        min_date = days[DATE][0]
        
        #Create dimension variables and fill with values
        times = ds.createVariable('time','f8',('time',),fill_value=False)
        times.long_name = "time"
        times.units = "".join(["days since ",str(min_date.year),"-",str(min_date.month),"-",str(min_date.day)," 0:0:0"])
        times.standard_name = "time"
        times.calendar = "standard"
        times.bounds = 'time_bnds'
        
        time_bnds = ds.createVariable('time_bnds','f8',('time','nv'),fill_value=False)
        
        time_nums = date2num(days[DATE],times.units) + 0.5
        time_nums_bnds = np.empty((days.size,2))
        time_nums_bnds[:,0] = time_nums - 0.5
        time_nums_bnds[:,1] = time_nums + 0.5
        
        times[:] = time_nums
        time_bnds[:] = time_nums_bnds

        time_means = ds.createVariable('time_normals','f8',('time_normals',),fill_value=False)
        time_means.long_name = times.long_name
        time_means.units = times.units
        time_means.standard_name = times.standard_name
        time_means.calendar = times.calendar
        time_means.climatology = "climatology_bounds"
        time_means.comment = "Time dimension for the 1981-2010 monthly normals"
        
        clim_bnds = ds.createVariable('climatology_bounds','f8',('time_normals','nv'),fill_value=False)
        
        def get_mid_date(dateStart,dateEnd):
            mdate = dateStart + ((dateEnd - dateStart)/2)
            return datetime.datetime(mdate.year,mdate.month,mdate.day)
        
        for mth in np.arange(1,13):
            
            mthNext = mth + 1 if mth != 12 else 1
            yrNext = 1981 if mth !=12 else 1982
            time_means[mth-1] = date2num(get_mid_date(date(1981,mth,1), date(yrNext,mthNext,1)),times.units)
        
            yrNext = 2010 if mth !=12 else 2011
            clim_bnds[mth-1,:] = date2num([datetime.datetime(1981,mth,1),datetime.datetime(yrNext,mthNext,1)],times.units)
        
        latitudes = ds.createVariable('lat','f8',('lat',),fill_value=False)
        latitudes.long_name = "latitude"
        latitudes.units = "degrees_north"
        latitudes.standard_name = "latitude"
        latitudes[:] = lats
    
        longitudes = ds.createVariable('lon','f8',('lon',),fill_value=False)
        longitudes.long_name = "longitude"
        longitudes.units = "degrees_east"
        longitudes.standard_name = "longitude"
        longitudes[:] = lons
        
        add_crs_wgs84_var(ds)
        self.add_dim_vars(ds, varname, days)
        
        ds.sync()
        
        return ds
        
    def add_dim_vars(self,ds,varname,days):
        
        long_name,units,standard_name,fill_value,cell_method = VAR_ATTRS[varname]
        mainvar = ds.createVariable(varname,'i2',('time','lat','lon',),
                                    chunksizes=(days[DATE].size,self.chk_size_y,self.chk_size_x),
                                    fill_value=fill_value)
        mainvar.long_name = long_name
        mainvar.units = units
        mainvar.standard_name = standard_name
        mainvar.scale_factor = SCALE_FACTOR
        mainvar.cell_methods = "".join(["area: mean ","time: ",cell_method])
        add_grid_mapping_attr(mainvar)
        
        avar = ds.createVariable("".join([varname,"_normal"]),'f4',('time_normals','lat','lon',),
                                 chunksizes=(12,self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar.long_name = "".join(["normal ",mainvar.long_name])
        avar.units = mainvar.units
        avar.standard_name = mainvar.standard_name#"".join(["mean_",mainvar.standard_name])
        avar.ancillary_variables = "".join([varname,"_se"])
        avar.comment = "The 1981-2010 monthly normals"
        
        if varname == 'tmin':
            avar.cell_methods="time: minimum within years time: mean over years"
        elif varname == 'tmax':
            avar.cell_methods="time: maximum within years time: mean over years"
        else:
            raise Exception("Do not recognize varname %s for determining cell methods"%varname)
        add_grid_mapping_attr(avar)
        
#        avar = ds.createVariable("".join([varname,"_cil"]),'f4',('lat','lon',),
#                                 chunksizes=(self.chk_size_y,self.chk_size_x),
#                                 fill_value=netCDF4.default_fillvals['f4'])
#        avar.long_name = "".join(["lower 95% confidence interval ",mainvar.long_name])
#        avar.units = mainvar.units
#        #avar.standard_name = "".join(["lower_95%_confidence_interval_",mainvar.standard_name])
#        add_grid_mapping_attr(avar)
#        
#        avar = ds.createVariable("".join([varname,"_ciu"]),'f4',('lat','lon',),
#                                 chunksizes=(self.chk_size_y,self.chk_size_x),
#                                 fill_value=netCDF4.default_fillvals['f4'])
#        avar.long_name = "".join(["upper 95% confidence interval ",mainvar.long_name])
#        avar.units = mainvar.units
#        #avar.standard_name = "".join(["upper_95%_confidence_interval_",mainvar.standard_name])
#        add_grid_mapping_attr(avar)
        
        avar = ds.createVariable("".join([varname,"_se"]),'f4',('time_normals','lat','lon',),
                                 chunksizes=(12,self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar.long_name = "".join([mainvar.long_name, " kriging standard error"])
        avar.standard_name = 'air_temperature standard_error'
        avar.units = mainvar.units
        avar.comment = "The uncertainty in the 1981-2010 monthly normals"
        #avar.standard_name = "".join(["kriging_standard_error_",mainvar.standard_name])
        add_grid_mapping_attr(avar)
        
        avar = ds.createVariable("inconsist_tair",'i4',('lat','lon',),
                                 chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['i4'])
        avar.long_name = "number of days interpolated tmin >= tmax"
        avar.units = "days"
        avar.comment = "The number of days daily tmin/tmax had to be adjusted due to interpolated tmin being >= interpolated tmax"
        add_grid_mapping_attr(avar)
        
        ds.sync()
        
    def write_rslts(self,tile_id,varname,days,str_row,str_col,*args):
    
        ds = self.open_dataset(tile_id, varname, days)
    
        nrows = self.chk_size_y
        ncols = self.chk_size_x
    
        rslt,rslt_norm,rslt_se,rslt_ninvalid = args
    
        #Data is already scaled and set to int16 so turn autoscale off
        ds.variables[varname].set_auto_maskandscale(False)
        
        ds.variables[varname][:,str_row:str_row+nrows,str_col:str_col+ncols] = rslt
        ds.variables["".join([varname,"_normal"])][:,str_row:str_row+nrows,str_col:str_col+ncols] = rslt_norm   
#        ds.variables["".join([varname,"_cil"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_cil 
#        ds.variables["".join([varname,"_ciu"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_ciu
        ds.variables["".join([varname,"_se"])][:,str_row:str_row+nrows,str_col:str_col+ncols] = rslt_se
        ds.variables['inconsist_tair'][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_ninvalid
        

        ds.close()
    

class tile_writer_gdd(tile_writer):
    

    def create_ncdf(self,fpath,tile_id,varname,days):
        
        ds = dbDataset(fpath,'w')
        ds.db_create_global_attributes("Wheat Growing Degree Day Norm and Anomalies")
        
        str_row,str_col = self.tile_rc[tile_id]
        lons = self.lons[str_col:str_col+self.tile_size_x]
        lats = self.lats[str_row:str_row+self.tile_size_y]
        
        ds.db_create_lonlat_dimvar(lons, lats)
        
        avar = ds.createVariable("gdd_norm",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar = ds.createVariable("gdd_anom_2004",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar = ds.createVariable("gdd_anom_2007",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        avar = ds.createVariable("gdd_anom_2011",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        ds.sync()
        
        return ds
                
    def write_rslts(self,tile_id,varname,days,str_row,str_col,*args):
    
        ds = self.open_dataset(tile_id, varname, days)
    
        nrows = self.chk_size_y
        ncols = self.chk_size_x
    
        gdd_norm,gdd_2004_anomly,gdd_2007_anomly,gdd_2011_anomly = args
        
        ds.variables["gdd_norm"][str_row:str_row+nrows,str_col:str_col+ncols] = gdd_norm
        ds.variables["gdd_anom_2004"][str_row:str_row+nrows,str_col:str_col+ncols] = gdd_2004_anomly
        ds.variables["gdd_anom_2007"][str_row:str_row+nrows,str_col:str_col+ncols] = gdd_2007_anomly
        ds.variables["gdd_anom_2011"][str_row:str_row+nrows,str_col:str_col+ncols] = gdd_2011_anomly
        
        ds.close()
        
class tile_writer_metric(tile_writer):
    
    def __init__(self,tile_grid_info,path_out,ds_title,ds_varnames):
        
        tile_writer.__init__(self,tile_grid_info, path_out)
        self.ds_varnames = ds_varnames
        self.ds_title = ds_title

    def create_ncdf(self,fpath,tile_id,varname,days):
        
        ds = dbDataset(fpath,'w')
        ds.db_create_global_attributes(self.ds_title)
        
        str_row,str_col = self.tile_rc[tile_id]
        lons = self.lons[str_col:str_col+self.tile_size_x]
        lats = self.lats[str_row:str_row+self.tile_size_y]
        
        ds.db_create_lonlat_dimvar(lons, lats)
        
        for varname in self.ds_varnames:
            
            avar = ds.createVariable(varname,'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                     fill_value=netCDF4.default_fillvals['f4'])
        ds.sync()
        
        return ds
                
    def write_rslts(self,tile_id,varname,days,str_row,str_col,*args):
    
        ds = self.open_dataset(tile_id, varname, days)
    
        nrows = self.chk_size_y
        ncols = self.chk_size_x
        
        for x in np.arange(len(self.ds_varnames)):
            
            ds.variables[self.ds_varnames[x]][str_row:str_row+nrows,str_col:str_col+ncols] = args[x]
        
        ds.close()
        
class tile_writer_tairdif(tile_writer):
    

    def create_ncdf(self,fpath,tile_id,varname,days):
        
        ds = dbDataset(fpath,'w')
        ds.db_create_global_attributes("Mean 2000-2009 Tair - Mean 1950-1959 Tair")
        
        str_row,str_col = self.tile_rc[tile_id]
        lons = self.lons[str_col:str_col+self.tile_size_x]
        lats = self.lats[str_row:str_row+self.tile_size_y]
        
        ds.db_create_lonlat_dimvar(lons, lats)
        
        avar = ds.createVariable("tmin",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        
        avar = ds.createVariable("tmax",'f4',('lat','lon',),chunksizes=(self.chk_size_y,self.chk_size_x),
                                 fill_value=netCDF4.default_fillvals['f4'])
        ds.sync()
        
        return ds
                
    def write_rslts(self,tile_id,varname,days,str_row,str_col,*args):
    
        ds = self.open_dataset(tile_id, varname, days)
    
        nrows = self.chk_size_y
        ncols = self.chk_size_x
    
        tair_dif,tair_var = args
        
        ds.variables[tair_var][str_row:str_row+nrows,str_col:str_col+ncols] = tair_dif
        
        ds.close()

class tile_reader():
    
    def __init__(self,tile_grid_info,path_out):
        
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
        
    def open_dataset(self,tile_id,varname):
    
        fpath = "".join([self.path_out,tile_id,"/",tile_id,"_",varname,".nc"])
        return Dataset(fpath)


class tile_delete():
    
    def __init__(self,tile_path,varname):
        
        self.tile_path = tile_path
        self.varname = varname
        
    def delete_all(self):
    
        tile_names = os.listdir(self.tile_path)
        
        for tile_name in tile_names:
            
            fpath = "".join([self.tile_path,tile_name,"/",tile_name,"_",self.varname,".nc"])
            rm_cmd = " ".join(["rm",fpath])
            print rm_cmd
            os.system(rm_cmd)

class tile_extract():
    
    def __init__(self,tile_path,path_out,varname,ds_varname):
        
        self.tile_path = tile_path
        self.path_out = path_out
        self.varname = varname
        self.ds_varname = ds_varname
        
    def extract_all(self):
    
        tile_names = os.listdir(self.tile_path)
        
        for tile_name in tile_names:
            
            fpath = "".join([self.tile_path,tile_name,"/",tile_name,"_",self.varname,".nc"])
            ds = Dataset(fpath)
            to_geotiff(ds, self.ds_varname,"".join([self.path_out,tile_name,"_",self.ds_varname,".tif"]))
            print "".join(["extracted ",tile_name]) 

    def mosaic(self):
        os.chdir(self.path_out)
        files = np.array(os.listdir(self.path_out))
        files = files[np.char.find(files, self.ds_varname) != -1]
        
                
        cmd = ["".join(["gdalwarp -dstnodata -9999"])]
        cmd.extend(files)
        cmd.append("".join(["mosaic_",self.ds_varname,".tif"]))
        
        print "running mosaic cmd..."+" ".join(cmd)
        os.system(" ".join(cmd))
        
class tile_extract_predictor():
    
    def __init__(self,tile_path,path_out,varname,fpath_predictor,predictor_varname):
        
        self.tile_path = tile_path
        self.path_out = path_out
        self.varname = varname
        self.predictor_varname = predictor_varname
        self.fpath_predictor = fpath_predictor
        
    def extract_all(self):
    
        tile_names = os.listdir(self.tile_path)
        
        dsPred = Dataset(self.fpath_predictor)
        lon = np.round(dsPred.variables['lon'][:],5)
        lat = np.round(dsPred.variables['lat'][:],5)
        nodata = dsPred.variables[self.predictor_varname].missing_value
        
        for tile_name in tile_names:
            
            fpath = "".join([self.tile_path,tile_name,"/",tile_name,"_",self.varname,".nc"])
            ds = Dataset(fpath)
            
            lon1 = np.round(ds.variables['lon'][0],5)
            lon2 = np.round(ds.variables['lon'][-1],5)
            lat1 = np.round(ds.variables['lat'][0],5)
            lat2 = np.round(ds.variables['lat'][-1],5)
            
            maskLon = np.logical_and(lon>=lon1,lon<=lon2)
            maskLat = np.logical_and(lat<=lat1,lat>=lat2)
            
            if np.sum(maskLon) != len(ds.variables['lon']) or np.sum(maskLat) != len(ds.variables['lat']):
                raise Exception("Output grid does not match input tile grid "+tile_name)
            
            aout = dsPred.variables[self.predictor_varname][maskLat,maskLon]
            
            to_geotiffa(ds, aout, "".join([self.path_out,tile_name,"_",self.predictor_varname,".tif"]), nodata)
            
            print "".join(["extracted ",tile_name]) 

    def mosaic(self):
        os.chdir(self.path_out)
        files = np.array(os.listdir(self.path_out))
        files = files[np.char.find(files, self.predictor_varname) != -1]
        
                
        cmd = ["".join(["gdalwarp -dstnodata -9999"])]
        cmd.extend(files)
        cmd.append("".join(["mosaic_",self.predictor_varname,".tif"]))
        
        print "running mosaic cmd..."+" ".join(cmd)
        os.system(" ".join(cmd))

def add_crs_wgs84_var(ds):
    #FROM CF Conventions: 
    #http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html#grid-mappings-and-projections
    crs = ds.createVariable("crs",'i2')
    crs.grid_mapping_name = "latitude_longitude"
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = 6378137.0
    crs.inverse_flattening = 298.257223563

def add_grid_mapping_attr(avar):
    
    avar.coordinates = "lat lon"
    avar.grid_mapping = "crs"
    
        