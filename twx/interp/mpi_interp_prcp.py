'''
A MPI driver for interpolating prcp to a specified grid using interp.interp_prcp

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_infill,STN_ID,DATE,LON,LAT,YMD
import interp.interp_prcp as ip
from interp.station_select import station_select
from utils.status_check import status_check
from netCDF4 import Dataset,date2num
import netCDF4
import datetime
from datetime import date
import os
from collections import deque

TAG_DOWORK = 1
TAG_STOPWORK = 1000
TAG_REQUEST_WRITE = 3
TAG_WRITE_PERMIT = 4
TAG_DONE_WRITE = 5

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_MASK = 'P_PATH_MASK'
P_PATH_DEM = 'P_PATH_DEM'
P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'

P_MINSTNS = 'P_MINSTNS'
P_TILESIZE_X = 'P_TILESIZE_X'
P_TILESIZE_Y = 'P_TILESIZE_Y'
P_CHCKSIZE_X = 'P_CHCKSIZE_X'
P_CHCKSIZE_Y = 'P_CHCKSIZE_Y'
P_TILE_INFO = 'P_TILE_INFO'
P_TILES_PROCESS = 'P_TILES_PROCESS'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NNGH_TMIN = 'P_NNGH_TMIN'
P_NNGH_TMAX = 'P_NNGH_TMAX'
P_NCDF_MODE = 'P_NCDF_MODE'

P_SIGMA = 'P_SIGMA'
P_DF = 'P_DF'

SCALE_FACTOR = np.float32(0.01)

#long name, units, standard name, missing value
VAR_ATTRS = {'tmin':("minimum air temperature","C","minimum_air_temperature",netCDF4.default_fillvals['i2']),
             'tmax':("maximum air temperature","C","maximum_air_temperature",netCDF4.default_fillvals['i2']),
             'prcp':("precipitation amount","cm","precipitation_amount",netCDF4.default_fillvals['i2'])}

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def proc_work(params,rank):

    status = MPI.Status()
    stn_da = station_data_infill(params[P_PATH_DB], 'prcp')
    days = stn_da.days
    
    stn_slct = station_select(stn_da.stns,params[P_MINSTNS],params[P_MINSTNS]+10)
    
    modC = ip.modeler_clib_prcp()   
    interper = ip.interp_prcp(modC)
    ainput = ip.prcp_input()
    aoutput = ip.prcp_output()
    ainput.sigma = params[P_SIGMA]
    ainput.df = params[P_DF]
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    tile_ids, tile_info, ntiles, lons, lats = bcast_msg
    params[LON] = lons
    params[LAT] = lats
    params[P_TILE_INFO] = tile_info
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    wrk_chk = np.zeros((6,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    rslt_prcp = np.empty((stn_da.days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
    rslt_prcp.fill(netCDF4.default_fillvals['i2'])
    rslt_prcp_mean = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_prcp_cil = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_prcp_ciu = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_prcp_cir = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    
    tile_num = np.zeros(1,dtype=np.int32)
    
    while 1:
    
        MPI.COMM_WORLD.Recv([wrk_chk,MPI.DOUBLE],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            tile_num[0] = status.tag
            tile_id = tile_ids[status.tag]
            
            str_row = np.int(wrk_chk[0,0,0])
            str_col = np.int(wrk_chk[1,0,0])
            
            for r in np.arange(params[P_CHCKSIZE_Y]):
            
                for c in np.arange(params[P_CHCKSIZE_X]):
    
                    if wrk_chk[2,r,c]:
                        
                        row,col,mask,lat,lon,elev = wrk_chk[:,r,c]
                        
                        stns,wgts,rad = stn_slct.get_interp_stns(lat, lon)
    
                        stn_obs = stn_da.load_obs(stns[STN_ID])
                                           
                        ainput.set_pt(lon, lat, elev)
                        ainput.stns = stns
                        ainput.stn_obs = stn_obs
                        ainput.stn_wgts = wgts
                        
                        interper.model_prcp(ainput, aoutput)
                        
                        rslt_prcp[:,r,c] = np.round(aoutput.prcp,2)/SCALE_FACTOR
                        rslt_prcp_mean[r,c] = aoutput.mean                   
                        rslt_prcp_cil[r,c] = aoutput.ci[0]         
                        rslt_prcp_ciu[r,c] = aoutput.ci[1]
                        rslt_prcp_cir[r,c] = ((aoutput.ci[1] - aoutput.ci[0])/aoutput.mean)*100
                        
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_REQUEST_WRITE) 
            MPI.COMM_WORLD.Recv([tile_num,MPI.INT],source=RANK_WRITE,tag=TAG_WRITE_PERMIT)

            ##do write
            write_rslts(params, tile_id, 'prcp', days, str_row, str_col,
                        rslt_prcp, rslt_prcp_mean, rslt_prcp_cil, rslt_prcp_ciu, rslt_prcp_cir)
            
            rslt_prcp.fill(netCDF4.default_fillvals['i2'])
            rslt_prcp_mean.fill(netCDF4.default_fillvals['f4'])
            rslt_prcp_cil.fill(netCDF4.default_fillvals['f4'])
            rslt_prcp_ciu.fill(netCDF4.default_fillvals['f4'])
            rslt_prcp_cir.fill(netCDF4.default_fillvals['f4'])
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_DONE_WRITE) 
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)
            print "Worker "+str(rank)+" completed chunk for tile "+tile_id

def write_rslts(params,tile_id,varname,days,str_row,str_col,rslt,rslt_mean,rslt_cil,rslt_ciu,rslt_cir):

    ds = open_dataset(params, tile_id, varname, days)
    nrows = params[P_CHCKSIZE_Y]
    ncols = params[P_CHCKSIZE_X]
    
    ds.variables[varname][:,str_row:str_row+nrows,str_col:str_col+ncols] = rslt
    ds.variables["".join([varname,"_mean"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_mean    
    ds.variables["".join([varname,"_cil"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_cil 
    ds.variables["".join([varname,"_ciu"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_ciu
    ds.variables["".join([varname,"_cir"])][str_row:str_row+nrows,str_col:str_col+ncols] = rslt_cir
    
    ds.close()

def open_dataset(params,tile_id,varname,days):

    fpath = "".join([params[P_PATH_OUT],tile_id,"/",tile_id,"_",varname,".nc"])

    try:
        
        ds = Dataset(fpath,'r+')
        
    except RuntimeError:
        
        if not os.path.exists("".join([params[P_PATH_OUT],tile_id])):
            os.mkdir("".join([params[P_PATH_OUT],tile_id]))
        ds = create_ncdf(params,fpath,tile_id,varname,days)
    
    #Data is already scaled and set to int16 so turn autoscale off
    ds.variables[varname].set_auto_maskandscale(False)
    return ds
    
def proc_write(params,nwrkers):

    status = MPI.Status()
    tile_num_msg = np.zeros(1,dtype=np.int32)
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    tile_ids, tile_info, ntiles, lons, lats = bcast_msg  
        
    chks_per_tile = (params[P_TILESIZE_X]/params[P_CHCKSIZE_X])*(params[P_TILESIZE_Y]/params[P_CHCKSIZE_Y])
    nchks = chks_per_tile*ntiles
    
    tile_status = {}
    for key in tile_ids.keys():
        tile_status[key] = 0
    
    tile_queues = {}
    for key in tile_ids.keys():
        tile_queues[key] = deque()
    
    stat_chk = status_check(nchks,1)
    
    while 1:
        
        MPI.COMM_WORLD.Recv([tile_num_msg,MPI.INT],source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        tile_num = tile_num_msg[0]
        
        if status.tag == TAG_REQUEST_WRITE:
            
            if len(tile_queues[tile_num]) > 0:
                
                tile_queues[tile_num].append(status.source)
            
            else:
            
                MPI.COMM_WORLD.Send([tile_num_msg,MPI.INT],dest=status.source,tag=TAG_WRITE_PERMIT)
                tile_queues[tile_num].append(status.source)
        
        elif status.tag == TAG_DONE_WRITE:
           
            tile_queues[tile_num].popleft()
            tile_status[tile_num]+=1
            if tile_status[tile_num] == chks_per_tile:
                print "".join(["WRITER|Tile ",tile_ids[tile_num]," complete."])
            stat_chk.increment()
            
            try:
            
                dest = tile_queues[tile_num][0]
                MPI.COMM_WORLD.Send([tile_num_msg,MPI.INT],dest=dest,tag=TAG_WRITE_PERMIT)
            
            except IndexError:
                
                continue

        else: #worker is done
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                return 0
                
def proc_coord(params,nwrkers):
    
    ds_mask = Dataset(params[P_PATH_MASK])
    mask = np.array(ds_mask.variables['mask'][:],dtype=np.bool)
    lons = ds_mask.variables['lon'][:]
    lats = ds_mask.variables['lat'][:]
    nrows = lats.size
    ncols = lons.size
    
    ds_dem = Dataset(params[P_PATH_DEM])
    elev = ds_dem.variables['elev'][:].data
    
    wrk_chk = np.zeros((6,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    
    #Build dict of tile ids
    tile_ids = {}
    tile_info = {}
    x = 0
    cnt_y = 0
    cnt_x = 0
    for i in np.arange(0,nrows,params[P_TILESIZE_Y]):
        
        cnt_x = 0
        for j in np.arange(0,ncols,params[P_TILESIZE_X]):
            
            msk_tile = mask[i:i+params[P_TILESIZE_Y],j:j+params[P_TILESIZE_X]]
            
            if msk_tile[msk_tile].size > 0:
                
                atileid = "".join(["h%02d"%(cnt_x,),"v%02d"%(cnt_y,)])
                tile_ids[x] = atileid
                tile_info[atileid] = (i,j)
                x+=1
            
            cnt_x+=1
        
        cnt_y+=1
    
    if params[P_TILES_PROCESS] is None:
        ntiles = len(tile_ids)
    else:
        ntiles = params[P_TILES_PROCESS].size
    
    MPI.COMM_WORLD.bcast((tile_ids, tile_info, ntiles, lons, lats), root=RANK_COORD)
    print "COORD: Starting to send work chunks to workers..."
    cnt = 0
    k = 0
    for i in np.arange(0,nrows,params[P_TILESIZE_Y]):
        
        for j in np.arange(0,ncols,params[P_TILESIZE_X]):
            
            msk_tile = mask[i:i+params[P_TILESIZE_Y],j:j+params[P_TILESIZE_X]]
            
            if msk_tile[msk_tile].size > 0:
                
                elev_tile = elev[i:i+params[P_TILESIZE_Y],j:j+params[P_TILESIZE_X]]
                llgrid = np.meshgrid(lons[j:j+params[P_TILESIZE_X]],lats[i:i+params[P_TILESIZE_Y]])
                
                process_tile = False
                try:
                    if k in params[P_TILES_PROCESS]:
                        process_tile = True
                except TypeError:
                    process_tile = True
                
                if process_tile:
                    
                    for y in np.arange(0,params[P_TILESIZE_Y],params[P_CHCKSIZE_Y]):
                        
                        for x in np.arange(0,params[P_TILESIZE_X],params[P_CHCKSIZE_X]):
                            
                            rcgrid = np.mgrid[y:y+params[P_CHCKSIZE_Y],x:x+params[P_CHCKSIZE_X]]
                
                            wrk_chk[0,:,:] = rcgrid[0,:,:] #row
                            wrk_chk[1,:,:] = rcgrid[1,:,:] #col
                            wrk_chk[2,:,:] = msk_tile[y:y+params[P_CHCKSIZE_Y],x:x+params[P_CHCKSIZE_X]] #mask
                            wrk_chk[3,:,:] = llgrid[1][y:y+params[P_CHCKSIZE_Y],x:x+params[P_CHCKSIZE_X]] #lat
                            wrk_chk[4,:,:] = llgrid[0][y:y+params[P_CHCKSIZE_Y],x:x+params[P_CHCKSIZE_X]] #lon
                            wrk_chk[5,:,:] = elev_tile[y:y+params[P_CHCKSIZE_Y],x:x+params[P_CHCKSIZE_X]] #elev
                            
                            if cnt < nwrkers:
                                dest = cnt+N_NON_WRKRS
                            else:
                                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                            
                            cnt+=1
                            
                            MPI.COMM_WORLD.Send([wrk_chk,MPI.DOUBLE], dest=dest, tag=k)
                        
                k+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.Send([wrk_chk,MPI.DOUBLE], dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
    print "coord_proc: done"

def create_ncdf(params,fpath,tile_id,varname,days):
    
    ds = Dataset(fpath,'w')
     
    #Set global attributes
    title = "".join(["Daily Interpolated CONUS Meteorological Data ",str(days[YMD][0]),"-",str(days[YMD][-1])])
    ds.title = title
    ds.institution = " | ".join(["University of Montana Numerical Terradynamics Simulation Group",
                                "NASA Ames Ecological Forecasting Lab"])
    ds.source = "wxTopo v0.0.1"
    ds.history = "".join(["Created on: ",datetime.datetime.strftime(date.today(),"%Y-%m-%d")]) 
    ds.references = "http://www.ntsg.umt.edu/project/topomet"
    ds.comment = "30-arcsec spatial resolution, daily timestep"
    
    str_row,str_col = params[P_TILE_INFO][tile_id]
    lons = params[LON][str_col:str_col+params[P_TILESIZE_X]]
    lats = params[LAT][str_row:str_row+params[P_TILESIZE_Y]]
    
    #Create 3-dimensions
    ds.createDimension('time',days.size)
    ds.createDimension('lat',lats.size)
    ds.createDimension('lon',lons.size)
    
    min_date = days[DATE][0]
    #Create dimension variables and fill with values
    times = ds.createVariable('time','f8',('time',),fill_value=False)
    times.long_name = "time"
    times.units = "".join(["days since ",str(min_date.year),"-",str(min_date.month),"-",str(min_date.day)," 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE],times.units)
    
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
    
    mainvar = ds.createVariable(varname,'i2',('time','lat','lon',),chunksizes=(days[DATE].size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))
    mainvar.long_name,mainvar.units,mainvar.standard_name,mainvar.missing_value = VAR_ATTRS[varname]
    mainvar.scale_factor = SCALE_FACTOR
    
    avar = ds.createVariable("".join([varname,"_mean"]),'f4',('lat','lon',),chunksizes=(params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))
    avar.long_name = "".join(["mean ",mainvar.long_name])
    avar.units = mainvar.units
    avar.standard_name = "".join(["mean_",mainvar.standard_name])
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable("".join([varname,"_cil"]),'f4',('lat','lon',),chunksizes=(params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))
    avar.long_name = "".join(["lower confidence interval ",mainvar.long_name])
    avar.units = mainvar.units
    avar.standard_name = "".join(["lower_confidence_interval_",mainvar.standard_name])
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable("".join([varname,"_ciu"]),'f4',('lat','lon',),chunksizes=(params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))
    avar.long_name = "".join(["upper confidence interval ",mainvar.long_name])
    avar.units = mainvar.units
    avar.standard_name = "".join(["upper_confidence_interval_",mainvar.standard_name])
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable("".join([varname,"_cir"]),'f4',('lat','lon',),chunksizes=(params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))
    avar.long_name = "".join(["confidence interval range ",mainvar.long_name])
    avar.units = mainvar.units
    avar.standard_name = "".join(["confidence_interval_range_",mainvar.standard_name])
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/infill/infill_prcp.nc'
    params[P_PATH_MASK] = '/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus_expand.nc'
    params[P_PATH_DEM] = '/projects/daymet2/dem/smoothed/ncdf/dem_orig_expand.nc'
    params[P_PATH_OUT] = '/projects/daymet2/interp_output/wxTopo_tests/wi/'
    params[P_TILESIZE_X] = 250
    params[P_TILESIZE_Y] = 250
    params[P_CHCKSIZE_X] = 50
    params[P_CHCKSIZE_Y] = 50
    params[P_MINSTNS] = 38
    params[P_TILES_PROCESS] = np.array([88])
    params[P_SIGMA] =  0.02507643
    params[P_DF] = 17585 
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
