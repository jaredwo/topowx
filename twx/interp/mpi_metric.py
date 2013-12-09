'''
An MPI driver for calculating various metrics from tiled output

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.utils.status_check import status_check
from netCDF4 import Dataset,num2date
import netCDF4
from collections import deque
import twx.interp.tiling as ti
import twx.utils.util_dates as utld
from twx.utils.util_dates import MONTH,YEAR
from twx.utils.degree_days import gdd_wheat,dd_cooling,dd_heating,gdd_wheat_basic
from twx.interp_constants import *

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
P_PATH_DB_TMIN = 'P_PATH_DB_TMIN'
P_PATH_DB_TMAX = 'P_PATH_DB_TMAX'
P_PATH_OUT = 'P_PATH_OUT'

P_MINSTNS_TMIN = 'P_MINSTNS_TMIN'
P_MINSTNS_TMAX = 'P_MINSTNS_TMAX'
P_TILESIZE_X = 'P_TILESIZE_X'
P_TILESIZE_Y = 'P_TILESIZE_Y'
P_CHCKSIZE_X = 'P_CHCKSIZE_X'
P_CHCKSIZE_Y = 'P_CHCKSIZE_Y'
P_TILE_INFO = 'P_TILE_INFO'
P_TILES_PROCESS = 'P_TILES_PROCESS'
P_SIGMA_TMIN = 'P_SIGMA_TMIN'
P_SIGMA_TMAX = 'P_SIGMA_TMAX'
P_DF_TMIN = 'P_DF_TMIN'
P_DF_TMAX = 'P_DF_TMAX'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NNGH_TMIN = 'P_NNGH_TMIN'
P_NNGH_TMAX = 'P_NNGH_TMAX'
P_NCDF_MODE = 'P_NCDF_MODE'

SCALE_FACTOR = np.float32(0.01)

RM_STN_IDS = np.array(['RAWS_NLIM','RAWS_OKEE','RAWS_NHCA'])

#long name, units, standard name, missing value
VAR_ATTRS = {'tmin':("minimum air temperature","C","minimum_air_temperature",netCDF4.default_fillvals['i2']),
             'tmax':("maximum air temperature","C","maximum_air_temperature",netCDF4.default_fillvals['i2'])}

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)


def proc_work_hdd(params,rank):
    
    status = MPI.Status()
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    tile_grid_info = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    ds_varnames = ["cdd_norm","cdd_anom_2004","cdd_anom_2007","cdd_anom_2011"]
    
    awriter = ti.tile_writer_metric(tile_grid_info, params[P_PATH_OUT], "Cooling Degree Days Norms/Anomalies",ds_varnames)
    areader = ti.tile_reader(tile_grid_info, params[P_PATH_OUT])
    
    wrk_chk = np.zeros((5,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    
    tile_num = np.zeros(1,dtype=np.int32)
    
    days = None
    
    #Years in 1949 - 2011 normals
    norm_yrs = np.arange(1948,2012)
    
    while 1:
    
        MPI.COMM_WORLD.Recv([wrk_chk,MPI.DOUBLE],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            tile_num[0] = status.tag
            tile_id = tile_grid_info.get_tile_id(tile_num[0])
            
            str_row = np.int(wrk_chk[0,0,0])
            str_col = np.int(wrk_chk[1,0,0])
            
            ds_tmin = areader.open_dataset(tile_id,"tmin")
            ds_tmax = areader.open_dataset(tile_id,"tmax")
            
            if days is None:
               
                #Get start and end date of current dataset
                min_date = num2date(ds_tmin.variables['time'][0], units=ds_tmin.variables['time'].units)
                max_date = num2date(ds_tmin.variables['time'][-1], units=ds_tmin.variables['time'].units)
                days = utld.get_days_metadata(min_date, max_date) 
                
            
            tmin_chk = ds_tmin.variables['tmin'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            tmax_chk = ds_tmax.variables['tmax'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            
            if np.ma.isMA(tmin_chk):
                nodata = tmin_chk.mask[0,:,:]
                tmin_chk = tmin_chk.data
                tmax_chk = tmax_chk.data
            else:
                nodata = np.zeros((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.bool)
            
            hdd_ann = np.empty((norm_yrs.size,tmin_chk.shape[1],tmin_chk.shape[2]))
            
            for x in np.arange(norm_yrs.size):
                
                yr = norm_yrs[x]
                #Winter mask (DJF)
                #day_mask = np.logical_or(np.logical_and(days[MONTH] == 12,days[YEAR] == (yr - 1)),np.logical_and(days[MONTH] <= 2,days[YEAR] == yr))
                #Summer mask
                day_mask = np.logical_and(days[MONTH] >= 6,np.logical_and(days[MONTH] <= 8,days[YEAR] == yr))
                
                hdd_ann[x,:,:] = np.sum(dd_cooling(tmin_chk[day_mask,:,:], tmax_chk[day_mask,:,:]) ,axis=0)     

            hdd_norm = np.mean(hdd_ann,axis=0)
            hdd_2004 = hdd_ann[np.nonzero(norm_yrs==2004)[0][0],:,:]
            hdd_2007 = hdd_ann[np.nonzero(norm_yrs==2007)[0][0],:,:]
            hdd_2011 = hdd_ann[np.nonzero(norm_yrs==2011)[0][0],:,:]
            
            hasdata = np.logical_not(nodata)
            hdd_2004_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            hdd_2007_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            hdd_2011_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            
            hdd_norm[nodata] = netCDF4.default_fillvals['f4']
            
            hdd_2004_anomly[hasdata] = ((hdd_2004[hasdata] - hdd_norm[hasdata])/hdd_norm[hasdata])*100.
            hdd_2004_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            hdd_2007_anomly[hasdata] = ((hdd_2007[hasdata] - hdd_norm[hasdata])/hdd_norm[hasdata])*100.
            hdd_2007_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            hdd_2011_anomly[hasdata] = ((hdd_2011[hasdata] - hdd_norm[hasdata])/hdd_norm[hasdata])*100.
            hdd_2011_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_REQUEST_WRITE) 
            MPI.COMM_WORLD.Recv([tile_num,MPI.INT],source=RANK_WRITE,tag=TAG_WRITE_PERMIT)

            ##do write
            awriter.write_rslts(tile_id, 'cdd', days, str_row, str_col,hdd_norm,hdd_2004_anomly,hdd_2007_anomly,hdd_2011_anomly)
                            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_DONE_WRITE) 
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)
            print "Worker "+str(rank)+" completed chunk for tile "+tile_id


def proc_work_gdd(params,rank):
    
    status = MPI.Status()
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    tile_grid_info = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    awriter = ti.tile_writer_gdd(tile_grid_info, params[P_PATH_OUT])
    areader = ti.tile_reader(tile_grid_info, params[P_PATH_OUT])
    
    wrk_chk = np.zeros((5,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    
    tile_num = np.zeros(1,dtype=np.int32)
    
    days = None
    
    #Years in 1949 - 2011 normals
    norm_yrs = np.arange(1949,2012)
    
    while 1:
    
        MPI.COMM_WORLD.Recv([wrk_chk,MPI.DOUBLE],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            tile_num[0] = status.tag
            tile_id = tile_grid_info.get_tile_id(tile_num[0])
            
            str_row = np.int(wrk_chk[0,0,0])
            str_col = np.int(wrk_chk[1,0,0])
            
            ds_tmin = areader.open_dataset(tile_id,"tmin")
            ds_tmax = areader.open_dataset(tile_id,"tmax")
            
            if days is None:
               
                #Get start and end date of current dataset
                min_date = num2date(ds_tmin.variables['time'][0], units=ds_tmin.variables['time'].units)
                max_date = num2date(ds_tmin.variables['time'][-1], units=ds_tmin.variables['time'].units)
                days = utld.get_days_metadata(min_date, max_date) 
                
            
            tmin_chk = ds_tmin.variables['tmin'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            tmax_chk = ds_tmax.variables['tmax'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            
            if np.ma.isMA(tmin_chk):
                nodata = tmin_chk.mask[0,:,:]
                tmin_chk = tmin_chk.data
                tmax_chk = tmax_chk.data
            else:
                nodata = np.zeros((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.bool)
            
            gdd_ann = np.empty((norm_yrs.size,tmin_chk.shape[1],tmin_chk.shape[2]))
            
            for x in np.arange(norm_yrs.size):
                
                yr = norm_yrs[x]
                day_mask = np.logical_or(np.logical_and(days[MONTH] >= 10,days[YEAR] == (yr - 1)),np.logical_and(days[MONTH] < 10,days[YEAR] == yr))
                
                gdd_cs = np.cumsum(gdd_wheat_basic(tmin_chk[day_mask,:,:], tmax_chk[day_mask,:,:]),axis=0)
                
                gdd_cs_tmask = gdd_cs >= 2915
                
                for i in np.arange(params[P_CHCKSIZE_Y]):
                    
                    for j in np.arange(params[P_CHCKSIZE_X]):
                        
                        if not nodata[i,j]:
                        
                            try:
                                gdd_ann[x,i,j] = np.nonzero(gdd_cs_tmask[:,i,j])[0][0] + 1
                            except:
                                gdd_ann[x,i,j] = 365
                            
                #gdd_ann[x,:,:] = np.sum(gdd_wheat(tmin_chk[day_mask,:,:], tmax_chk[day_mask,:,:]) ,axis=0)     

            gdd_norm = np.mean(gdd_ann,axis=0)
            gdd_2004 = gdd_ann[np.nonzero(norm_yrs==2004)[0][0],:,:]
            gdd_2007 = gdd_ann[np.nonzero(norm_yrs==2007)[0][0],:,:]
            gdd_2011 = gdd_ann[np.nonzero(norm_yrs==2011)[0][0],:,:]
            
            hasdata = np.logical_not(nodata)
            gdd_2004_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            gdd_2007_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            gdd_2011_anomly = np.empty((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float)
            
            gdd_norm[nodata] = netCDF4.default_fillvals['f4']
            
            #gdd_2004_anomly[hasdata] = ((gdd_2004[hasdata] - gdd_norm[hasdata])/gdd_norm[hasdata])*100.
            gdd_2004_anomly[hasdata] = gdd_2004[hasdata] - gdd_norm[hasdata]
            gdd_2004_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            #gdd_2007_anomly[hasdata] = ((gdd_2007[hasdata] - gdd_norm[hasdata])/gdd_norm[hasdata])*100.
            gdd_2007_anomly[hasdata] = gdd_2007[hasdata] - gdd_norm[hasdata]
            gdd_2007_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            #gdd_2011_anomly[hasdata] = ((gdd_2011[hasdata] - gdd_norm[hasdata])/gdd_norm[hasdata])*100.
            gdd_2011_anomly[hasdata] = gdd_2011[hasdata] - gdd_norm[hasdata]
            gdd_2011_anomly[nodata] = netCDF4.default_fillvals['f4']
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_REQUEST_WRITE) 
            MPI.COMM_WORLD.Recv([tile_num,MPI.INT],source=RANK_WRITE,tag=TAG_WRITE_PERMIT)

            ##do write
            awriter.write_rslts(tile_id, 'gdd_len', days, str_row, str_col,gdd_norm,gdd_2004_anomly,gdd_2007_anomly,gdd_2011_anomly)
                            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_DONE_WRITE) 
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)
            print "Worker "+str(rank)+" completed chunk for tile "+tile_id

def proc_work_tair_dif(params,rank):
    
    status = MPI.Status()
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    tile_grid_info = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    ds_varnames = ["tavg_norm","tavg_anom_2007","tavg_anom_2011"]

    awriter = ti.tile_writer_metric(tile_grid_info, params[P_PATH_OUT], "Tavg Anomalies 2007 and 2011", ds_varnames)
    areader = ti.tile_reader(tile_grid_info, params[P_PATH_OUT])
    
    wrk_chk = np.zeros((5,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    
    tile_num = np.zeros(1,dtype=np.int32)
    
    days = None
    norm_yr_masks = None
    mask_2007 = None
    mask_2011 = None
    
    #Years in 1981 - 2010 normals
    norm_yrs = np.arange(1948,2012)

    while 1:
    
        MPI.COMM_WORLD.Recv([wrk_chk,MPI.DOUBLE],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            tile_num[0] = status.tag
            tile_id = tile_grid_info.get_tile_id(tile_num[0])
            
            str_row = np.int(wrk_chk[0,0,0])
            str_col = np.int(wrk_chk[1,0,0])
            
            ds_tmin = areader.open_dataset(tile_id,"tmin")
            ds_tmax = areader.open_dataset(tile_id,"tmax")
            
            if days is None:
               
                #Get start and end date of current dataset
                min_date = num2date(ds_tmin.variables['time'][0], units=ds_tmin.variables['time'].units)
                max_date = num2date(ds_tmin.variables['time'][-1], units=ds_tmin.variables['time'].units)
                days = utld.get_days_metadata(min_date, max_date)
                
                norm_yr_masks = []
                for yr in norm_yrs:
                    norm_yr_masks.append(days[YEAR]==yr)
                
                mask_2007 = days[YEAR] == 2007
                mask_2011 = days[YEAR] == 2011
            
            tmin_chk = ds_tmin.variables['tmin'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            tmax_chk = ds_tmax.variables['tmax'][:,str_row:str_row+params[P_CHCKSIZE_Y],str_col:str_col+params[P_CHCKSIZE_X]].astype(np.float64)
            
            if np.ma.isMA(tmin_chk):
                nodata_mask = tmin_chk.mask[0,:,:]
            else:
                nodata_mask = None
            
            tavg_chk = (tmin_chk + tmax_chk)/2.0
            
            tavg_norm_chk = np.zeros((tavg_chk.shape[1],tavg_chk.shape[2]))
            
            for yr_mask in norm_yr_masks:
                
                tavg_norm_chk = tavg_norm_chk + np.mean(tavg_chk[yr_mask,:,:],axis=0)
            
            tavg_norm_chk =  tavg_norm_chk/float(len(norm_yr_masks))
            tavg_2007_chk = np.mean(tavg_chk[mask_2007,:,:],axis=0)
            tavg_2011_chk = np.mean(tavg_chk[mask_2011,:,:],axis=0)
            
            tavg_2007_chk = tavg_2007_chk - tavg_norm_chk
            tavg_2011_chk = tavg_2011_chk - tavg_norm_chk
            
            if nodata_mask is not None:
                tavg_norm_chk[nodata_mask] = netCDF4.default_fillvals['f4']
                tavg_2007_chk[nodata_mask] = netCDF4.default_fillvals['f4']
                tavg_2011_chk[nodata_mask] = netCDF4.default_fillvals['f4']
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_REQUEST_WRITE) 
            MPI.COMM_WORLD.Recv([tile_num,MPI.INT],source=RANK_WRITE,tag=TAG_WRITE_PERMIT)

            #do write
            awriter.write_rslts(tile_id, 'tair_anom', days, str_row, str_col,tavg_norm_chk,tavg_2007_chk,tavg_2011_chk)
                            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_DONE_WRITE) 
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)
            print "Worker "+str(rank)+" completed chunk for tile "+tile_id
    
    
def proc_write(params,nwrkers):

    status = MPI.Status()
    tile_num_msg = np.zeros(1,dtype=np.int32)
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    tile_grid_info = bcast_msg  
    tile_ids = tile_grid_info.tile_ids
    nchks = tile_grid_info.nchks
    chks_per_tile = tile_grid_info.chks_per_tile
    
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
    
    atiler = ti.tiler(ds_mask, {}, params[P_TILESIZE_Y], params[P_TILESIZE_X], params[P_CHCKSIZE_Y], params[P_CHCKSIZE_X], params[P_TILES_PROCESS])
    
    MPI.COMM_WORLD.bcast(atiler.build_tile_grid_info(), root=RANK_COORD)
    print "COORD: Starting to send work chunks to workers..."
    
    cnt = 0
    
    try:
    
        while 1:
            
            tile_num,wrk_chk = atiler.next()
            
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            
            cnt+=1
            
            MPI.COMM_WORLD.Send([wrk_chk,MPI.DOUBLE], dest=dest, tag=tile_num)
    
    except StopIteration:
        pass
        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.Send([wrk_chk,MPI.DOUBLE], dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
    print "coord_proc: done"

if __name__ == '__main__':
    
#    np.seterr(all='raise')
#    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_MASK] = '/projects/daymet2/dem/smoothed/ncdf/interp_mask_conus_expand.nc'
    params[P_PATH_OUT] = '/projects/daymet2/climate_office/interp_grid/'
    params[P_TILESIZE_X] = 250
    params[P_TILESIZE_Y] = 250
    params[P_CHCKSIZE_X] = 50
    params[P_CHCKSIZE_Y] = 50
    params[P_TILES_PROCESS] = TILES_MONTANA
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work_gdd(params,rank)

    MPI.COMM_WORLD.Barrier()
