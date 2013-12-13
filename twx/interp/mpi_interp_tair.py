'''
A MPI driver for interpolating tair to a specified grid using interp.interp_tair

@author: jared.oyler
'''

from mpi4py import MPI
import sys
from twx.db.station_data import LON,LAT,NEON,ELEV,TDI
import twx.interp.interp_tair as it
from twx.utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from collections import deque
import tiling as tl
import numpy as np

TAG_DOWORK = 1
TAG_STOPWORK = 1000
TAG_REQUEST_WRITE = 3
TAG_WRITE_PERMIT = 4
TAG_DONE_WRITE = 5

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_MASK = 'P_PATH_MASK'
P_PATH_ELEV = 'P_PATH_ELEV'
P_PATH_TDI = 'P_PATH_TDI'
P_PATH_LST_TMIN = 'P_PATH_LST_TMIN'
P_PATH_LST_TMAX = 'P_PATH_LST_TMAX'
P_PATH_LC = 'P_PATH_LC'
P_PATH_VCF = 'P_PATH_VCF'
P_PATH_NEON = 'P_PATH_NEON'
P_PATH_DB_TMIN = 'P_PATH_DB_TMIN'
P_PATH_DB_TMAX = 'P_PATH_DB_TMAX'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_DUPS_TMIN = 'P_PATH_DUPS_TMIN'
P_PATH_DUPS_TMAX = 'P_PATH_DUPS_TMAX'
P_PATH_NEON_PARAMS_TMIN = 'P_PATH_NEON_PARAMS_TMIN'
P_PATH_RMSTNS_TMIN = "P_PATH_RMSTNS_TMIN"
P_PATH_RMSTNS_TMAX = "P_PATH_RMSTNS_TMAX"
P_PATH_PARAMS_MEAN_TMIN = 'P_PATH_PARAMS_MEAN_TMIN'
P_PATH_PARAMS_ANOM_TMIN = 'P_PATH_PARAMS_ANOM_TMIN'
P_PATH_PARAMS_MEAN_TMAX = 'P_PATH_PARAMS_MEAN_TMAX'
P_PATH_PARAMS_ANOM_TMAX = 'P_PATH_PARAMS_ANOM_TMAX'

P_TILESIZE_X = 'P_TILESIZE_X'
P_TILESIZE_Y = 'P_TILESIZE_Y'
P_CHCKSIZE_X = 'P_CHCKSIZE_X'
P_CHCKSIZE_Y = 'P_CHCKSIZE_Y'
P_TILE_INFO = 'P_TILE_INFO'
P_TILES_PROCESS = 'P_TILES_PROCESS'

SCALE_FACTOR = np.float32(0.01)

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

def proc_work(params,rank):

    status = MPI.Status()
    
    stndaTmin = it.StationDataWrkChk(params[P_PATH_DB_TMIN], 'tmin')
    stndaTmax = it.StationDataWrkChk(params[P_PATH_DB_TMAX], 'tmax')
    ptInterp = it.PtInterpTair(stndaTmin,stndaTmax)    
    
    days = ptInterp.days   
    ninvalid_warn_cutoff = np.round(days.size*.10) #10% or greater days tmin >= tmax
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    tile_grid_info = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    awriter = tl.tile_writer(tile_grid_info, params[P_PATH_OUT])
    
    wrk_chk = np.zeros((32,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]))*np.nan
    
    rslt_tmin = np.empty((days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
    rslt_tmin.fill(netCDF4.default_fillvals['i2'])
    rslt_tmin_norm = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_tmin_se = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    
    rslt_tmax = np.empty((days.size,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16)
    rslt_tmax.fill(netCDF4.default_fillvals['i2'])
    rslt_tmax_norm = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    rslt_tmax_se = np.ones((12,params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.float32)*netCDF4.default_fillvals['f4']
    
    rslt_ninvalid = np.ones((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int32)*netCDF4.default_fillvals['i4']
    
    tile_num = np.zeros(1,dtype=np.int32)
    
    mths = np.arange(1,13)
    
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
            
            minLat = wrk_chk[3,-1,0]
            maxLat = wrk_chk[3,0,0]
            minLon = wrk_chk[4,0,0]
            maxLon = wrk_chk[4,0,-1]
        
            bnds = (minLat,maxLat,minLon,maxLon)
            stndaTmin.set_obs(bnds)
            stndaTmax.set_obs(bnds)
            
            if rank == 2:
                stat_chk = status_check(params[P_CHCKSIZE_Y]*params[P_CHCKSIZE_X],30)
            
            for r in np.arange(params[P_CHCKSIZE_Y]):
            
                for c in np.arange(params[P_CHCKSIZE_X]):
                    
                    error = False
                    
                    if wrk_chk[2,r,c]:
                        
                        try:
                            #row,col,mask,lat,lon,elev,tdi,lst_tmin,lst_tmax,neon = wrk_chk[:,r,c]
                          
                            ptInterp.a_pt[LAT] = wrk_chk[3,r,c]
                            ptInterp.a_pt[LON] = wrk_chk[4,r,c]
                            ptInterp.a_pt[ELEV] = wrk_chk[5,r,c]
                            ptInterp.a_pt[TDI] = wrk_chk[6,r,c]
                            ptInterp.a_pt[NEON] = wrk_chk[7,r,c]
                            for mth in mths:
                                ptInterp.a_pt["tmin%02d"%mth] = wrk_chk[8+(mth-1),r,c]
                                ptInterp.a_pt["tmax%02d"%mth] = wrk_chk[20+(mth-1),r,c]
                            ptInterp.a_pt[NEON]
                            
                            tmin_dly, tmax_dly, tmin_norms, tmax_norms, tmin_se, tmax_se, ninvalid = ptInterp.interpPt()
                          
                            if ninvalid >= ninvalid_warn_cutoff:
                                print "".join(["WARNING: ","Point had ",str(ninvalid)," days tmin >= tmax: ",
                                               str(ptInterp.a_pt[LON])," ",str(ptInterp.a_pt[LAT])])
                            
                        except Exception as e:
                            
                            print "".join(["ERROR: Could not interp ",
                                           str(ptInterp.a_pt[LON])," ",str(ptInterp.a_pt[LAT]),". Leaving output as fill values: ",str(e)])
                            error = True
                            
                        if not error:                                          
                            rslt_tmin[:,r,c] = np.round(tmin_dly,2)/SCALE_FACTOR
                            rslt_tmax[:,r,c] = np.round(tmax_dly,2)/SCALE_FACTOR
                            
                            rslt_tmin_norm[:,r,c] = tmin_norms
                            rslt_tmax_norm[:,r,c] = tmax_norms
                                                   
                            rslt_tmin_se[:,r,c] = tmin_se
                            rslt_tmax_se[:,r,c] = tmax_se
                            
                            rslt_ninvalid[r,c] = ninvalid
                        
                        if rank == 2:
                            stat_chk.increment()
            
            MPI.COMM_WORLD.Send([tile_num,MPI.INT],dest=RANK_WRITE,tag=TAG_REQUEST_WRITE) 
            MPI.COMM_WORLD.Recv([tile_num,MPI.INT],source=RANK_WRITE,tag=TAG_WRITE_PERMIT)

            ##do write
            awriter.write_rslts(tile_id,'tmin', days, str_row, str_col,
                                rslt_tmin, rslt_tmin_norm, rslt_tmin_se,rslt_ninvalid)
            
            awriter.write_rslts(tile_id, 'tmax', days, str_row, str_col,
                                rslt_tmax, rslt_tmax_norm, rslt_tmax_se,rslt_ninvalid)
                
            rslt_tmin.fill(netCDF4.default_fillvals['i2'])
            rslt_tmin_norm.fill(netCDF4.default_fillvals['f4'])
            rslt_tmin_se.fill(netCDF4.default_fillvals['f4'])
            
            rslt_tmax.fill(netCDF4.default_fillvals['i2'])
            rslt_tmax_norm.fill(netCDF4.default_fillvals['f4'])
            rslt_tmax_se.fill(netCDF4.default_fillvals['f4'])
            rslt_ninvalid.fill(netCDF4.default_fillvals['i4'])
            
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
    ds_elev = Dataset(params[P_PATH_ELEV])
    ds_tdi = Dataset(params[P_PATH_TDI])
    ds_neon = Dataset(params[P_PATH_NEON])
    
    ds_attrs = [('elev',ds_elev),('tdi',ds_tdi),('neon',ds_neon)]
    ds_attrs_lst_tmin = [('tmin%02d'%mth,Dataset(params[P_PATH_LST_TMIN][mth-1])) for mth in np.arange(1,13)]
    ds_attrs_lst_tmax = [('tmax%02d'%mth,Dataset(params[P_PATH_LST_TMAX][mth-1])) for mth in np.arange(1,13)]
    ds_attrs.extend(ds_attrs_lst_tmin)
    ds_attrs.extend(ds_attrs_lst_tmax)
    
    atiler = tl.tiler(ds_mask, ds_attrs, params[P_TILESIZE_Y], params[P_TILESIZE_X], 
                      params[P_CHCKSIZE_Y], params[P_CHCKSIZE_X], params[P_TILES_PROCESS])
    
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
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    
    #CONUS SCALE RUN
    ############################################################################################
    params[P_PATH_DB_TMIN] = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmin.nc'
    params[P_PATH_DB_TMAX] = '/projects/daymet2/station_data/infill/serial_fnl/serial_tmax.nc'

    gridPath = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    params[P_PATH_MASK] = "".join([gridPath,'fnl_mask.nc'])
    params[P_PATH_ELEV] = "".join([gridPath,'fnl_elev.nc'])
    params[P_PATH_TDI] = "".join([gridPath,'fnl_tdi.nc'])
    params[P_PATH_LST_TMIN] = ["".join([gridPath,'fnl_lst_tmin%02d.nc'%mth]) for mth in np.arange(1,13)]
    params[P_PATH_LST_TMAX] = ["".join([gridPath,'fnl_lst_tmax%02d.nc'%mth]) for mth in np.arange(1,13)]
    params[P_PATH_NEON] = "".join([gridPath,'fnl_climdiv.nc'])
    
    params[P_PATH_OUT] = '/stage/climate/topowx_tile_output/'
    params[P_TILESIZE_X] = 250
    params[P_TILESIZE_Y] = 250
    params[P_CHCKSIZE_X] = 50
    params[P_CHCKSIZE_Y] = 50
    
    mcoUsgsTiles = np.loadtxt('/stage/climate/topowx_tiles_shp/McoUsgsTileList.csv',
                              dtype=np.int,usecols=[0],delimiter=',',skiprows=1)
    #doneTiles = np.array([2,3,4,16,17,18,37,38])
    params[P_TILES_PROCESS] = mcoUsgsTiles[mcoUsgsTiles > 173]
    #params[P_TILES_PROCESS] = mcoUsgsTiles[~np.in1d(mcoUsgsTiles, doneTiles, True)]
    ############################################################################################
    #params[P_TILES_PROCESS] = np.array([74,75]) #Wisconsin
    #params[P_TILES_PROCESS] = np.array([16,17,18,37,38])#16,17,18,37,38 #CCE
    #params[P_TILES_PROCESS] = np.array([17,139])
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
