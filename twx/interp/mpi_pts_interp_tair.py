'''
A MPI driver for interpolating tair to a specified grid using interp.interp_tair

@author: jared.oyler
'''

from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,LON,LAT,NEON,ELEV,TDI,LST,VCF,LC,BAD,\
    MASK, OPTIM_NNGH, OPTIM_NNGH_ANOM
import twx.interp.interp_tair as it
from twx.interp.station_select import station_select
from twx.utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from collections import deque
import tiling as tl
import numpy as np
from twx.db.create_db_all_stations import dbDataset
import twx.utils.util_dates as utld
from datetime import datetime
from httplib import HTTPException

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
P_PATH_CLIB = 'P_PATH_CLIB'
P_PATH_RLIB = 'P_PATH_RLIB'
P_PATH_NEON_PARAMS_TMIN = 'P_PATH_NEON_PARAMS_TMIN'
P_PATH_RMSTNS_TMIN = "P_PATH_RMSTNS_TMIN"
P_PATH_RMSTNS_TMAX = "P_PATH_RMSTNS_TMAX"
P_PATH_PARAMS_MEAN_TMIN = 'P_PATH_PARAMS_MEAN_TMIN'
P_PATH_PARAMS_ANOM_TMIN = 'P_PATH_PARAMS_ANOM_TMIN'
P_PATH_PARAMS_MEAN_TMAX = 'P_PATH_PARAMS_MEAN_TMAX'
P_PATH_PARAMS_ANOM_TMAX = 'P_PATH_PARAMS_ANOM_TMAX'
P_STN_IDS = "P_STN_IDS"
P_DAYS = 'P_DAYS'

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

class InterpResults(object):
    
    def __init__(self):
        
        self.stnid = None
        self.tmin_dly = None
        self.tmax_dly = None
        self.tmin_mean = None
        self.tmax_mean = None
        self.tmin_se = None
        self.tmax_se = None
        self.ninvalid = None
        self.ptLon = None
        self.ptLat = None
        self.ptElev = None
        self.ptTdi = None
        self.ptLstTmin = None
        self.ptLstTmax = None
        self.ptClimDiv = None

def proc_work(params,rank):

    status = MPI.Status()
    
    
    auxFpaths = [params[P_PATH_ELEV],
                 params[P_PATH_TDI],
                 params[P_PATH_LST_TMAX],
                 params[P_PATH_LST_TMIN],
                 params[P_PATH_NEON]]
            
    ptInterper = it.PtInterpTair(params[P_PATH_DB_TMIN],
                    params[P_PATH_DB_TMAX],
                    params[P_PATH_RLIB],
                    params[P_PATH_CLIB], 
                    auxFpaths)
            
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                try:
                    idx_stn = ptInterper.interp_tmin.krig_tair.stn_slct.stn_da.stn_idxs[stn_id]
                    xval_stn = ptInterper.interp_tmin.krig_tair.stn_slct.stn_da.stns[idx_stn]
                except KeyError:
                    idx_stn = ptInterper.interp_tmax.krig_tair.stn_slct.stn_da.stn_idxs[stn_id]
                    xval_stn = ptInterper.interp_tmax.krig_tair.stn_slct.stn_da.stns[idx_stn]
                
                tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(xval_stn[LON], xval_stn[LAT])
                
                rslt = InterpResults()
                rslt.stnid = stn_id
                rslt.tmin_dly = tmin_dly
                rslt.tmax_dly = tmax_dly
                rslt.tmin_mean = tmin_mean
                rslt.tmax_mean = tmax_mean
                rslt.tmin_se = tmin_se
                rslt.tmax_se = tmax_se
                rslt.ninvalid = ninvalid
                rslt.ptLon = ptInterper.a_pt[LON]
                rslt.ptLat = ptInterper.a_pt[LAT]
                rslt.ptElev = ptInterper.a_pt[ELEV]
                rslt.ptTdi = ptInterper.a_pt[TDI]
                rslt.ptLstTmin = ptInterper.a_pt[it.LST_TMIN]
                rslt.ptLstTmax = ptInterper.a_pt[it.LST_TMAX]
                rslt.ptClimDiv = ptInterper.a_pt[NEON]
  
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not interp ",stn_id,str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send(rslt, dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
    
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    stat_chk = status_check(params[P_STN_IDS].size,10)
    stn_ids = params[P_STN_IDS]
    dsout = createNcdfOut(params)
    
    while 1:
        
        aRslt = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stn_ids==aRslt.stnid)[0][0]
            dsout.variables['tmin'][:,x] = aRslt.tmin_dly
            dsout.variables['tmax'][:,x] = aRslt.tmax_dly
            dsout.variables['mean_tmin'][x] = aRslt.tmin_mean
            dsout.variables['mean_tmax'][x] = aRslt.tmax_mean
            dsout.variables['se_tmin'][x] = aRslt.tmin_se
            dsout.variables['se_tmax'][x] = aRslt.tmax_se
            dsout.variables['ninvalid'][x] = aRslt.ninvalid
            dsout.variables['lon'][x] = aRslt.ptLon
            dsout.variables['lat'][x] = aRslt.ptLat
            dsout.variables['elev'][x] = aRslt.ptElev
            dsout.variables['tdi'][x] = aRslt.ptTdi
            dsout.variables['lst_tmin'][x] = aRslt.ptLstTmin
            dsout.variables['lst_tmax'][x] = aRslt.ptLstTmax
            dsout.variables['climdiv'][x] = aRslt.ptClimDiv
            
            stat_chk.increment()

def createNcdfOut(params):
    
    ds = dbDataset(params[P_PATH_OUT],'w')
    ds.db_create_stnid_dimvar(params[P_STN_IDS])
    ds.db_create_time_dimvar(params[P_DAYS])
    
    ds.createVariable('tmin',np.float32,('time','stn_id'))
    ds.createVariable('tmax',np.float32,('time','stn_id'))
    ds.createVariable('mean_tmin',np.float64,('stn_id',))
    ds.createVariable('mean_tmax',np.float64,('stn_id',))
    ds.createVariable('se_tmin',np.float64,('stn_id',))
    ds.createVariable('se_tmax',np.float64,('stn_id',))
    ds.createVariable('ninvalid',np.int32,('stn_id',))     
    ds.createVariable('lon',np.float64,('stn_id',))
    ds.createVariable('lat',np.float64,('stn_id',))
    ds.createVariable('elev',np.float64,('stn_id',))
    ds.createVariable('tdi',np.float64,('stn_id',))
    ds.createVariable('lst_tmin',np.float64,('stn_id',))
    ds.createVariable('lst_tmax',np.float64,('stn_id',))
    ds.createVariable('climdiv',np.int32,('stn_id',))
    
    ds.sync()
    
    return ds    
   
               
def proc_coord(params,nwrkers):
        
    print "COORD: Starting to send work chunks to workers..."
    
    cnt = 0
    nrec = 0
                
    for stn_id in params[P_STN_IDS]:
            
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt+=1
                        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "COORD: Done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_CLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C'
    params[P_PATH_RLIB] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R'
    
    #CONUS SCALE RUN
    ############################################################################################
    params[P_PATH_DB_TMIN] = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc'
    params[P_PATH_DB_TMAX] = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc'
    params[P_PATH_MASK] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_mask.nc'
    params[P_PATH_ELEV] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc'
    params[P_PATH_TDI] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc'
    params[P_PATH_LST_TMIN] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc'
    params[P_PATH_LST_TMAX] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc'
    params[P_PATH_NEON] = '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/forRuben/StnGridCellInterps.nc'
    params[P_DAYS] = utld.get_days_metadata(datetime(1948,1,1), datetime(2012,12,31))
    ############################################################################################
    
    
    if rank == RANK_COORD:
        
        stnids = []
        fpaths = ['/projects/daymet2/station_data/forRuben/min20obs/tmax1971-2000.csv',
                  '/projects/daymet2/station_data/forRuben/min20obs/tmin1971-2000.csv',
                  '/projects/daymet2/station_data/forRuben/min20obs/tmax1981-2010.csv',
                  '/projects/daymet2/station_data/forRuben/min20obs/tmin1981-2010.csv']
        
        for fpath in fpaths:
            stnids.extend(np.loadtxt(fpath,np.str, delimiter=",",skiprows=1, usecols=[0]))
        
        params[P_STN_IDS] = np.unique(np.array(stnids))
        
        #Send stn ids to all processes
        MPI.COMM_WORLD.bcast(params[P_STN_IDS], root=RANK_COORD)
        
        proc_coord(params, nsize-N_NON_WRKRS)
    
    elif rank == RANK_WRITE:
        
        bcast_msg = None
        bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
        params[P_STN_IDS] = bcast_msg
        
        proc_write(params,nsize-N_NON_WRKRS)
    
    else:
        
        bcast_msg = None
        bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
        params[P_STN_IDS] = bcast_msg
        
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
