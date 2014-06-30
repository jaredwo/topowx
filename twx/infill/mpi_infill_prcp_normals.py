'''
A MPI driver for producing mean value estimates (i.e--normals) of daily prcp for each station over a specified time period
using the methods of obs_infill_normal.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationDataDb,STN_ID,YEAR,DATE,STN_NAME,ELEV,LON,LAT,STATE
from twx.infill.obs_por import load_por_csv,build_valid_por_masks
from twx.utils.status_check import StatusCheck
from netCDF4 import Dataset,date2num
import netCDF4
import datetime
from twx.infill.infill_normals import infill_prcp_norm,build_mth_masks,MTH_BUFFER
from twx.db.create_db_all_stations import dbDataset

LAST_VAR_WRITTEN = 'err_amt'

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_NCDF_MODE = 'P_NCDF_MODE'

NCDF_CHK_COLS = 50
DS_NAME = 'normals_prcp.nc'

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
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.send(None, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        
        else:
            
            try:
                
                rslts = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks,use_prcp_only=False)
                rslts.calc_obs_vs_fit_stats()
            
            except Exception as e:
                
                print "".join(["ERROR: Could not infill ",stn_id,"|",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send(rslts, dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_prcp = bcast_msg
    print "Writer: Received broadcast msg"
    
    
    if params[P_NCDF_MODE] == 'r+':
        
        ds_prcp = Dataset("".join([params[P_PATH_OUT],DS_NAME]),'r+')
        ttl_infills = stnids_prcp.size
        stnids_prcp = np.array(ds_prcp.variables['stn_id'][:], dtype="<S16")
        
    else:
        
        ds_prcp = create_ncdf(params[P_PATH_OUT],stnids_prcp, stn_da.stns, days)
        ttl_infills = stnids_prcp.size
    
    print "Writer: Output NCDF files ready"
    
    stat_chk = StatusCheck(ttl_infills,100)
    
    while 1:

        rslts = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                ds_prcp.close()
                print "Writer: Finished"
                return 0
        else:
            
            print "|".join(["WRITER",rslts.stn_id,"%.2f"%(rslts.perr_ttlamt,)])
            
            stn_idx = np.nonzero(stnids_prcp == rslts.stn_id)[0][0]
            
            ds_prcp.variables['prcp'][:,stn_idx] = rslts.prcp
            ds_prcp.variables['prcp_mod'][:,stn_idx] = rslts.prcp_fit
            ds_prcp.variables['flag_fill'][:,stn_idx] = rslts.fill_mask
            ds_prcp.variables[LAST_VAR_WRITTEN][stn_idx] = rslts.perr_ttlamt
            
            ds_prcp.sync()
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_prcp = build_valid_por_masks(por)[2]
    
    stnids_prcp = stn_da.stn_ids[mask_por_prcp]
    
    #Check if we're restarting a run
    if params[P_NCDF_MODE] == 'r+':
        
        #If rerunning remove stn ids that have already been completed
        ds_prcp = Dataset("".join([params[P_PATH_OUT],DS_NAME]))
        mask_incplt = ds_prcp.variables[LAST_VAR_WRITTEN][:].mask
        stnids_prcp = stnids_prcp[mask_incplt]
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stnids_prcp, root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in stnids_prcp:
                
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1

        MPI.COMM_WORLD.send(stn_id, dest=dest, tag=TAG_DOWORK)
        cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send(None, dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_ncdf(path_out,stnids_prcp,all_stns,days):
    
    stns_prcp = all_stns[np.in1d(all_stns[STN_ID], stnids_prcp, assume_unique=True)]
    
    ds = dbDataset("".join([params[P_PATH_OUT],DS_NAME]),'w')
    
    ds.db_create_global_attributes('Infilled daily prcp for overall daily mean calculations.')
    
    ds.db_create_time_dimvar(days)
    ds.db_create_stnid_dimvar(stns_prcp[STN_ID])
    ds.db_create_stn_vars(stns_prcp)
    
    ds.db_create_prcp_var((days.size,NCDF_CHK_COLS))
    ds.db_create_binflag_var('flag_fill', "infilled flag", "infilled_flag", chunk=(days[DATE].size,NCDF_CHK_COLS))
    
    ncdf_var = ds.createVariable('prcp_mod','f4',('time','stn_id'),chunksizes=(days.size,NCDF_CHK_COLS))
    ncdf_var.long_name = "precipitation amount modeled"
    ncdf_var.units = "cm"
    ncdf_var.standard_name = "precipitation_amount_modeled"
    ncdf_var.missing_value = netCDF4.default_fillvals['f4']
        
    err_var = ds.createVariable('err_amt','f4',('stn_id',))
    err_var.long_name = "percentage error amount"
    err_var.units = "percentage"
    err_var.standard_name = "percentage_error_amount"
    err_var.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/'
    params[P_NCDF_MODE] = 'w' #w or r+
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()