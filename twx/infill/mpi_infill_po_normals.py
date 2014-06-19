'''
A MPI driver for producing mean value estimates (i.e--normals) for each station over a specified time period
using the methods of obs_infill_normal.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationDataDb,STN_ID,YEAR,DATE,STN_NAME,ELEV,LON,LAT,STATE
from twx.infill.obs_por import load_por_csv,build_valid_por_masks
from twx.utils.status_check import status_check
from netCDF4 import Dataset,date2num
import netCDF4
import datetime
from twx.infill.infill_normals import infill_po,build_mth_masks,MTH_BUFFER

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
            MPI.COMM_WORLD.send([None]*5, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                fit_po, obs_po = infill_po(stn_id, stn_da, days, mth_masks, mthbuf_masks)
                
                nan_mask = np.isnan(obs_po)
                fnl_po = np.copy(obs_po)
                fnl_po[nan_mask] = fit_po[nan_mask]
                fill_mask = nan_mask
                
                fin_mask = np.logical_not(nan_mask)
                val_fit = fit_po[fin_mask]
                val_obs = obs_po[fin_mask]
                
                mae = np.mean(np.abs(val_fit-val_obs))
                bias = np.mean(val_fit-val_obs)
            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill po for ",stn_id,"|",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send((stn_id,mae,bias,fnl_po,fill_mask), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stnids_prcp = bcast_msg
    print "Writer: Received broadcast msg"
            
    if params[P_NCDF_MODE] == 'r+':
        
        ds_prcp = Dataset(params[P_PATH_OUT],'r+')
        ttl_infills = stnids_prcp.size
        stnids_prcp = np.array(ds_prcp.variables['stn_id'][:], dtype="<S16")
        print "Writer: Output NCDF file ready"
        
    else:
        
        ds_prcp = create_ncdf(params[P_PATH_OUT],stnids_prcp, stn_da.stns, days)
        ttl_infills = stnids_prcp.size
        print "Writer: Output NCDF file created"

    stat_chk = status_check(ttl_infills,30)
    
    while 1:
       
        stn_id,mae,bias,fnl_po,fill_mask = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            x = np.nonzero(stnids_prcp == stn_id)[0][0]
            
            ds_prcp.variables['po'][:,x] = fnl_po
            ds_prcp.variables['flag_fill'][:,x] = fill_mask
            ds_prcp.variables['mae'][x] = mae
            ds_prcp.variables['bias'][x] = bias
            ds_prcp.sync()
            
            print "|".join(["WRITER",stn_id,"%.4f"%(mae,),"%.4f"%(bias,)])
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_prcp = build_valid_por_masks(por)[2]
    
    #Extract stn_ids that have min # of observations
    stn_ids_prcp = por[STN_ID][mask_por_prcp]
    
    #Check if we're restarting a run
    if params[P_NCDF_MODE] == 'r+':
        
        #If rerunning remove stn ids that have already been completed
        ds = Dataset(params[P_PATH_OUT])
        mask_incplt = ds.variables['bias'][:].mask
        stnids_incplt = stn_da.stn_ids[mask_incplt]
        
        stn_ids_prcp = stn_ids_prcp[np.in1d(stn_ids_prcp, stnids_incplt, assume_unique=True)]
    
    print "".join(["Coord: Total # of stns to infill: ",str(stn_ids_prcp.size)])
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stn_ids_prcp, root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in stn_ids_prcp:
        
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
    
    ds = Dataset(path_out,'w')
    
    #Set global attributes
    
    ds.title = "".join(['Infilled daily prcp occurannce via HSS weighted average.'])
    ds.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ds.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    
    dim_time = ds.createDimension('time',days.size)
    dim_station = ds.createDimension('stn_id',stns_prcp.size)
    
    times = ds.createVariable('time','f8',('time',))
    times.long_name = "time"
    times.units = "".join(["days since ",str(np.min(days[YEAR])),"-1-1 0:0:0"])
    times.standard_name = "time"
    times.calendar = "standard"
    times[:] = date2num(days[DATE],times.units)
    
    stations = ds.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stns_prcp[STN_ID],dtype=np.object)
    
    names = ds.createVariable('name','str',('stn_id',))
    names.long_name = "station name"
    names.standard_name = "name"
    names[:] = np.array(stns_prcp[STN_NAME],dtype=np.object)
    
    states = ds.createVariable('state','str',('stn_id',))
    states.long_name = "state"
    states.standard_name = "state"
    states[:] = np.array(stns_prcp[STATE],dtype=np.object)
    
    latitudes = ds.createVariable('lat','f8',('stn_id',))
    latitudes.long_name = "latitude"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    latitudes[:] = stns_prcp[LAT]
    
    longitudes = ds.createVariable('lon','f8',('stn_id',))
    longitudes.long_name = "longitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"
    longitudes[:] = stns_prcp[LON]
    
    elevs = ds.createVariable('elev','f8',('stn_id',))
    elevs.long_name = "elevation"
    elevs.units = "m"
    elevs.standard_name = "elevation"
    elevs[:] = stns_prcp[ELEV]
    
    flag_var = ds.createVariable('flag_fill','i1',('time','stn_id'),chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    flag_var.long_name = "infilled flag"
    flag_var.standard_name = "infilled_flag"
    flag_var.missing_value = netCDF4.default_fillvals['i1']
    
    prcp_var = ds.createVariable('po','f4',('time','stn_id'),chunksizes=(days[DATE].size,NCDF_CHK_COLS))
    prcp_var.long_name = "precipitation occurrence"
    prcp_var.standard_name = "precipitation_occurrence"
    prcp_var.missing_value = netCDF4.default_fillvals['f4']
    
    mae_var = ds.createVariable('mae','f4',('stn_id',))
    mae_var.long_name = "mae"
    mae_var.standard_name = "mae"
    mae_var.missing_value = netCDF4.default_fillvals['f4']
    
    bias_var = ds.createVariable('bias','f4',('stn_id',))
    bias_var.long_name = "bias"
    bias_var.standard_name = "bias"
    bias_var.missing_value = netCDF4.default_fillvals['f4']
    
    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/normals_po.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_NCDF_MODE] = 'w'#'r+'
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()