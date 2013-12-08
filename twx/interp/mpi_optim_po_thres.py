'''
A MPI driver for calculating monthly optimized prcp. occurrence thresholds for individual stations
@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_infill,STN_ID,LON,LAT
from interp.station_select import station_select
from utils.status_check import status_check
import utils.util_geo as utlg
from utils.ncdf_raster import ncdf_raster
from netCDF4 import Dataset
import netCDF4
import datetime
import interp.interp_prcp as ip

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_MINNGH = 'P_MINNGH'
P_VARNAME = 'P_VARNAME'

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
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    
    mod = ip.modeler_clib()
    po_interper = ip.interp_po(mod)
    
    mth_masks = ip.build_mth_masks(stn_da.days)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    thres_vals = np.zeros(12)
    thres_vals.fill(netCDF4.default_fillvals['f8'])
    
    stn_slct = station_select(stn_da.stns,params[P_MINNGH],params[P_MINNGH]+10)
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*2, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
            
                xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
                xval_obs = stn_da.load_obs(stn_id)
                xval_obs[xval_obs > 0] = 1
            
                dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
                rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))
            
                stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
                
                stn_obs = stn_da.load_obs(stns[STN_ID])
                stn_obs[stn_obs > 0] = 1
                
                po = po_interper.model_po(stns, wgts, stn_obs, xval_stn)
                
                x = 0
                for x in np.arange(len(mth_masks)):
                    
                    mth_mask = mth_masks[x]
                    
                    obs_po_mth = xval_obs[mth_mask]
                    fit_po_mth = po[mth_mask]
                    
                    max_hss = 0
                    max_thres = 0
                    for thres in ip.PO_THRESHS:
                        
                        thres_mask = fit_po_mth >= thres
                        
                        hss = ip.calc_hss(obs_po_mth, thres_mask)
                        if hss > max_hss:
                            max_hss = hss
                            max_thres = thres
                    
                    
                    thres_vals[x] = max_thres
            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not calc thres for ",stn_id,"...",str(e)])
                thres_vals.fill(netCDF4.default_fillvals['f8'])
            
            MPI.COMM_WORLD.send((stn_id,thres_vals), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
            thres_vals.fill(netCDF4.default_fillvals['f8'])
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids = bcast_msg
    print "Writer: Received broadcast msg"
    
    ds = create_ncdf(params,stn_ids)
    print "Writer: Output NCDF file created"
    
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    ttl_xvals = stn_ids.size
    
    stat_chk = status_check(ttl_xvals,100)
    
    while 1:
       
        stn_id,thres_vals = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            ds.variables['po_thres'][:,stn_idxs[stn_id]] = thres_vals
            ds.sync()
            
            print "|".join(["WRITER",stn_id,str(np.mean(thres_vals))])
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_infill(params[P_PATH_DB], params[P_VARNAME])
    stn_ids = stn_da.stn_ids
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(stn_ids, root=RANK_COORD)
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in stn_ids:
                
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

def create_ncdf(params,stn_ids):
    
    path = params[P_PATH_OUT]
    
    ncdf_file = Dataset(path,'w')
    
    
    cmts = ''
    
    for key,val in params.items():
        
        if "PATH" not in key:
            
            cmts = "".join([cmts,"|",key,": ",str(val)])
            
    
    #Set global attributes
    title = "Cross Validation "+params[P_VARNAME]
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    ncdf_file.comments = cmts
    
    dim_mth = ncdf_file.createDimension('mth',12)
    dim_station = ncdf_file.createDimension('stn_id',stn_ids.size)
    
    mth = ncdf_file.createVariable('mth','i4',('mth',),fill_value=False)
    mth.long_name = "month"
    mth.standard_name = "month"
    mth[:] = np.arange(1,13)
    
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    thres_var = ncdf_file.createVariable('po_thres','f8',('mth','stn_id'))
    thres_var.long_name = "po_thres"
    thres_var.standard_name = "po_thres"
    thres_var.missing_value = netCDF4.default_fillvals['f8']
    
    ncdf_file.sync()
    
    return ncdf_file

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/infill/infill_prcp.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/po_thres.nc'
    params[P_MINNGH] = 51
    params[P_VARNAME] = 'prcp' 
    
    if rank == RANK_COORD:
        
        proc_coord(params, nsize-N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()