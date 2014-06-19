'''
A MPI driver for producing mean value estimates (i.e--normals) for each station over a specified time period
using the methods of obs_infill_normal.

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationDataDb,STN_ID,MEAN_TMIN,MEAN_TMAX,VAR_TMIN,VAR_TMAX
from twx.infill.obs_por import load_por_csv,build_valid_por_masks
from twx.utils.status_check import status_check
import netCDF4
from twx.infill.infill_normals import impute_tair_norm
from twx.infill.infill_daily import source_r
from twx.db.reanalysis import NNRNghData
from twx.interp.clibs import clib_wxTopo

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_POR = 'P_PATH_POR'
P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
P_PATH_NNR = 'P_PATH_NNR'
P_PATH_CLIB = 'P_PATH_CLIB'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_MIN_POR = 'P_MIN_POR'
P_STN_LOC_BNDS = 'P_STN_LOC_BNDS'
P_MIN_NNGH_DAILY = 'P_MIN_NNGH_DAILY'

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
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    #mask_por_tmin,mask_por_tmax = bcast_msg
    
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por,params[P_MIN_POR],params[P_STN_LOC_BNDS])
    
    source_r(params[P_PATH_R_FUNCS])
    
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
        
    aclib = clib_wxTopo()
        
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    mae = netCDF4.default_fillvals['f4']
    bias = netCDF4.default_fillvals['f4']
    
    while 1:
    
        stn_id,tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)

        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*6, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                norm,vary = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var],tair_var,ds_nnr,aclib,nnghs=params[P_MIN_NNGH_DAILY])[0]
            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill ",
                               tair_var," for ",stn_id,str(e)])
                
                norm = netCDF4.default_fillvals['f8']
                vary = netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,tair_var,mae,bias,norm,vary), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    stn_da = StationDataDb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]),mode="r+")
    var_meantmin = stn_da.add_stn_variable(MEAN_TMIN,MEAN_TMIN,"C",'f8')
    var_meantmax = stn_da.add_stn_variable(MEAN_TMAX,MEAN_TMAX,"C",'f8')
    var_vartmin = stn_da.add_stn_variable(VAR_TMIN,VAR_TMIN,"C**2",'f8')
    var_vartmax = stn_da.add_stn_variable(VAR_TMAX,VAR_TMAX,"C**2",'f8')
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    mask_por_tmin,mask_por_tmax = bcast_msg
    stn_ids_tmin,stn_ids_tmax = stn_da.stn_ids[mask_por_tmin],stn_da.stn_ids[mask_por_tmax]
    print "Writer: Received broadcast msg"
    stn_ids_uniq = np.unique(np.concatenate([stn_ids_tmin,stn_ids_tmax]))
    
    stn_idxs = {}
    for x in np.arange(stn_da.stn_ids.size):
        if stn_da.stn_ids[x] in stn_ids_uniq:
            stn_idxs[stn_da.stn_ids[x]] = x
        
    tair_varmeans = {'tmin':var_meantmin,'tmax':var_meantmax}
    tair_varvary = {'tmin':var_vartmin,'tmax':var_vartmax}
    
    ttl_infills = stn_ids_tmin.size + stn_ids_tmax.size
    
    stat_chk = status_check(ttl_infills,30)
    
    while 1:
       
        stn_id,tair_var,mae,bias,norm,vary = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            stnid_dim = stn_idxs[stn_id]
            
            #print "|".join(["WRITER",stn_id,tair_var,"%.4f"%(mae,),"%.4f"%(bias,),"%.4f"%(norm,),"%.4f"%(vary,)])
            
            tair_varmeans[tair_var][stnid_dim] = norm
            tair_varvary[tair_var][stnid_dim] = vary
            stn_da.ds.sync()
                        
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    
    mask_por_tmin,mask_por_tmax,mask_por_prcp = build_valid_por_masks(por,params[P_MIN_POR],params[P_STN_LOC_BNDS])
    
    #mask_ca = np.char.startswith(por[STN_ID], "GHCN_CA")
    #mask_por_tmin = np.logical_and(mask_por_tmin,mask_ca)
    #mask_por_tmax = np.logical_and(mask_por_tmax,mask_ca)
    
#    stns_ids = np.array(['GHCN_USC00045866','GHCN_USC00046943','GHCN_USC00047011',
#                         'GHCN_USC00047024','GHCN_USC00047767','GHCN_USW00093226','RAWS_CDNO','RAWS_CSBB'])
    
    #Extract stn_ids that have min # of observations
    stn_ids_tmin = por[STN_ID][mask_por_tmin]
    stn_ids_tmax = por[STN_ID][mask_por_tmax]
    
#    stn_ids_tmin = stn_ids_tmin[np.in1d(stn_ids_tmin, stns_ids, True)]
#    stn_ids_tmax = stn_ids_tmax[np.in1d(stn_ids_tmax, stns_ids, True)]
    
    #Send stn masks to all processes
    MPI.COMM_WORLD.bcast((mask_por_tmin,mask_por_tmax), root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in np.unique(np.concatenate([stn_ids_tmin,stn_ids_tmax])):
        
        tair_vars = []
        if stn_id in stn_ids_tmin:
            tair_vars.append('tmin')
        if stn_id in stn_ids_tmax:
            tair_vars.append('tmax')
            
        for tair_var in tair_vars:
            
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send((stn_id,tair_var), dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    params[P_PATH_R_FUNCS] = '/home/jared.oyler/repos/twx/twx/lib/rpy/imputation.R'
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_MIN_POR] = 5  #2 for CCE
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_MIN_NNGH_DAILY] = 3
    #left,right,bottom,top
    #params[P_STN_LOC_BNDS] = (-118.5,-109.2,44.0,52.6) #Crown of the Continent
    params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,53.0) #CONUS
    
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()