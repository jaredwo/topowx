'''
A MPI driver for validating the prcp occurrence infilling methods of obs_infill_normal. A random set of
HCN stations spread across different ecoregions have all but a specified # 
of years of data artificially set to missing. Infilled values are then compared to observed values. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationDataDb,STN_ID,LON,LAT
from twx.infill.obs_por import load_por_csv,POR_DTYPE
from twx.utils.input_raster import input_raster
from twx.utils.status_check import StatusCheck
from netCDF4 import Dataset
import netCDF4
import datetime
from twx.infill.infill_normals import build_mth_masks,MTH_BUFFER,infill_po
from twx.infill.infill_daily import calc_forecast_scores,calc_hss

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

FORECAST_SCORES = ['pc','h','far','ts','b','hss']
PO_THRESHS = np.arange(0.1,0.91,.01)

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NEON = 'P_PATH_NEON'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_MIN_POR_PCT = 'P_MIN_POR_PCT'
P_STNS_PER_RGN = 'P_STNS_PER_RGN'
P_NYRS_MOD = 'P_NYRS_MOD'
P_EXCLUDE_STNIDS = 'P_EXCLUDE_STNIDS'
P_INCLUDE_STNIDS = 'P_INCLUDE_STNIDS'

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
    stn_da = StationDataDb(params[P_PATH_DB])
    stn_da.set_day_mask(params[P_START_YMD],params[P_END_YMD])
    days = stn_da.days[stn_da.day_mask]
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    po = np.zeros(days.size)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp,last_idxs_prcp = bcast_msg
    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:

            try:
                
                x = np.nonzero(stn_ids==stn_id)[0][0]
                prcp_mask = xval_masks_prcp[x]
                last_idxs = last_idxs_prcp[x]
                    
                obs_po = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]),'prcp')[0],dtype=np.float64)
                obs_po[obs_po > 0] = 1
                
                fit_po = infill_po(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask)[0]
                
                mean_fit = np.mean(fit_po[prcp_mask])
                mean_obs = np.mean(obs_po[prcp_mask])
                    
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill po for ",stn_id,str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send((stn_id,mean_fit,mean_obs), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp,last_idxs_prcp = bcast_msg
    print "Writer: Received broadcast msg"
    
    #ds = create_ncdf(params,stn_ids)
    #print "Writer: Output NCDF file created"
    
#    stn_idxs = {}
#    for x in np.arange(stn_ids.size):
#        stn_idxs[stn_ids[x]] = x
    
    ttl_infills = stn_ids.size
    
    stat_chk = StatusCheck(ttl_infills,10)
    
    errs = []
    
    while 1:
       
        stn_id,mean_fit,mean_obs = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                print "####################################"
                print "ERROR SUMMARY"
                print "####################################"
                print "MAE: ",np.mean(np.abs(errs))
                print "BIAS: ",np.mean(errs)
                print "####################################"
                
                print "Writer: Finished"
                return 0
        else:
            
            dif = mean_fit - mean_obs
            errs.append(dif)
            
#            y = stn_idxs[stn_id]
#            
#            scores = np.array(scores)
#            scores[np.isnan(scores)] = netCDF4.default_fillvals['f4']
#            
#            pc,h,far,ts,b,hss = scores
#            
#            ds.variables['pc'][y] = pc
#            ds.variables['h'][y] = h
#            ds.variables['far'][y] = far
#            ds.variables['ts'][y] = ts
#            ds.variables['b'][y] = b
#            ds.variables['hss'][y] = hss
#            ds.variables['po_thresh'][y] = max_thres
#                
#            ds.sync()
            stat_chk.increment()
            print "|".join(["WRITER",stn_id,'%.4f'%(mean_fit,),'%.4f'%(mean_obs,),'%.4f'%(dif,)])
                
def proc_coord(params,nwrkers):
    
    stn_da = StationDataDb(params[P_PATH_DB])
    stn_da.set_day_mask(params[P_START_YMD],params[P_END_YMD])
    days = stn_da.days[stn_da.day_mask]
    
    #The number of observations that should not be set to nan
    #and are used to build the infill model
    nmask = int(np.round(params[P_NYRS_MOD]*365.25))
    
    #The min number of observations at station must have to be considered
    #for extrapolation testing
    min_obs = np.round(params[P_MIN_POR_PCT]*days.size)
    
    if params[P_INCLUDE_STNIDS] is None:
        #Load the period-of-record datafile
        por = load_por_csv(params[P_PATH_POR])
        
        #Number of obs in period-of-record for each month and variable
        por_cols = np.array([x[0] for x in POR_DTYPE])
        cols_prcp = por_cols[30:]
        por_prcp = por[cols_prcp].view(np.int32).reshape((por.size,cols_prcp.size))
        
        #Mask stations that have the min number of observations for prcp
        mask_prcp = np.sum(por_prcp,axis=1)>=min_obs
        mask_all = mask_prcp
        
        #Load the neon ecoregion raster into memory
        neon_rast = input_raster(params[P_PATH_NEON])
        neon = neon_rast.readEntireRaster()
        neon = np.array(neon,dtype=np.float32)
        neon[neon==255] = np.nan
        uniq_rgns = np.unique(neon[np.isfinite(neon)])
    
        #Extract lons, lats, and stn_ids that have min # of observations
        lons = por[LON][mask_all]
        lats = por[LAT][mask_all]
        stn_ids = por[STN_ID][mask_all]
        
        #Only use HCN stations
        hcn_mask = np.zeros(stn_ids.size,dtype=np.bool)
        hcn_dict = get_hcn_dict()
        
        for x in np.arange(stn_ids.size):
        
            if hcn_dict[stn_ids[x]] == "HCN":
                hcn_mask[x] = True
        
        lons = lons[hcn_mask]
        lats = lats[hcn_mask]
        stn_ids = stn_ids[hcn_mask]
        
        #Determine neon region for each station
        rgns = np.zeros(lons.size)
        for x in np.arange(lons.size):
            try:
                rgns[x] = neon_rast.getDataValue(lons[x],lats[x])
            except:
                rgns[x] = np.nan
                
        #Pick a specific number of random stations for each region
        fnl_stn_ids = []
        for rgn in uniq_rgns:
            
            stn_ids_rgn = stn_ids[rgns==rgn]
            
            rndm_stnids = stn_ids_rgn[np.random.randint(0,stn_ids_rgn.size,params[P_STNS_PER_RGN])]
            rndm_stnids = rndm_stnids[np.logical_not(np.in1d(rndm_stnids,params[P_EXCLUDE_STNIDS],assume_unique=True))]
            rndm_stnids = np.unique(rndm_stnids)
            
            while rndm_stnids.size < params[P_STNS_PER_RGN]:
                
                temp_rndm_stnids = stn_ids_rgn[np.random.randint(0,stn_ids_rgn.size,params[P_STNS_PER_RGN])]
                rndm_stnids = np.unique(np.concatenate([rndm_stnids,temp_rndm_stnids]))
                rndm_stnids = rndm_stnids[np.logical_not(np.in1d(rndm_stnids,params[P_EXCLUDE_STNIDS],assume_unique=True))]
            
            rndm_stnids = rndm_stnids[0:params[P_STNS_PER_RGN]]
            
            fnl_stn_ids.extend(rndm_stnids)
        
        #Make sure fnl_stn_ids is in right order for loading observation data
        fnl_stn_ids = np.array(fnl_stn_ids)
        fnl_stn_ids = stn_da.stn_ids[np.in1d(stn_da.stn_ids, fnl_stn_ids, assume_unique=True)]
    
    else:
        
        fnl_stn_ids = params[P_INCLUDE_STNIDS]
    
    #Load prcp observations for each station
    prcp = stn_da.load_all_stn_obs_var(fnl_stn_ids,'prcp')[0]
    
    #Build masks of the data values that should be set to nan for each station
    #and then set the data values to nan
    xval_masks_prcp = []
    last_idxs_prcp = []
    
    idxs = np.arange(days.size)
    
    for x in np.arange(fnl_stn_ids.size):
        
        fin_prcp = np.isfinite(prcp[:,x])
        
        last_idxs = np.nonzero(fin_prcp)[0][-nmask:]
        xval_mask_prcp = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_prcp)
        xval_masks_prcp.append(xval_mask_prcp)
        last_idxs_prcp.append(last_idxs)
        
        prcp_stn = prcp[:,x]
        
        prcp_stn[xval_mask_prcp] = np.nan
        
        prcp[:,x] = prcp_stn
    
    #Send stn ids and masks to all processes
    MPI.COMM_WORLD.bcast((fnl_stn_ids,xval_masks_prcp,last_idxs_prcp), root=RANK_COORD)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
                
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
    title = "PRCP OCCURRENCE INFILL EXTRAPOLATION TESTING"
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    ncdf_file.comments = cmts
    
    dim_station = ncdf_file.createDimension('stn_id',stn_ids.size)
    
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    po_threshs = ncdf_file.createVariable('po_thresh','f4',('stn_id',))
    po_threshs.long_name = "prcp probability threshold"
    po_threshs.standard_name = "po_thresh"
    po_threshs.missing_value = netCDF4.default_fillvals['f4']
    
    for score in FORECAST_SCORES:
        
        score_var = ncdf_file.createVariable(score,'f4',('stn_id'))
        score_var.long_name = score
        score_var.standard_name = score
        score_var.missing_value = netCDF4.default_fillvals['f4']

    
    ncdf_file.sync()
    
    return ncdf_file

def get_hcn_dict():
    
    afile = open('/projects/daymet2/station_data/ghcn/ghcnd-stations.txt')
    hcn_dict = {}
    
    for line in afile.readlines():
        
        stn_id = "".join(["GHCN_",line[0:11].strip()])
        hcn_flg = line[76:79]
        hcn_dict[stn_id] = hcn_flg
        
    return hcn_dict

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    #params[P_PATH_OUT] = '/projects/daymet2/station_data/all/extrap_stats_wgtavg_po.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_NEON] = '/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.tif'
    params[P_MIN_POR_PCT] = 0.90
    params[P_STNS_PER_RGN] = 10
    params[P_NYRS_MOD] = 5
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20111231
    
    if rank == RANK_COORD:
        
        params[P_EXCLUDE_STNIDS] = np.array([])
        ds = Dataset('/projects/daymet2/station_data/all/extrap_stats_wgtavg_po.nc')
        params[P_INCLUDE_STNIDS] = np.array(ds.variables['stn_id'][:],dtype="<S16")
        ds.close()
        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()