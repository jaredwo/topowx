'''
A MPI driver for validating the infilling methods of obs_infill_daily. A random set of
HCN stations spread across different ecoregions have all but a specified # 
of years of data artificially set to missing. Infilled values are then compared to observed values. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import station_data_ncdb,STN_ID,LON,LAT
from infill_daily import source_r
from twx.infill.obs_por import load_por_csv,POR_DTYPE,build_valid_por_masks
from twx.utils.ncdf_raster import ncdf_raster
from twx.utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
import datetime
from twx.infill.infill_normals import impute_tair_norm,infill_tair,MTH_BUFFER,build_mth_masks
from twx.db.reanalysis import NNRNghData
from httplib import HTTPException
from twx.infill.random_xval_stations import XvalStnsTairSnotelRaws
from twx.interp.clibs import clib_wxTopo

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_RPCA = 'P_PATH_RPCA'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NEON = 'P_PATH_NEON'
P_PATH_NORMALS = 'P_PATH_NORMALS'
P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
P_PATH_NNR = 'P_PATH_NNR'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_MIN_POR_PCT = 'P_MIN_POR_PCT'
P_STNS_PER_RGN = 'P_STNS_PER_RGN'
P_NYRS_MOD = 'P_NYRS_MOD'
P_NGH_RNG = 'P_NGH_RNG'
P_EXCLUDE_STNIDS = 'P_EXCLUDE_STNIDS'
P_INCLUDE_STNIDS = 'P_INCLUDE_STNIDS'
P_NCDF_MODE = 'P_NCDF_MODE'
P_MIN_POR = 'P_MIN_POR'
P_STN_LOC_BNDS = 'P_STN_LOC_BNDS'
P_SNOTEL_RAWS_XVAL = 'P_SNOTEL_RAWS_XVAL'

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
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    source_r(params[P_PATH_R_FUNCS])

    ds_nnr = NNRNghData(params[P_PATH_NNR], (params[P_START_YMD],params[P_END_YMD]))
        
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    
    stn_ids,xval_masks_tmin,xval_masks_tmax,mask_por_tmin,mask_por_tmax = bcast_msg
    xval_masks = {'tmin':xval_masks_tmin,'tmax':xval_masks_tmax}
    stn_masks = {'tmin':mask_por_tmin,'tmax':mask_por_tmax}
    
    aclib = clib_wxTopo('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C')
    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
        
    while 1:
    
        stn_id,nnghs,tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*6, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                x = np.nonzero(stn_ids==stn_id)[0][0]
                tair_mask = xval_masks[tair_var][x]
                
                #mth_masks = build_mth_masks(stn_da.days)
                #mthbuf_masks = build_mth_masks(stn_da.days,MTH_BUFFER)
                #imp_mean = infill_tair(stn_id, stn_da, stn_da.days, tair_var,stn_masks[tair_var], mth_masks, mthbuf_masks, tair_mask)
                #imp_var = 0

                imp_mean,imp_var = impute_tair_norm(stn_id, stn_da, stn_masks[tair_var],tair_var,ds_nnr,
                                                    aclib,tair_mask=tair_mask,nnghs=nnghs)[0]
                
                obs_tair = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var)[0],dtype=np.float64)
                
                fin_mask = np.isfinite(obs_tair)
                obs_mean,obs_var = np.mean(obs_tair[fin_mask]),np.var(obs_tair[fin_mask], ddof=1)
                
                bias = imp_mean-obs_mean
                mae = np.abs(bias)
                #var_pct = (imp_var/obs_var)*100.
                var_pct = ((imp_var - obs_var)/obs_var)*100
            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill ",
                               tair_var," for ",stn_id," with ",str(nnghs)," nghs...",str(e)])
                
                bias = netCDF4.default_fillvals['f4']
                mae = netCDF4.default_fillvals['f4']
                var_pct = netCDF4.default_fillvals['f4']
            
            MPI.COMM_WORLD.send((stn_id,tair_var,nnghs,mae,bias,var_pct), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_tmin,xval_masks_tmax,mask_por_tmin,mask_por_tmax = bcast_msg
    print "Writer: Received broadcast msg"
    
    ds = create_ncdf(params,stn_ids)
    print "Writer: Output NCDF file created"
    ttl_infills = params[P_NGH_RNG].size * stn_ids.size * 2
            
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    ngh_idxs = {}
    for x in np.arange(params[P_NGH_RNG].size):
        ngh_idxs[params[P_NGH_RNG][x]] = x
        
    tair_idxs = {'tmin':0,'tmax':1}
    
    stat_chk = status_check(ttl_infills,10)
    
    rslts = {'tmin':([],[],[]),'tmax':([],[],[])}
    
    
    while 1:
       
        stn_id,tair_var,nnghs,mae,bias,var_pct = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                print "|".join(["OVERALL TMIN",str(np.mean(rslts['tmin'][0])),
                                str(np.mean(rslts['tmin'][1])),
                                str(np.mean(np.abs(rslts['tmin'][2])))])
                
                print "|".join(["OVERALL TMAX",str(np.mean(rslts['tmax'][0])),
                                str(np.mean(rslts['tmax'][1])),
                                str(np.mean(np.abs(rslts['tmax'][2])))])
                
                print "Writer: Finished"
                return 0
        else:
            
            dim1 = tair_idxs[tair_var]
            dim2 = ngh_idxs[nnghs]
            dim3 = stn_idxs[stn_id]
            
            print "|".join(["WRITER",stn_id,tair_var,str(nnghs),"%.4f"%(mae,),"%.4f"%(bias,),"%.4f"%(var_pct,)])
            
            ds.variables['mae'][dim1,dim2,dim3] = mae
            ds.variables['bias'][dim1,dim2,dim3] = bias
            ds.variables['var_pct'][dim1,dim2,dim3] = var_pct
            ds.sync()
            
            rslts[tair_var][0].append(mae)
            rslts[tair_var][1].append(bias)
            rslts[tair_var][2].append(var_pct)
            
            stat_chk.increment()

def proc_coord_sntl_raws(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,params[P_MIN_POR],params[P_STN_LOC_BNDS])[0:2]
    
    xval_stns = XvalStnsTairSnotelRaws(stn_da, params[P_NYRS_MOD],params[P_MIN_POR_PCT], por, mask_por_tmin, mask_por_tmax)
    
    #Send stn ids and masks to all processes
    MPI.COMM_WORLD.bcast((xval_stns.stn_ids,xval_stns.xval_masks_tmin,xval_stns.xval_masks_tmax,
                          mask_por_tmin,mask_por_tmax), root=RANK_COORD)
    
    fnl_stn_ids = xval_stns.stn_ids
            
    print "SNOTEL/RAWS Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
        
        for min_ngh in params[P_NGH_RNG]:
            
            for tair_var in ['tmin','tmax']:
                    
                if cnt < nwrkers:
                    dest = cnt+N_NON_WRKRS
                else:
                    dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    nrec+=1

                MPI.COMM_WORLD.send((stn_id,min_ngh,tair_var), dest=dest, tag=TAG_DOWORK)
                cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"
    
def proc_coord(params,nwrkers):
    
    if params[P_SNOTEL_RAWS_XVAL]:
        proc_coord_sntl_raws(params, nwrkers)
    else:
        proc_coord_hcn(params, nwrkers)

def proc_coord_hcn(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    
    #The number of observations that should not be set to nan
    #and are used to build the infill model
    nmask = int(np.round(params[P_NYRS_MOD]*365.25))
    
    #The min number of observations at station must have to be considered
    #for extrapolation testing
    min_obs = np.round(params[P_MIN_POR_PCT]*days.size)
    
    mask_ic = None
    tair_idxs = {'tmin':0,'tmax':1}
    ngh_idxs = {}
    for x in np.arange(params[P_NGH_RNG].size):
        ngh_idxs[params[P_NGH_RNG][x]] = x
    
    if params[P_INCLUDE_STNIDS] is None:
        #Load the period-of-record datafile
        por = load_por_csv(params[P_PATH_POR])
        
        #Number of obs in period-of-record for each month and variable
        por_cols = np.array([x[0] for x in POR_DTYPE])
        cols_tmin = por_cols[6:18]
        cols_tmax = por_cols[18:30]
        por_tmin = por[cols_tmin].view(np.int32).reshape((por.size,cols_tmin.size))
        por_tmax = por[cols_tmax].view(np.int32).reshape((por.size,cols_tmax.size))
        
        #Mask stations that have the min number of observations for both tmin and tmax
        mask_tmin = np.sum(por_tmin,axis=1)>=min_obs
        mask_tmax = np.sum(por_tmax,axis=1)>=min_obs
        mask_all = np.logical_and(mask_tmax,mask_tmin)
        
        #Load the neon ecoregion raster into memory
        neon_rast = ncdf_raster(params[P_PATH_NEON], 'neon')
        nodata_mask = neon_rast.vals.mask
        neon_rast.vals = np.array(neon_rast.vals.data,dtype=np.float32)
        neon_rast.vals[nodata_mask] = np.nan
        neon = neon_rast.vals
        uniq_rgns = np.unique(neon[np.isfinite(neon)])
    
        #Extract lons, lats, and stn_ids that have min # of observations
        lons = por[LON][mask_all]
        lats = por[LAT][mask_all]
        stn_ids = por[STN_ID][mask_all]
        
        #Only use HCN stations
        hcn_mask = np.zeros(stn_ids.size,dtype=np.bool)
        hcn_dict = get_hcn_dict()
        
        for x in np.arange(stn_ids.size):
            
            try:
                if hcn_dict[stn_ids[x]] == "HCN":
                    hcn_mask[x] = True
            except KeyError:
                pass
        
        lons = lons[hcn_mask]
        lats = lats[hcn_mask]
        stn_ids = stn_ids[hcn_mask]
        #np.savetxt('/projects/daymet2/station_data/ghcn/hcnXvalStns.txt', stn_ids,fmt="%s")
        
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
            print rgn
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
    
    #Load tmin and tmax observations for each station
    tmin = stn_da.load_all_stn_obs_var(fnl_stn_ids,'tmin')[0]
    tmax = stn_da.load_all_stn_obs_var(fnl_stn_ids,'tmax')[0]
    
    #Build masks of the data values that should be set to nan for each station
    #and then set the data values to nan
    xval_masks_tmin = []
    xval_masks_tmax = []
    
    idxs = np.arange(days.size)
    
    for x in np.arange(fnl_stn_ids.size):
        
        fin_tmin = np.isfinite(tmin[:,x])
        fin_tmax = np.isfinite(tmax[:,x])
        
        last_idxs = np.nonzero(fin_tmin)[0][-nmask:]
        xval_mask_tmin = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmin)
        xval_masks_tmin.append(xval_mask_tmin)
        
        last_idxs = np.nonzero(fin_tmax)[0][-nmask:]
        xval_mask_tmax = np.logical_and(np.logical_not(np.in1d(idxs,last_idxs,assume_unique=True)),fin_tmax)
        xval_masks_tmax.append(xval_mask_tmax)
        
        tmin_stn = tmin[:,x]
        tmax_stn = tmax[:,x]
        
        tmin_stn[xval_mask_tmin] = np.nan
        tmax_stn[xval_mask_tmax] = np.nan
        
        tmin[:,x] = tmin_stn
        tmax[:,x] = tmax_stn
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por,params[P_MIN_POR],params[P_STN_LOC_BNDS])[0:2]
    
    #Send stn ids and masks to all processes
    MPI.COMM_WORLD.bcast((fnl_stn_ids,xval_masks_tmin,xval_masks_tmax,mask_por_tmin,mask_por_tmax), root=RANK_COORD)
    
    stn_idxs = {}
    for x in np.arange(fnl_stn_ids.size):
        stn_idxs[fnl_stn_ids[x]] = x
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
        
        for min_ngh in params[P_NGH_RNG]:
            
            for tair_var in ['tmin','tmax']:
                
                send_work = True
                if mask_ic is not None:
                    
                    dim1 = tair_idxs[tair_var]
                    dim2 = ngh_idxs[min_ngh]
                    dim3 = stn_idxs[stn_id]
                    send_work = mask_ic[dim1,dim2,dim3]
                    
                if send_work:
                    if cnt < nwrkers:
                        dest = cnt+N_NON_WRKRS
                    else:
                        dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                        nrec+=1

                    MPI.COMM_WORLD.send((stn_id,min_ngh,tair_var), dest=dest, tag=TAG_DOWORK)
                    cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_ncdf(params,stn_ids):
    
    path = params[P_PATH_OUT]
    min_ngh_rng = params[P_NGH_RNG]
    
    ncdf_file = Dataset(path,'w')
        
    #Set global attributes
    title = "IMPUTE MEAN/VARIANCE X-VALIDATION"
    ncdf_file.title = title
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    ncdf_file.history = "".join(["Created on: ",datetime.datetime.strftime(datetime.date.today(),"%Y-%m-%d")]) 
    
    dim_ngh = ncdf_file.createDimension('min_nghs',min_ngh_rng.size)
    dim_station = ncdf_file.createDimension('stn_id',stn_ids.size)
    dim_tair = ncdf_file.createDimension('tair_var',2)
    
    nghs = ncdf_file.createVariable('min_nghs','i4',('min_nghs',),fill_value=False)
    nghs.long_name = "min_nghs"
    nghs.standard_name = "min_nghs"
    nghs[:] = min_ngh_rng
    
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    tair_var = ncdf_file.createVariable('tair_var','str',('tair_var',))
    tair_var.long_name = "tair_var"
    tair_var.standard_name = "tair_var"
    tair_var[0] = 'tmin'
    tair_var[1] = 'tmax'
    
    mae_var = ncdf_file.createVariable('mae','f4',('tair_var','min_nghs','stn_id'))
    mae_var.long_name = "mae"
    mae_var.standard_name = "mae"
    mae_var.missing_value = netCDF4.default_fillvals['f4']
    
    bias_var = ncdf_file.createVariable('bias','f4',('tair_var','min_nghs','stn_id'))
    bias_var.long_name = "bias"
    bias_var.standard_name = "bias"
    bias_var.missing_value = netCDF4.default_fillvals['f4']
    
    var_pct_var = ncdf_file.createVariable('var_pct','f4',('tair_var','min_nghs','stn_id'))
    var_pct_var.long_name = "var_pct"
    var_pct_var.standard_name = "var_pct"
    var_pct_var.missing_value = netCDF4.default_fillvals['f4']
    
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
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/xval_impute_norm.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    params[P_PATH_NEON] =  '/projects/daymet2/dem/fwpusgs_lcc_impxval.nc'
    params[P_PATH_R_FUNCS] = '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/imputation.R'
    params[P_PATH_NNR] = '/projects/daymet2/reanalysis_data/conus_subset/'
    params[P_MIN_POR_PCT] = 0.95#0.31#0.31#0.40 #0.90 for HCN stations
    params[P_STNS_PER_RGN] = 10
    params[P_NYRS_MOD] = 5
    params[P_MIN_POR] = 5  #2 for CCE
    params[P_NGH_RNG] = np.array([3])
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,50.0) #CONUS
    params[P_SNOTEL_RAWS_XVAL] = False
    
    if rank == RANK_COORD:
        
        params[P_EXCLUDE_STNIDS] = np.array([])
        params[P_INCLUDE_STNIDS] = None
        
        params[P_INCLUDE_STNIDS] = np.unique(np.loadtxt('/projects/daymet2/station_data/ghcn/hcnXvalStns.txt',dtype=np.str))
        #ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_norm.nc')
        #params[P_INCLUDE_STNIDS] = np.array(ds.variables['stn_id'][:],dtype="<S16")
        #ds.close()

                
#        ds = Dataset('/projects/daymet2/station_data/infill/xval_impute_norm_tair.nc')
#        params[P_INCLUDE_STNIDS] = np.array(ds.variables['stn_id'][:],dtype="<S16")
#        ds.close()
        
        #params[P_INCLUDE_STNIDS] = np.sort(np.loadtxt('/projects/daymet2/station_data/cce/stn_ids_xval.txt',dtype="<S16"))    
        #params[P_INCLUDE_STNIDS] = np.sort(np.array(['RAWS_MGIR','RAWS_MHHS','RAWS_IPOW']))
        
        proc_coord(params, nsize-N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()