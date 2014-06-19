'''
A MPI driver for performing "leave one out" cross-validation of po interpolation in interp_prcp

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,STN_ID,LON,LAT
from twx.interp.station_select import station_select
from twx.utils.status_check import status_check
import twx.utils.util_geo as utlg
from twx.utils.ncdf_raster import ncdf_raster
from netCDF4 import Dataset
import netCDF4
import datetime
import twx.interp.interp_prcp as ip

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_NEON = 'P_PATH_NEON'
P_PATH_DB_XVAL = 'P_PATH_DB_XVAL'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_STNS_PER_RGN = 'P_STNS_PER_RGN'
P_NGH_RNG = 'P_NGH_RNG'
P_EXCLUDE_STNIDS = 'P_EXCLUDE_STNIDS'
P_INCLUDE_STNIDS = 'P_INCLUDE_STNIDS'
P_VARNAME = 'P_VARNAME'
P_PO_THRES = "P_PO_THRES"

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def calc_hss(obs_po, mod_po):
    '''
    Calculates heidke skill score of modeled prcp occurrence
    See http://www.wxonline.info/topics/verif2.html
    @param obs: array of observed occurrences (1s and 0s)
    @param mod: array of modeled occurrences (1s and 0s)
    @return hss: heidke skill score
    '''
    
    #model_obs
    true_true = mod_po[np.logical_and(obs_po == 1, mod_po == 1)].size
    false_false = mod_po[np.logical_and(obs_po == 0, mod_po == 0)].size
    true_false = mod_po[np.logical_and(obs_po == 0, mod_po == 1)].size
    false_true = mod_po[np.logical_and(obs_po == 1, mod_po == 0)].size
    
    a = float(true_true)
    b = float(true_false)
    c = float(false_true)
    d = float(false_false)
    
    #special case handling
    if a == 0.0 and c == 0.0 and b != 0:
        #This means that were no observed days of rain so can't calc
        #appropriate hss. Set a = 1 to get a more appropriate hss
        a = 1.0
    
    if b == 0.0 and d == 0.0 and c != 0.0:
        #This means that there was observed rain every day so can't calc
        #appropriate hss. Set d = 1 to get a more appropriate hss
        d = 1.0    

    den = ((a + c) * (c + d)) + ((a + b) * (b + d))
    
    if den == 0.0:
        #This is a perfect forecast with all true_true or false_false
        return 1.0
    
    return (2.0 * ((a * d) - (b * c))) / den

def proc_work(params,rank):
    
    status = MPI.Status()
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    
    mod = ip.modeler_clib_po()
    po_interper = ip.interp_po(mod)
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)    
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    while 1:
    
        stn_id,min_ngh = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*3, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            try:
                
                stn_slct = station_select(stn_da.stns,min_ngh,min_ngh+10)
            
                xval_stn = stn_da.stns[stn_da.stn_idxs[stn_id]]
                xval_obs = stn_da.load_obs(stn_id)
                xval_obs[xval_obs > 0] = 1
            
                dist = utlg.grt_circle_dist(xval_stn[LON], xval_stn[LAT], stn_da.stns[LON], stn_da.stns[LAT])
                rm_stn_ids = np.unique(np.concatenate((np.array([stn_id]),stn_da.stn_ids[dist==0])))
            
                stns,wgts,rad = stn_slct.get_interp_stns(xval_stn[LAT], xval_stn[LON], rm_stn_ids)
                
                stn_obs = stn_da.load_obs(stns[STN_ID])
                stn_obs[stn_obs > 0] = 1
                
                po = po_interper.model_po(stns, wgts, stn_obs, xval_stn,params[P_PO_THRES])
                
                hss = calc_hss(xval_obs, po)
            
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not xval ",stn_id," with ",str(min_ngh)," nghs...",str(e)])
                
                hss = netCDF4.default_fillvals['f8']
            
            MPI.COMM_WORLD.send((stn_id,min_ngh,hss), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                
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
    
    ngh_idxs = {}
    for x in np.arange(params[P_NGH_RNG].size):
        ngh_idxs[params[P_NGH_RNG][x]] = x
    
    ttl_xvals = params[P_NGH_RNG].size * stn_ids.size
    
    stat_chk = status_check(ttl_xvals,1000)
    
    while 1:
       
        stn_id,min_ngh,hss = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            dim1 = ngh_idxs[min_ngh]
            dim2 = stn_idxs[stn_id]
            
            ds.variables['hss'][dim1,dim2] = hss
            ds.sync()
            
            #print "|".join(["WRITER",stn_id,str(min_ngh),"%.4f"%(hss,)])
            
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    
    if params[P_INCLUDE_STNIDS] is None:
        
        #Load the neon ecoregion raster into memory
        neon_rast = ncdf_raster(params[P_PATH_NEON], 'neon')
        
        nodata_mask = neon_rast.vals.mask
        neon_rast.vals = np.array(neon_rast.vals.data,dtype=np.float32)
        neon_rast.vals[nodata_mask] = np.nan
        neon = neon_rast.vals
        uniq_rgns = np.unique(neon[np.isfinite(neon)])
    
        #U.S. only mask
        mask_us = np.logical_and(np.char.find(stn_da.stns[STN_ID],'GHCN_CA')==-1,np.char.find(stn_da.stns[STN_ID],'GHCN_MX')==-1)
    
        #Extract lons, lats, and stn_ids
        lons = stn_da.stns[LON][mask_us]
        lats = stn_da.stns[LAT][mask_us]
        stn_ids = stn_da.stns[STN_ID][mask_us]
        
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
            
            if stn_ids_rgn.size < params[P_STNS_PER_RGN]:
                raise Exception("".join(["NEON ",str(rgn)," has ",str(stn_ids_rgn.size)," stns but requested ",str(params[P_STNS_PER_RGN])," random stns."]))
                
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
        fnl_stn_ids = np.sort(fnl_stn_ids)
    
    else:
        
        fnl_stn_ids = params[P_INCLUDE_STNIDS]
    
    #Send stn ids to all processes
    MPI.COMM_WORLD.bcast(fnl_stn_ids, root=RANK_COORD)
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
        
        for min_ngh in params[P_NGH_RNG]:
                
            if cnt < nwrkers:
                dest = cnt+N_NON_WRKRS
            else:
                dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                nrec+=1

            MPI.COMM_WORLD.send((stn_id,min_ngh), dest=dest, tag=TAG_DOWORK)
            cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_ncdf(params,stn_ids):
    
    path = params[P_PATH_OUT]
    min_ngh_rng = params[P_NGH_RNG]
    
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
    
    dim_ngh = ncdf_file.createDimension('min_nghs',min_ngh_rng.size)
    dim_station = ncdf_file.createDimension('stn_id',stn_ids.size)
    
    nghs = ncdf_file.createVariable('min_nghs','i4',('min_nghs',),fill_value=False)
    nghs.long_name = "min_nghs"
    nghs.standard_name = "min_nghs"
    nghs[:] = min_ngh_rng
    
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    mae_var = ncdf_file.createVariable('hss','f8',('min_nghs','stn_id'))
    mae_var.long_name = "hss"
    mae_var.standard_name = "hss"
    mae_var.missing_value = netCDF4.default_fillvals['f8']
    
    ncdf_file.sync()
    
    return ncdf_file

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/infill/infill_prcp.nc'
    #params[P_PATH_DB_XVAL] = '/projects/daymet2/station_data/infill/infill_prcp.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/xval_po.nc'
    params[P_PATH_NEON] = '/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc'
    params[P_STNS_PER_RGN] = 100
    params[P_NGH_RNG] = np.arange(10,76)
    params[P_VARNAME] = 'prcp' 
    params[P_PO_THRES] = 0.5
    
    if rank == RANK_COORD:
        
        params[P_EXCLUDE_STNIDS] = np.array([])
        stn_da = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
        
        #U.S. only mask
        mask_us = np.logical_and(np.char.find(stn_da.stns[STN_ID],'GHCN_CA')==-1,np.char.find(stn_da.stns[STN_ID],'GHCN_MX')==-1)
        
        params[P_INCLUDE_STNIDS] = stn_da.stns[STN_ID][mask_us]
        stn_da = None
        proc_coord(params, nsize-N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
