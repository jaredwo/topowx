'''
A MPI driver for validating the prcp infilling methods of obs_infill_daily. A random set of
HCN stations spread across different ecoregions have all but a specified # 
of years of data artificially set to missing. Infilled values are then compared to observed values. 

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from db.station_data import station_data_ncdb
from utils.status_check import status_check
from netCDF4 import Dataset
import netCDF4
from utils.util_dates import YEAR,MONTH
from infill.infill_daily import calc_forecast_scores,infill_prcp,source_r
from infill.infill_normals import build_mth_masks,MTH_BUFFER,infill_prcp_norm
import matplotlib.pyplot as plt
import scipy.stats as ss
from random_xval_stations import build_xval_masks,get_random_xval_stns
from db.all_create_db import dbDataset

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

FORECAST_SCORES = ['pc','h','far','ts','b','hss']

P_PATH_DB = 'P_PATH_DB'
P_PATH_OUT = 'P_PATH_OUT'
P_PATH_POR = 'P_PATH_POR'
P_PATH_NEON = 'P_PATH_NEON'
P_PATH_NORMS = 'P_PATH_NORMS'
P_PATH_FIGS = 'P_PATH_FIGS'
P_NCDF_MODE = 'P_NCDF_MODE'

P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'

P_MIN_POR_PCT = 'P_MIN_POR_PCT'
P_STNS_PER_RGN = 'P_STNS_PER_RGN'
P_NYRS_MOD = 'P_NYRS_MOD'
P_NGH_RNG = 'P_NGH_RNG'
P_EXCLUDE_STNIDS = 'P_EXCLUDE_STNIDS'
P_INCLUDE_STNIDS = 'P_INCLUDE_STNIDS'
P_PATH_GHCN_STNS = 'P_PATH_GHCN_STNS'

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
    
    source_r()
    
    status = MPI.Status()
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    days = stn_da.days
    mth_masks = build_mth_masks(days)
    mthbuf_masks = build_mth_masks(days,MTH_BUFFER)
    yrmth_masks = build_yr_mth_masks(days)
    
    ds_prcp = Dataset(params[P_PATH_NORMS])
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp = bcast_msg
    print "".join(["Worker ",str(rank),": Received broadcast msg"])
    
    def transfunct_bin(x,params=None):
        x = np.copy(x)
        x[x > 0] = 1
        return x,params
    
    def btrans_square(x,params=None):
        
        x = np.copy(x)
        x[x < 0] = 0
        return np.square(x)
    
    ################################################
    mean_center_prcp = False
    scale_prcp = False
    trans_prcp = transfunct_bin#lambda x,params=None: (np.sqrt(x),None)#lambda x,params=None: (x,None)
    btrans_prcp = lambda x, params=None: x #btrans_square
    ################################################
    
    current_stn_id = None
    pca_cache = None
    
    while 1:
    
        stn_id,nnghs = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.send([None]*10, dest=RANK_WRITE, tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:

            try:
                
                if stn_id != current_stn_id:
                    pca_cache = None
                    current_stn_id = stn_id
                
                x = np.nonzero(stn_ids==stn_id)[0][0]
                prcp_mask = xval_masks_prcp[x]
                
                if pca_cache is not None:
                    
                    obs_po_xval,obs_prcp_xval,fnl_norm = pca_cache
                
                else:
                    
                    obs_prcp_xval = np.array(stn_da.load_all_stn_obs_var(np.array([stn_id]),'prcp')[0],dtype=np.float64)
                    obs_po_xval = np.copy(obs_prcp_xval)
                    obs_po_xval[obs_po_xval > 0] = 1
                    
                    prcp_norm,obs_norm = infill_prcp_norm(stn_id, stn_da, days, mth_masks, mthbuf_masks, prcp_mask=prcp_mask)
                
                    fnl_norm = np.copy(obs_norm)
                    fnl_norm[np.isnan(obs_norm)] = prcp_norm[np.isnan(obs_norm)]
                                        
                    pca_cache = (obs_po_xval,obs_prcp_xval,fnl_norm)
                    
                rslts = infill_prcp(stn_id, stn_da, ds_prcp, days, mth_masks, mthbuf_masks, nnghs, 
                                    prcp_trans_funcs=(trans_prcp,btrans_prcp), mean_c=mean_center_prcp, 
                                    scale=scale_prcp, prcp_mask=prcp_mask, prcp_norm=fnl_norm)
                
                max_dist = rslts.maxdist_po
                
                fit_prcp = rslts.prcp
                fit_po = np.copy(fit_prcp)
                fit_po[fit_po > 0] = 1
                                
                fnl_xval_obs_po = obs_po_xval[prcp_mask]
                fnl_xval_obs_prcp = obs_prcp_xval[prcp_mask]
                fnl_xval_fit_po = fit_po[prcp_mask]
                fnl_xval_fit_prcp = fit_prcp[prcp_mask]
                
                #pc,h,far,ts,b,hss
                scores = calc_forecast_scores(fnl_xval_obs_po, fnl_xval_fit_po)
                
                freq_obs = np.nonzero(fnl_xval_obs_po > 0)[0].size / np.float(fnl_xval_obs_po.size)
                freq_fit = np.nonzero(fnl_xval_fit_po > 0)[0].size / np.float(fnl_xval_fit_po.size)
                amt_obs = np.sum(fnl_xval_obs_prcp,dtype=np.float64)
                amt_fit = np.sum(fnl_xval_fit_prcp,dtype=np.float64)
                intsy_obs = np.sum(fnl_xval_obs_prcp[fnl_xval_obs_prcp  > 0])/fnl_xval_obs_prcp[fnl_xval_obs_prcp  > 0].size
                intsy_fit = np.sum(fnl_xval_fit_prcp[fnl_xval_fit_prcp  > 0])/fnl_xval_fit_prcp[fnl_xval_fit_prcp  > 0].size
                
                perr_intsy = ((intsy_fit-intsy_obs)/intsy_obs)*100.
                perr_freq = ((freq_fit - freq_obs) / freq_obs) * 100.
                perr_amt = (amt_fit - amt_obs)/amt_obs*100.
                
                mask_both_prcp = np.logical_and(fnl_xval_obs_po.astype(np.bool),fnl_xval_fit_po.astype(np.bool))
                
                var_obs = np.var(fnl_xval_obs_prcp[mask_both_prcp], ddof=1)
                var_fit = np.var(fnl_xval_fit_prcp[mask_both_prcp], ddof=1)
                
                mth_fit_prcp = []
                mth_obs_prcp = []
                
                for yrmth_mask in yrmth_masks:
                    
                    yrmth_xval_mask = np.logical_and(yrmth_mask,prcp_mask)
                    
                    if np.sum(yrmth_xval_mask) > 0:
                        
                        mth_fit_prcp.append(np.sum(fit_prcp[yrmth_xval_mask]))
                        mth_obs_prcp.append(np.sum(obs_prcp_xval[yrmth_xval_mask]))
                
                mth_fit_prcp = np.array(mth_fit_prcp)
                mth_obs_prcp = np.array(mth_obs_prcp)
                
                mth_corr = ss.pearsonr(mth_fit_prcp,mth_obs_prcp)[0]
                ########################################################################
                
#                ymax = np.max(np.concatenate((fnl_xval_obs_prcp,fnl_xval_fit_prcp)))
#                
#                plot_mask = np.logical_and(fnl_xval_obs_prcp > 0,fnl_xval_fit_prcp > 0)
#                
#                plt.clf()            
#                plt.subplot(211)
#                plt.plot(fnl_xval_obs_prcp[plot_mask])
#                plt.ylim((0,ymax))
#                plt.subplot(212)
#                plt.plot(fnl_xval_fit_prcp[plot_mask])
#                plt.ylim((0,ymax))
#                plt.savefig("".join([params[P_PATH_FIGS],"prcp_",stn_id,".png"]))
                
                
#                plt.clf()
#                plt.plot(fit_prcp)
#                plt.plot(obs_prcp)
#                plt.savefig("".join([params[P_PATH_FIGS],"prcp_",stn_id,".png"]))
                
                ##################################################################
                
            except Exception as e:
            
                print "".join(["ERROR: Worker ",str(rank),": could not infill po for ",stn_id," ",str(e)])
                MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
                continue
            
            MPI.COMM_WORLD.send((stn_id,scores,perr_freq,perr_amt,perr_intsy,var_obs,var_fit,mth_corr,nnghs,max_dist), dest=RANK_WRITE, tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)

def build_yr_mth_masks(days):
    
    masks = []
    for yr in np.unique(days[YEAR]):
        
        yr_mask = days[YEAR] == yr
        for mth in np.unique(days[MONTH][yr_mask]):
            
            masks.append(np.logical_and(yr_mask,days[MONTH]==mth))
    
    return masks
            
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    bcast_msg = None
    bcast_msg = MPI.COMM_WORLD.bcast(bcast_msg, root=RANK_COORD)
    stn_ids,xval_masks_prcp = bcast_msg
    print "Writer: Received broadcast msg"
        
    if params[P_NCDF_MODE] == "w":
        
        ds = create_ncdf(params,stn_ids)
        print "Writer: Output NCDF file created"
        ttl_infills = params[P_NGH_RNG].size * stn_ids.size
    
    else:
        
        ds = Dataset(params[P_PATH_OUT],"r+")
        mask_ic = ds.variables['mth_cor'][:].mask
        ttl_infills = np.sum(mask_ic)
        print "Writer: Reopened NCDF file"
    
    stn_idxs = {}
    for x in np.arange(stn_ids.size):
        stn_idxs[stn_ids[x]] = x
    
    ngh_idxs = {}
    for x in np.arange(params[P_NGH_RNG].size):
        ngh_idxs[params[P_NGH_RNG][x]] = x

    stat_chk = status_check(ttl_infills,10)
    
    while 1:
        
        stn_id,scores,perr_freq,perr_amt,perr_intsy,var_obs,var_fit,mth_corr,nnghs,max_dist = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            #Dimensions: ('nnghs','stn_id')
            dim1 = ngh_idxs[nnghs]
            dim2 = stn_idxs[stn_id]
            
            scores = np.array(scores)
            scores[np.isnan(scores)] = netCDF4.default_fillvals['f4']
            
            pc,h,far,ts,b,hss = scores
            ds.variables['pc'][dim1,dim2] = pc
            ds.variables['h'][dim1,dim2] = h
            ds.variables['far'][dim1,dim2] = far
            ds.variables['ts'][dim1,dim2] = ts
            ds.variables['b'][dim1,dim2] = b
            ds.variables['hss'][dim1,dim2] = hss
            ds.variables['freq_err'][dim1,dim2] = perr_freq
            ds.variables['amt_err'][dim1,dim2] = perr_amt
            ds.variables['intsy_err'][dim1,dim2] = perr_intsy
            ds.variables['var_obs'][dim1,dim2] = var_obs
            ds.variables['var_fit'][dim1,dim2] = var_fit
            ds.variables['mth_cor'][dim1,dim2] = mth_corr
                
            ds.sync()
            stat_chk.increment()
            print "|".join(["WRITER",stn_id,str(nnghs),str(max_dist),'%.4f'%(hss,),'%.4f'%(perr_freq,),'%.4f'%(perr_amt,),'%.4f'%(perr_intsy,),'%.4f'%(mth_corr,)])
                
def proc_coord(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    mask_ic = None
    
    if params[P_INCLUDE_STNIDS] is None:
        
        fnl_stn_ids = get_random_xval_stns(stn_da, params[P_MIN_POR_PCT], params[P_STNS_PER_RGN], params[P_PATH_POR], params[P_PATH_NEON],
                             params[P_PATH_GHCN_STNS], params[P_EXCLUDE_STNIDS])
    
        
    elif params[P_NCDF_MODE] == "r+":
        
        ds_out = Dataset(params[P_PATH_OUT])
        fnl_stn_ids = np.array(ds_out.variables['stn_id'][:],dtype="<S16")
        mask_ic = ds_out.variables['mth_cor'][:].mask
        ds_out.close()

    else:
        
        fnl_stn_ids = params[P_INCLUDE_STNIDS]
    
    
    xval_masks_prcp = build_xval_masks(fnl_stn_ids, params[P_NYRS_MOD], stn_da)
    
    #Send stn ids and masks to all processes
    MPI.COMM_WORLD.bcast((fnl_stn_ids,xval_masks_prcp), root=RANK_COORD)
    
    ngh_idxs = {}
    for x in np.arange(params[P_NGH_RNG].size):
        ngh_idxs[params[P_NGH_RNG][x]] = x
        
    stn_idxs = {}
    for x in np.arange(fnl_stn_ids.size):
        stn_idxs[fnl_stn_ids[x]] = x
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
    
    for stn_id in fnl_stn_ids:
        
        for nngh in params[P_NGH_RNG]:
            
            send_work = True
            if mask_ic is not None:
                
                dim1 = ngh_idxs[nngh]
                dim2 = stn_idxs[stn_id]
                send_work = mask_ic[dim1,dim2]
            
            if send_work:    
                if cnt < nwrkers:
                    dest = cnt+N_NON_WRKRS
                else:
                    dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    nrec+=1
    
                MPI.COMM_WORLD.send((stn_id,nngh), dest=dest, tag=TAG_DOWORK)
                cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

def create_ncdf(params,stn_ids):
    
    path = params[P_PATH_OUT]
    ngh_rng = params[P_NGH_RNG]
    
    ds = dbDataset(path,'w')
    ds.db_create_global_attributes("PRCP OCCURRENCE INFILL EXTRAPOLATION TESTING")
    
    ds.db_create_stnid_dimvar(stn_ids)
    ds.createDimension('nnghs',ngh_rng.size)
    
    nghs = ds.createVariable('nnghs','i4',('nnghs',),fill_value=False)
    nghs.long_name = "nnghs"
    nghs.standard_name = "nnghs"
    nghs[:] = ngh_rng
            
    for score in FORECAST_SCORES:
        
        score_var = ds.createVariable(score,'f4',('nnghs','stn_id'))
        score_var.long_name = score
        score_var.standard_name = score
        score_var.missing_value = netCDF4.default_fillvals['f4']
    
    score_var = ds.createVariable("freq_err",'f4',('nnghs','stn_id'))
    score_var.long_name = "frequency % error"
    score_var.standard_name = "frequency % error"
    score_var.missing_value = netCDF4.default_fillvals['f4']
    
    score_var = ds.createVariable("amt_err",'f4',('nnghs','stn_id'))
    score_var.long_name = "amount % error"
    score_var.standard_name = "amount % error"
    score_var.missing_value = netCDF4.default_fillvals['f4']
    
    score_var = ds.createVariable("intsy_err",'f4',('nnghs','stn_id'))
    score_var.long_name = "intensity % error"
    score_var.standard_name = "intensity % error"
    score_var.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable('var_obs','f4',('nnghs','stn_id'))
    avar.long_name = "observed variance"
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable('var_fit','f4',('nnghs','stn_id'))
    avar.long_name = "fit variance"
    avar.missing_value = netCDF4.default_fillvals['f4']
    
    avar = ds.createVariable('mth_cor','f4',('nnghs','stn_id'))
    avar.long_name = "monthly correlation coefficient"
    avar.missing_value = netCDF4.default_fillvals['f4']

    ds.sync()
    
    return ds

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all.nc'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/infill/xval_infill_prcp.nc'
    params[P_PATH_FIGS] = '/projects/daymet2/station_data/infill/figs/'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por.csv'
    params[P_PATH_NEON] =  '/projects/daymet2/dem/NEON_DOMAINS/neon_mask3.nc'
    params[P_PATH_NORMS] =  '/projects/daymet2/station_data/infill/normals_prcp.nc'
    params[P_PATH_GHCN_STNS] = '/projects/daymet2/station_data/ghcn/ghcnd-stations.txt'
    params[P_MIN_POR_PCT] = 0.90
    params[P_STNS_PER_RGN] = 10
    params[P_NYRS_MOD] = 5
    params[P_NGH_RNG] = np.arange(5,41)
    params[P_START_YMD] = None #19480101
    params[P_END_YMD] = None #20111231
    params[P_NCDF_MODE] = 'r+' #w or r+
    
    if rank == RANK_COORD:
        
        params[P_EXCLUDE_STNIDS] = np.array([])
        #params[P_INCLUDE_STNIDS] = np.array(['GHCN_USC00357169','GHCN_USC00489770'])#None
        
        ds = Dataset('/projects/daymet2/station_data/infill/xval_infill_po.nc')
        params[P_INCLUDE_STNIDS] = np.array(ds.variables['stn_id'][:],dtype="<S16")
        ds.close()
        
        proc_coord(params, nsize-N_NON_WRKRS)
        
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
