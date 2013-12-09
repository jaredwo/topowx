'''
MPI driver for running quality assurance procedures.

@author: jared.oyler
'''

from mpi4py import MPI
import numpy as np
from twx.db.station_data import station_data_ncdb,STATE
from netCDF4 import Dataset
from twx.utils.util_misc import Unbuffered
import sys
from twx.utils.status_check import status_check
from twx.infill.obs_por import build_valid_por_masks,load_por_csv
import twx.qa.qa_change_point as qcp

import rpy2.robjects as robjects
r = robjects.r

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = 'P_PATH_DB'
P_SPATIAL_QA = 'P_SPATIAL_QA'
P_START_YMD = 'P_START_YMD'
P_END_YMD = 'P_END_YMD'
P_STN_MASKS = 'P_STN_MASKS'
P_PATH_POR = 'P_PATH_POR'
P_PATH_R_FUNCS = 'P_PATH_R_FUNCS'
P_PATH_OUT = 'P_PATH_OUT'

sys.stdout=Unbuffered(sys.stdout)
     
def proc_work(params,rank):
    
    status = MPI.Status()
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_mask,mask_por_tmin,mask_por_tmax = params[P_STN_MASKS]
    
    #chg_pt = qcp.ChgPtMthly(stn_da, mask_por_tmin, mask_por_tmax)
    chg_pt = qcp.ChgPtDaily(stn_da, mask_por_tmin, mask_por_tmax)
    
    while 1:
        
        try:
            
            stn_id,tair_var = MPI.COMM_WORLD.recv(source=RANK_COORD,tag=MPI.ANY_TAG,status=status)
            
            if status.tag == TAG_STOPWORK:
                MPI.COMM_WORLD.send((None,None,None,None), dest=RANK_WRITE, tag=TAG_STOPWORK)
                print "".join(["Worker ",str(rank),": Finished"]) 
                return 0
            
            chg_pt.find_stn_chg_pt(stn_id, tair_var)
            
            MPI.COMM_WORLD.send((stn_id,tair_var,chg_pt.stn_maxT,chg_pt.stn_sigchgpt), dest=RANK_WRITE, tag=TAG_DOWORK)
        
        except Exception,e:
            
            print "".join(["Error in QA of ",stn_id,":",str(e),"\n"])

        MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)
             
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_mask,mask_por_tmin,mask_por_tmax = params[P_STN_MASKS]
        
    dims_tair = {'tmin':0,'tmax':1}
    
    stnids = stn_da.stn_ids[np.logical_and(stn_mask,np.logical_or(mask_por_tmin,mask_por_tmax))]
    
    ds = create_ncdf(params[P_PATH_OUT], stnids)
    
    n = np.sum(np.logical_and(mask_por_tmin,stn_mask)) + np.sum(np.logical_and(mask_por_tmax,stn_mask))
    
    stat_chk = status_check(n,30)
    while 1:
       
        stn_id,tair_var,Tm,Tm_sig = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                print "Writer: Finished"
                return 0
        else:
            
            dim_stnid = np.nonzero(stnids==stn_id)[0][0]
            ds.variables['T'][dims_tair[tair_var],dim_stnid] = Tm
            ds.variables['sig'][dims_tair[tair_var],dim_stnid] = Tm_sig
            stat_chk.increment()

def proc_coord(params,nwrkers):
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    
    stn_mask,mask_por_tmin,mask_por_tmax = params[P_STN_MASKS]
    
    stnids_tmin = stn_da.stn_ids[mask_por_tmin]
    stnids_tmax = stn_da.stn_ids[mask_por_tmax]
    stnids_tair = {'tmin':stnids_tmin,'tmax':stnids_tmax}
    
    stnids = stn_da.stn_ids[np.logical_and(stn_mask,np.logical_or(mask_por_tmin,mask_por_tmax))]
    
    cnt = 0
    nrec = 0
    
    for stnid in stnids:
        
        for tair_var in ['tmin','tmax']:
            
            if stnid in stnids_tair[tair_var]:
                
                if cnt < nwrkers:
                    dest = cnt+N_NON_WRKRS
                else:
                    dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    nrec+=1
        
                MPI.COMM_WORLD.send((stnid,tair_var), dest=dest, tag=TAG_DOWORK)
                cnt+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)

def create_ncdf(path_out,stn_ids):
        
    ncdf_file = Dataset(path_out,'w')
        
    #Set global attributes
    ncdf_file.title = "SNHT Change Point Analysis"
    ncdf_file.institution = "University of Montana Numerical Terradynamics Simulation Group"
    
    dim_station = ncdf_file.createDimension('stn_id',stn_ids.size)
    dim_tair = ncdf_file.createDimension('tair_var',2)
        
    stations = ncdf_file.createVariable('stn_id','str',('stn_id',))
    stations.long_name = "station id"
    stations.standard_name = "station id"
    stations[:] = np.array(stn_ids,dtype=np.object)
    
    tair_var = ncdf_file.createVariable('tair_var','str',('tair_var',))
    tair_var.long_name = "tair_var"
    tair_var.standard_name = "tair_var"
    tair_var[0] = 'tmin'
    tair_var[1] = 'tmax'
    
    t_var = ncdf_file.createVariable('T','f8',('tair_var','stn_id'))
    t_var.long_name = 'Max T SNHT value'
    
    t_var = ncdf_file.createVariable('sig','i1',('tair_var','stn_id'))
    t_var.long_name = 'Statistically Significant Max T SNHT value'
    
    ncdf_file.sync()
    
    return ncdf_file


if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    np.seterr(invalid='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = '/projects/daymet2/station_data/all/all_1948_2012.nc'
    params[P_PATH_POR] = '/projects/daymet2/station_data/all/all_por_1948_2012.csv'
    params[P_PATH_OUT] = '/projects/daymet2/station_data/all/snht_chgpt_all.nc'
    params[P_START_YMD] = 19480101
    params[P_END_YMD] = 20121231
    
    stn_da = station_data_ncdb(params[P_PATH_DB],(params[P_START_YMD],params[P_END_YMD]))
    stn_mask = np.ones(stn_da.stn_ids.size,dtype=np.bool)
    #sntl_mask = np.logical_and(np.char.startswith(stn_da.stn_ids,"SNOTEL"),stn_da.stns[STATE] != "AK")
    #raws_mask = np.char.startswith(stn_da.stn_ids,"RAWS")
    #stn_mask = np.logical_or(raws_mask,sntl_mask)
    stn_da.ds.close()
    stn_da = None
    
    #Load the period-of-record datafile
    por = load_por_csv(params[P_PATH_POR])
    mask_por_tmin,mask_por_tmax = build_valid_por_masks(por)[0:2]
    
    params[P_STN_MASKS] = (stn_mask,mask_por_tmin,mask_por_tmax)
    
    if rank == RANK_COORD:   
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)