'''
A MPI driver for performing GWR interpolation of Tair

@author: jared.oyler
'''

import numpy as np
from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,STN_ID,MASK,BAD, LAT, LON,\
    get_lst_varname, get_optim_varname, get_optim_anom_varname, NEON
from twx.utils.status_check import StatusCheck
import netCDF4
from netCDF4 import Dataset
from twx.utils.input_raster import RasterDataset
from twx.interp.interp_tair import PredictorGrids, build_empty_pt,\
    get_rgn_nnghs_dict, GwrTairNorm
from twx.interp.station_select import StationSelect
import os

TAG_DOWORK = 1
TAG_STOPWORK = 2
TAG_OBSMASKS = 3

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_DB = "P_PATH_DB"
P_PATH_GRID = "P_PATH_GRID"
P_VARNAME = 'P_VARNAME'
P_PATH_OUT = 'P_PATH_OUT'
P_PATHS_PREDICTORS = 'P_PATHS_PREDICTORS'
P_MTH = 'P_MTH'


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
    
    p_grids = PredictorGrids(params[P_PATHS_PREDICTORS])
    #(row,col,lat,lon)
    rc_latlon = np.zeros(4)
    #(row,col,norm,se)
    interp_tair = np.zeros(4)
    pt = build_empty_pt()
    
    stnda = StationSerialDataDb(params[P_PATH_DB], params[P_VARNAME])
    stns = stnda.stns[np.logical_and(np.isfinite(stnda.stns[MASK]),np.isnan(stnda.stns[BAD]))]
    rgn_nnghs = get_rgn_nnghs_dict(stns)
    varname = params[P_VARNAME]
    interp_mth = params[P_MTH]
    
    mask_stns = np.isnan(stnda.stns[BAD])         
    stn_slct = StationSelect(stnda, stn_mask=mask_stns, rm_zero_dist_stns=False)
    
    gwr = GwrTairNorm(stn_slct)
    
    while 1:
    
        MPI.COMM_WORLD.Recv([rc_latlon,MPI.DOUBLE],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            MPI.COMM_WORLD.Send([interp_tair,MPI.DOUBLE],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        
        else:
              
            pt[LAT] = rc_latlon[2]
            pt[LON] = rc_latlon[3]
            p_grids.setPtValues(pt, chgLatLon=False)
            
            #Set the monthly lst values and optim nnghs on the point
            for mth in np.arange(1,13):
                pt[get_lst_varname(mth)] = pt["%s%02d"%(varname,mth)]
                pt[get_optim_varname(mth)],pt[get_optim_anom_varname(mth)] = rgn_nnghs[pt[NEON]][mth]
            
            tair_mean, tair_var = gwr.gwr_predict(pt, interp_mth)
            tair_se = np.sqrt(tair_var)
            interp_tair[:] = rc_latlon[0],rc_latlon[1],tair_mean,tair_se
            MPI.COMM_WORLD.Send([interp_tair,MPI.DOUBLE],dest=RANK_WRITE,tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK) 
                
def proc_write(params,nwrkers):

    status = MPI.Status()
    nwrkrs_done = 0
        
    ds_grid = RasterDataset(params[P_PATH_GRID])
    a_grid = ds_grid.read_as_array()
    n_interps = np.sum(~a_grid.mask)
    lats,lons = ds_grid.get_coord_grid_1d()
    
    fname_out = "gwr_norm_%s_%02d.nc"%(params[P_VARNAME],params[P_MTH])
    ds_out = Dataset(os.path.join(params[P_PATH_OUT],fname_out),'w')
    ds_out.createDimension('lat',lats.size)
    ds_out.createDimension('lon',lons.size)
    vlat = ds_out.createVariable('lat','f8',('lat',),fill_value=False)
    vlon = ds_out.createVariable('lon','f8',('lon',),fill_value=False)
    vlat[:] = lats
    vlon[:] = lons
    vnorm = ds_out.createVariable("norm",'f8',('lat','lon',),fill_value=netCDF4.default_fillvals['f8'])
    vse = ds_out.createVariable("se",'f8',('lat','lon',),fill_value=netCDF4.default_fillvals['f8'])
    #(row,col,norm,se)
    interp_tair = np.zeros(4)
    
    stat_chk = StatusCheck(n_interps,5000)
    
    while 1:
        
        MPI.COMM_WORLD.Recv([interp_tair,MPI.DOUBLE],source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
                
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                ds_out.sync()
                ds_out.close()
                
                print "Writer: Finished"
                return 0
        else:

            r,c = interp_tair[0:2].astype(np.int)
            vnorm[r,c] = interp_tair[2]
            vse[r,c] = interp_tair[3]
            stat_chk.increment()
                
def proc_coord(params,nwrkers):
    
    ds_grid = RasterDataset(params[P_PATH_GRID])
    a_grid = ds_grid.read_as_array()
    lat,lon = ds_grid.get_coord_mesh_grid()
    grid_mask = np.nonzero(~a_grid.mask)
    lat = lat[grid_mask]
    lon = lon[grid_mask]
    rows = grid_mask[0]
    cols = grid_mask[1]
    
    #(row,col,lat,lon)
    rc_latlon = np.zeros(4)
    
    print "Coord: Done initialization. Starting to send work."
    
    cnt = 0
    nrec = 0
                
    for x in np.arange(cols.size):
        
        rc_latlon[:] = rows[x],cols[x],lat[x],lon[x]
        
        if cnt < nwrkers:
            dest = cnt+N_NON_WRKRS
        else:
            dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            nrec+=1
            
        MPI.COMM_WORLD.Send([rc_latlon,MPI.DOUBLE], dest=dest, tag=TAG_DOWORK)
        cnt+=1
                        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.Send([rc_latlon,MPI.DOUBLE], dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DB] = "/projects/daymet2/station_data/infill/serial_gwr_norm/serial_tmin.nc"
    params[P_PATH_GRID] = "/projects/daymet2/docs/final_writeup/fnl_uploads/krig_vs_gwr/twx_tmin_se_08_wmt.tif"
    params[P_VARNAME] = 'tmin'
    params[P_MTH] = 8
    params[P_PATH_OUT] = '/projects/daymet2/docs/final_writeup/fnl_uploads/krig_vs_gwr/'
    
    gridPath = '/projects/daymet2/dem/interp_grids/conus/ncdf/'
    auxFpaths = ["".join([gridPath,'fnl_elev.nc']),
                 "".join([gridPath,'fnl_tdi.nc']),
                 "".join([gridPath,'fnl_climdiv.nc']),
                 "".join([gridPath,'fnl_mask.nc'])]
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmin%02d.nc'%mth]) for mth in np.arange(1,13)])
    auxFpaths.extend(["".join([gridPath,'fnl_lst_tmax%02d.nc'%mth]) for mth in np.arange(1,13)])
    params[P_PATHS_PREDICTORS] = auxFpaths
        
    if rank == RANK_COORD:        
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()