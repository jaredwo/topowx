'''
Created on Nov 15, 2011

@author: jared.oyler
'''
import numpy as np
from twx.utils.input_raster import input_raster
from twx.utils.output_raster import output_raster
from twx.interp.topo_disect import TopoDisectDEM 
from mpi4py import MPI
import sys
from twx.utils.status_check import StatusCheck

TAG_DOWORK = 1
TAG_STOPWORK = 2

RANK_COORD = 0
RANK_WRITE = 1

P_PATH_DEM = 'P_PATH_DEM'
P_PATH_OUTPUT = 'P_PATH_OUTPUT'
P_WIN_SIZES = 'P_WIN_SIZES'
P_STN_LOC_BNDS = 'P_STN_LOC_BNDS'

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

def work_proc(params,rank):
    
    rDEM = TopoDisectDEM(params[P_PATH_DEM])
        
    pt = np.zeros(2,dtype=np.int32)
    result = np.zeros(3)
    status = MPI.Status()
        
    print "".join(["work_proc ",str(rank),": ready to receive work"])
    while 1:
        
        MPI.COMM_WORLD.Recv([pt,MPI.INT],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
            
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send(result, dest=RANK_WRITE, tag=TAG_STOPWORK) 
            return 0
        else:
            
            r,c = pt
            
            #lon,lat = rDEM.getGeoLocation(c, r)
            #elev = rDEM.a[r,c]
            tdi = rDEM.get_tdi(r,c, params[P_WIN_SIZES])
            
            result[0:2] = pt
            result[2] = tdi
            
            MPI.COMM_WORLD.Send(result, dest=RANK_WRITE, tag=TAG_DOWORK) 
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)


def write_proc(params,nwrkers):
    
    rDEM = input_raster(params[P_PATH_DEM])
    aDEM = rDEM.readEntireRaster()
    nodata_val = np.float32(rDEM.data.GetNoDataValue())
    
    lat = rDEM.getLatLon(0.0,np.arange(rDEM.rows),transform=False)[0]
    lon = rDEM.getLatLon(np.arange(rDEM.cols),0.0,transform=False)[1]    
    mask_lon = np.logical_and(lon >= params[P_STN_LOC_BNDS][0],lon <= params[P_STN_LOC_BNDS][1])
    mask_lat = np.logical_and(lat >= params[P_STN_LOC_BNDS][2],lat <= params[P_STN_LOC_BNDS][3])
    aDEM = aDEM[mask_lat,:]
    aDEM = aDEM[:,mask_lon]
    npts = np.sum(aDEM!=nodata_val)
    aDEM = None
    
    a_out = np.ones((rDEM.rows,rDEM.cols),dtype=np.float32)*rDEM.data.GetNoDataValue()
    
    status = MPI.Status()
    nwrkrs_done = 0
    
    result = np.zeros(3)
    stat_chk = StatusCheck(npts,10000)
    while 1:
       
        MPI.COMM_WORLD.Recv([result,MPI.DOUBLE],source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
       
        if status.tag == TAG_STOPWORK:
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                r_out = output_raster(params[P_PATH_OUTPUT],rDEM,noDataVal=rDEM.data.GetNoDataValue())
                r_out.writeDataArray(a_out,0,0)
                r_out = None
                print "writer finished"
                return 0
        else:
            a_out[int(result[0]),int(result[1])] = result[2]
            stat_chk.increment()


def coord_proc(params,nwrkers):
        
    rDEM = input_raster(params[P_PATH_DEM])
    aDEM = rDEM.readEntireRaster()
    nodata_val = np.float32(rDEM.data.GetNoDataValue())
    aDEM[aDEM==nodata_val] = np.nan
    
    lat = rDEM.getLatLon(0.0,np.arange(rDEM.rows),transform=False)[0]
    lon = rDEM.getLatLon(np.arange(rDEM.cols),0.0,transform=False)[1]
    
    #params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,53.0) #CONUS
    
    mask_lon = np.logical_and(lon >= params[P_STN_LOC_BNDS][0],lon <= params[P_STN_LOC_BNDS][1])
    mask_lat = np.logical_and(lat >= params[P_STN_LOC_BNDS][2],lat <= params[P_STN_LOC_BNDS][3])
    
    rows = np.nonzero(mask_lat)[0]
    cols = np.nonzero(mask_lon)[0]

    cnt = 0
    nrec = 0
    
    pt = np.zeros(2,dtype=np.int32)
    
    for r in rows:
        
        for c in cols:
            
            if np.isfinite(aDEM[r,c]):
            
                if cnt < nwrkers:
                    dest = cnt+2
                else:
                    dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    nrec+=1
                pt[0:2] = r,c
                MPI.COMM_WORLD.Send([pt,MPI.INT], dest=dest, tag=TAG_DOWORK)
                cnt+=1
    
    while nrec < cnt:
        MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        nrec+=1
    
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.Send([pt,MPI.INT], dest=w+2, tag=TAG_STOPWORK)
        
    print "coord_proc: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    params[P_PATH_DEM] = "/projects/daymet2/dem/interp_grids/conus/tifs/crop_lst_tmin08.tif"
    params[P_PATH_OUTPUT] = "/projects/daymet2/dem/interp_grids/conus/tifs/crop_lsttdi_tmin08.tif"
    #params[P_WIN_SIZES] = [14] #in km
    params[P_WIN_SIZES] = [3,6,9,12,15] #in km
    params[P_STN_LOC_BNDS] = (-126.0,-64.0,22.0,53.0) #CONUS
    
    if rank == RANK_COORD:
        coord_proc(params, nsize-2)
    elif rank == RANK_WRITE:
        write_proc(params,nsize-2)
    else:
        work_proc(params,rank)

    MPI.COMM_WORLD.Barrier()
    
    