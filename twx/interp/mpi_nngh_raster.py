'''
Created on Nov 15, 2011

@author: jared.oyler
'''
import numpy as np
from twx.utils.input_raster import input_raster
from twx.utils.output_raster import output_raster
from mpi4py import MPI
import sys
from twx.utils.status_check import status_check
from twx.db.station_data import StationSerialDataDb, MASK,VARIO_NUG,\
    VARIO_PSILL, VARIO_RNG
from twx.interp.station_select import station_select

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
    
    rDEM = input_raster(params[P_PATH_DEM])
    
    stn_da = StationSerialDataDb('/projects/daymet2/station_data/infill/infill_fnl/serial_tmin.nc', 'tmin')
    stn_mask = np.isfinite(stn_da.stns[VARIO_NUG])
    stn_slct = station_select(stn_da, stn_mask)
    stn_slct.set_params(100, 100)
    
    pt = np.zeros(2,dtype=np.int32)
    result = np.zeros(5)
    status = MPI.Status()
        
    print "".join(["work_proc ",str(rank),": ready to receive work"])
    while 1:
        
        MPI.COMM_WORLD.Recv([pt,MPI.INT],source=RANK_COORD, tag=MPI.ANY_TAG,status=status)
            
        if status.tag == TAG_STOPWORK:
            MPI.COMM_WORLD.Send(result, dest=RANK_WRITE, tag=TAG_STOPWORK) 
            return 0
        else:
            
            r,c = pt
            lon,lat = rDEM.getGeoLocation(c, r)
            stn_slct.set_pt(lat, lon)
            stn_slct.set_ngh_stns(load_obs=False)
            
            nug = np.average(stn_slct.ngh_stns[VARIO_NUG],weights=stn_slct.ngh_wgt)
            psill = np.average(stn_slct.ngh_stns[VARIO_PSILL],weights=stn_slct.ngh_wgt)
            rng = np.average(stn_slct.ngh_stns[VARIO_RNG],weights=stn_slct.ngh_wgt)
                        
            result[0:2] = pt
            result[2:5] = nug,psill,rng
            
            MPI.COMM_WORLD.Send(result, dest=RANK_WRITE, tag=TAG_DOWORK) 
            MPI.COMM_WORLD.send(rank, dest=RANK_COORD, tag=TAG_DOWORK)


def write_proc(params,nwrkers):
    
    rDEM = input_raster(params[P_PATH_DEM])
    aDEM = rDEM.readEntireRaster()
    npts = np.sum(aDEM==1)
    
    nug_out = np.ones((rDEM.rows,rDEM.cols),dtype=np.float32)*-1
    psill_out = np.ones((rDEM.rows,rDEM.cols),dtype=np.float32)*-1
    rng_out = np.ones((rDEM.rows,rDEM.cols),dtype=np.float32)*-1
    
    status = MPI.Status()
    nwrkrs_done = 0
    
    result = np.zeros(5)
    stat_chk = status_check(npts,10000)
    while 1:
       
        MPI.COMM_WORLD.Recv([result,MPI.DOUBLE],source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
       
        if status.tag == TAG_STOPWORK:
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                r_out = output_raster("".join([params[P_PATH_OUTPUT],"nug.tif"]),rDEM,noDataVal=-1)
                r_out.writeDataArray(nug_out,0,0)
                r_out = None
                
                r_out = output_raster("".join([params[P_PATH_OUTPUT],"psill.tif"]),rDEM,noDataVal=-1)
                r_out.writeDataArray(psill_out,0,0)
                r_out = None
                
                r_out = output_raster("".join([params[P_PATH_OUTPUT],"rng.tif"]),rDEM,noDataVal=-1)
                r_out.writeDataArray(rng_out,0,0)
                r_out = None
                
                print "writer finished"
                return 0
        else:
            #nug,psill,rng
            #print result
            nug_out[int(result[0]),int(result[1])] = result[2]
            psill_out[int(result[0]),int(result[1])] = result[3]
            rng_out[int(result[0]),int(result[1])] = result[4]
            stat_chk.increment()


def coord_proc(params,nwrkers):
        
    rDEM = input_raster(params[P_PATH_DEM])
    aDEM = rDEM.readEntireRaster()
    
    rows = np.arange(rDEM.rows)
    cols = np.arange(rDEM.cols)

    cnt = 0
    nrec = 0
    
    pt = np.zeros(2,dtype=np.int32)
    
    for r in rows:
        
        for c in cols:
            
            if aDEM[r,c]==1:
            
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
    params[P_PATH_DEM] = '/projects/daymet2/dem/interp_grids/conus/tifs/mask_all_cce_nd.tif'
    params[P_PATH_OUTPUT] = '/projects/daymet2/dem/interp_grids/conus/tifs/krigsmth_'
    
    if rank == RANK_COORD:
        coord_proc(params, nsize-2)
    elif rank == RANK_WRITE:
        write_proc(params,nsize-2)
    else:
        work_proc(params,rank)

    MPI.COMM_WORLD.Barrier()
    
    