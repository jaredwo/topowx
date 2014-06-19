'''
A MPI driver for interpolating tair to a specified grid using interp.interp_tair

@author: jared.oyler
'''

from mpi4py import MPI
import sys
from twx.db.station_data import StationSerialDataDb,DTYPE_STN_BASIC
from twx.utils.status_check import status_check
from netCDF4 import Dataset,num2date
import os
import numpy as np
from twx.modis.ModisLst import ImputeLST,ImputeLstNorm,LstData,MYD11A2_MTH_DAYS8
import twx.utils.util_dates as utld
from twx.utils.util_dates import MONTH
from twx.db.ushcn import TairAggregate
from pyhdf.SD import SD, SDC

TAG_DOWORK = 1
TAG_STOPWORK = 1000

RANK_COORD = 0
RANK_WRITE = 1
N_NON_WRKRS = 2

P_PATH_STNDB = 'P_PATH_STNDB'

P_CHCKSIZE_X = 'P_CHCKSIZE_X'
P_CHCKSIZE_Y = 'P_CHCKSIZE_Y'
P_TAIR_VAR = 'P_TAIR_VAR'
P_LST_VAR = 'P_LST_VAR'
P_PATH_NC_STACK = 'P_PATH_NC_STACK'
P_PATH_OUT = 'P_PATH_OUT'
P_MODIS_TILES = 'P_MODIS_TILES'
P_PATH_HDF = 'P_PATH_HDF'

CONUS_TILES = [ 'h07v05',
                'h07v06',
                'h08v04',
                'h08v05',
                'h08v06',
                'h09v03',
                'h09v04',
                'h09v05',
                'h09v06',
                'h10v03',
                'h10v04',
                'h10v05',
                'h10v06',
                'h11v03',
                'h11v04',
                'h11v05',
                'h11v06',
                'h12v03',
                'h12v04',
                'h12v05',
                'h13v03',
                'h13v04',
                'h14v03',
                'h14v04']

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

#def proc_work(params,rank):
#
#    status = MPI.Status()
#    
#    stnda = StationSerialDataDb(params[P_PATH_STNDB], params[P_TAIR_VAR],stn_dtype=DTYPE_STN_BASIC)
#    
#    impLST = ImputeLST(stnda, params[P_TAIR_VAR])
#            
#    while 1:
#        
#        r,c,ds_num = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)
#        
#        if status.tag == TAG_STOPWORK:
#            #MPI.COMM_WORLD.Send([np.zeros((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16),MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
#            MPI.COMM_WORLD.send([None]*4,dest=RANK_WRITE,tag=TAG_STOPWORK)
#            print "".join(["Worker ",str(rank),": Finished"]) 
#            return 0
#        else:
#            
#            ds = Dataset(params[P_PATHS_LST_NC][ds_num])
#            lst = ds.variables[params[P_LST_VAR]][:,r:r+params[P_CHCKSIZE_Y],c:c+params[P_CHCKSIZE_X]]
#            xPts = ds.variables['x'][c:c+params[P_CHCKSIZE_X]]
#            yPts = ds.variables['x'][r:r+params[P_CHCKSIZE_Y]]
#            fval = ds.variables[params[P_LST_VAR]]._FillValue
#            ds.close()
#            
#            nmiss = np.sum(lst.mask,axis=0)
#            rows,cols = np.nonzero(np.logical_and(nmiss < lst.shape[0], nmiss > 0))
#            
#            for aRow,aCol in zip(rows,cols):
#                    
#                rclst = (lst[:,aRow,aCol]*0.02) - 273.15
#                
#                try:
#                    rclstImp = (impLST.imputeByMth(xPts[aCol], yPts[aRow], rclst) + 273.15)/0.02
#                    lst[:,aRow,aCol] = np.round(rclstImp)
#                except Exception as e:
#                    print "ERROR: could impute lst: "+str(e)
#                    
#            #MPI.COMM_WORLD.Send([np.ma.filled(lst, fval),MPI.INTEGER16],dest=RANK_WRITE,tag=TAG_DOWORK) 
#            MPI.COMM_WORLD.send((np.ma.filled(lst, fval),r,c,ds_num),dest=RANK_WRITE,tag=TAG_DOWORK)
#            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)

def proc_work(params,rank):

    status = MPI.Status()
    
    stnda = StationSerialDataDb(params[P_PATH_STNDB], params[P_TAIR_VAR],stn_dtype=DTYPE_STN_BASIC)
    
    timeVarsSet = False
    
    imputeNorms = {}
    mthMasks = []
            
    while 1:
        
        r,c,tileNum = MPI.COMM_WORLD.recv(source=RANK_COORD, tag=MPI.ANY_TAG, status=status)
        
        if status.tag == TAG_STOPWORK:
            #MPI.COMM_WORLD.Send([np.zeros((params[P_CHCKSIZE_Y],params[P_CHCKSIZE_X]),dtype=np.int16),MPI.INT],dest=RANK_WRITE,tag=TAG_STOPWORK) 
            MPI.COMM_WORLD.send([None]*4,dest=RANK_WRITE,tag=TAG_STOPWORK)
            print "".join(["Worker ",str(rank),": Finished"]) 
            return 0
        else:
            
            tileName = params[P_MODIS_TILES][tileNum]
            
            if imputeNorms.has_key(tileName):
                imp = imputeNorms[tileName]
            else:
                impData = LstData(params[P_PATH_NC_STACK], tileName, params[P_LST_VAR],params[P_TAIR_VAR],stnda)
                imp = ImputeLstNorm(impData)
                imputeNorms[tileName] = imp
                
                if not timeVarsSet:
                    
                    yday = imp.lstData.dsMain.variables['yday'][imp.lstData.maskTime]
                    
                    for mth in np.arange(1,13):
                        mthMasks.append(np.in1d(yday, np.array(MYD11A2_MTH_DAYS8[mth]), False))
    
#                    var_time = imp.lstData.dsMain.variables['time'][imp.lstData.maskTime]         
#                    dates = num2date(var_time[:], imp.lstData.dsMain.variables['time'].units)
#                    days = utld.get_days_metadata_dates(dates)                                    
#                    tagg = TairAggregate(days)
                    timeVarsSet = True
            
            imp.lstData.set_lst_chk(r, r+params[P_CHCKSIZE_Y], c,c+params[P_CHCKSIZE_X])
            lst = impData.lstChk
            
            #lst = ds.variables[params[P_LST_VAR]][imp.lstData.maskTime,r:r+params[P_CHCKSIZE_Y],c:c+params[P_CHCKSIZE_X]]
            tileRows = np.arange(r,r+params[P_CHCKSIZE_Y])
            tileCols = np.arange(c,c+params[P_CHCKSIZE_X])
            
            
            
            #xPts = ds.variables['x'][c:c+params[P_CHCKSIZE_X]]
            #yPts = ds.variables['y'][r:r+params[P_CHCKSIZE_Y]]
            #fval = ds.variables[params[P_LST_VAR]]._FillValue

            nmiss = np.sum(lst.mask,axis=0)
            
            rows,cols = np.nonzero(np.logical_and(nmiss < lst.shape[0], nmiss > 0))
            
            lstImp = np.ma.copy(lst)*np.float(0.02)
            #lstImp = np.zeros(lst.shape,dtype=np.float)
            #lstImp = np.ma.masked_equal(lstImp, 0)
            
            for aRow,aCol in zip(rows,cols):
                                   
                try:
                    
                    rclstImp = imp.imputeNorm(tileCols[aCol], tileRows[aRow])
                    lstImp[:,aRow,aCol] = rclstImp
                    
                except Exception as e:
                    print "ERROR: could impute lst Row %d Col %d. Error: %s "%(tileRows[aRow],tileCols[aCol],str(e))
                        
            lstMthMeans = []
            for mthMask in mthMasks:
                
                mthMean = np.ma.mean(lstImp[mthMask,:,:],axis=0)
                mthMean = np.round(mthMean/0.02).astype(np.uint16)
                lstMthMeans.append(mthMean)
            
#            lstImpMthly = tagg.dailyToMthly(lstImp, -1)[0]
#            
#            lstMthMeans = []
#            for mth in np.arange(1,13):
#                
#                mthMean = np.ma.mean(lstImpMthly[tagg.yrMths[MONTH]==mth,:,:],axis=0)
#                mthMean = np.round(mthMean/0.02).astype(np.uint16)
#                lstMthMeans.append(mthMean)
                
            MPI.COMM_WORLD.send((lstMthMeans,r,c,tileNum),dest=RANK_WRITE,tag=TAG_DOWORK)
            MPI.COMM_WORLD.send(rank,RANK_COORD,tag=TAG_DOWORK)

def proc_write(params,nwrkers):
    
    status = MPI.Status()
    nwrkrs_done = 0
    
    nchksPerTile = (1200/params[P_CHCKSIZE_X])*(1200/params[P_CHCKSIZE_Y])
    nchks = nchksPerTile*len(params[P_MODIS_TILES])
    
    allTileOutputFiles = []
    print "WRITER: Performing output setup..."
    for i in np.arange(len(params[P_MODIS_TILES])):
        
        hdfTileFiles = np.array(os.listdir("".join([params[P_PATH_HDF],"MYD11A2.005.",params[P_MODIS_TILES][i]])))
        
        tileFile = hdfTileFiles[np.char.startswith(hdfTileFiles, "MYD11A2")][0]
        
        cpFile = "".join([params[P_PATH_HDF],"MYD11A2.005.",params[P_MODIS_TILES][i],"/",tileFile])
        
        tileOutputFiles = []
        for mth in np.arange(1,13):
            outputFile = "".join([params[P_PATH_OUT],params[P_LST_VAR],".",params[P_MODIS_TILES][i],'.%02d.hdf' % mth])
            os.system("".join(["cp ",cpFile," ",outputFile]))
            tileOutputFiles.append(outputFile)
        
        allTileOutputFiles.append(tileOutputFiles)
    
    print "WRITER: Setup complete..."
    
    stat_chk = status_check(nchks,10)
    
    tileRslts = {}
    tileChksDone = {}
    
    for i in np.arange(len(params[P_MODIS_TILES])):
        tileChksDone[i] = 0
    
    while 1:
        
        impLst,r,c,tileNum = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE,tag=MPI.ANY_TAG,status=status)
        
        if status.tag == TAG_STOPWORK:
            
            nwrkrs_done+=1
            if nwrkrs_done == nwrkers:
                
                print "Writer: Finished"
                return 0
        else:
            
            if not tileRslts.has_key(tileNum):
                tileMthRslts = []
                for i in np.arange(len(allTileOutputFiles[tileNum])):
                    lstMth = np.zeros((1200,1200),dtype=np.uint16)
                    lstMth = np.ma.masked_equal(lstMth, 0)
                    tileMthRslts.append(lstMth)
                tileRslts[tileNum] = tileMthRslts
                 
            for i in np.arange(len(allTileOutputFiles[tileNum])):
                tileRslts[tileNum][i][int(r):int(r+params[P_CHCKSIZE_Y]),int(c):int(c+params[P_CHCKSIZE_X])] = impLst[i]
            
            tileChksDone[tileNum]+=1
            
            if tileChksDone[tileNum] == nchksPerTile: 
                print "WRITER: Writing tile: "+params[P_MODIS_TILES][tileNum]
                for i in np.arange(len(allTileOutputFiles[tileNum])):
    
                    sd = SD(allTileOutputFiles[tileNum][i],SDC.WRITE)
                    sds_lst = sd.select(params[P_LST_VAR])
                    fval = sds_lst.attributes()['_FillValue']
                    sds_lst[:] = np.ma.filled(tileRslts[tileNum][i],fval)
                    sd.end()
                
                tileRslts.pop(tileNum)

            stat_chk.increment()
                
def proc_coord(params,nwrkers):
        
    #MPI.COMM_WORLD.bcast(atiler.build_tile_grid_info(), root=RANK_COORD)
    print "COORD: Starting to send work chunks to workers..."
    
    cnt = 0
        
    for i in np.arange(len(params[P_MODIS_TILES])):
        
        ncPath = "".join([params[P_PATH_NC_STACK],params[P_LST_VAR],".",params[P_MODIS_TILES][i],".nc"])
        
        ds = Dataset(ncPath)
        nrows,ncols = ds.variables[params[P_LST_VAR]].shape[1:]
        ds.close()
        
        for r in np.arange(0,nrows,params[P_CHCKSIZE_Y]):
        
            for c in np.arange(0,ncols,params[P_CHCKSIZE_X]):
                 
                if cnt < nwrkers:
                    dest = cnt+N_NON_WRKRS
                else:
                    dest = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                
                cnt+=1
                MPI.COMM_WORLD.send((r,c,i), dest=dest, tag=TAG_DOWORK)
        
    for w in np.arange(nwrkers):
        MPI.COMM_WORLD.send((None,None,None), dest=w+N_NON_WRKRS, tag=TAG_STOPWORK)
    
    print "COORD: done"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    rank = MPI.COMM_WORLD.Get_rank()
    nsize = MPI.COMM_WORLD.Get_size()

    params = {}
    
    params[P_PATH_STNDB] = '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax_8day.nc'
    params[P_TAIR_VAR] = 'tmax'
    params[P_LST_VAR] = 'LST_Day_1km'
    params[P_PATH_NC_STACK] = '/projects/daymet2/climate_office/modis/MYD11A2/nc_stacks/'
    params[P_PATH_HDF] = '/projects/daymet2/climate_office/modis/MYD11A2/'
    params[P_MODIS_TILES] = CONUS_TILES#['h08v05']#CONUS_TILES#['h10v04']#['h11v05']
    params[P_PATH_OUT] = '/projects/daymet2/climate_office/modis/MYD11A2/imputed_mth_means/LST_Day/'
    params[P_CHCKSIZE_X] = 50
    params[P_CHCKSIZE_Y] = 50
        
    if rank == RANK_COORD:
        proc_coord(params, nsize-N_NON_WRKRS)
    elif rank == RANK_WRITE:
        proc_write(params,nsize-N_NON_WRKRS)
    else:
        proc_work(params,rank)

    MPI.COMM_WORLD.Barrier()
