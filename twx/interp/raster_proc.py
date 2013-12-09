'''
Created on Jul 5, 2011

@author: jared.oyler
'''
from twx.utils.input_raster import input_raster
from twx.utils.output_raster import output_raster
import numpy as np
import twx.utils.util_geo as utlg
import twx.utils.util_misc as utlm
from twx.utils.status_check import status_check
from twx.utils.multiprocess import multiprocess,worker,multiprocess_config

class output_handler_smooth():
    
    def __init__(self,input,output_path,nodata_val):
        self.input = input
        self.output = output_raster(output_path,input,noDataVal=nodata_val)
        self.nodata_val = nodata_val
        self.a_out = None
    
    def init_a_out(self):
        self.a_out = np.ones((self.input.rows,self.input.cols),dtype=np.float32)*self.nodata_val
     
    def handleOutput(self,output):
        rows,cols,vals = output
        self.a_out[rows,cols] = vals
        
    def write_raster(self):
        self.output.writeDataArray(self.a_out, 0, 0)

class worker_smooth(worker):

    def __init__(self,inq,outq,config):
        worker.__init__(self, inq, outq)
        
        self.a = config.a
        self.rows,self.cols = self.a.shape
        self.lon = config.lon
        self.lat = config.lat
        self.radius = config.radius
        self.nodata_val = config.nodata_val
        self.win_size = config.win_size
    
    def do_work(self,work_item):
        rows,cols = work_item
        
        results = np.zeros(len(rows))
        for x in np.arange(len(rows)):
            row = rows[x]
            col = cols[x]
        
            row_start = row-self.win_size
            row_end = row+self.win_size+1
            col_start = col-self.win_size
            col_end = col+self.win_size+1
            
            row_start = row_start if row_start >= 0 else 0
            row_end = row_end if row_end < self.rows else self.rows - 1
            col_start = col_start if col_start >= 0 else 0
            col_end = col_end if col_end < self.cols else self.cols - 1
            
            vals = self.a[row_start:row_end,col_start:col_end]
            ca = utlg.dist_ca(self.lon[row,col],self.lat[row,col],self.lon[row_start:row_end,col_start:col_end],self.lat[row_start:row_end,col_start:col_end])[1]
            
            wgts = gaussian_filter(ca, self.radius)
            results[x] = np.average(vals[vals!=self.nodata_val],weights=wgts[vals!=self.nodata_val])
        
        return rows,cols,results
        
class worker_smooth_config():
    def __init__(self):
        self.a = None
        self.lon = None
        self.lat = None
        self.radius = None
        self.nodata_val = None
        self.win_size = None

class multiprocess_smooth(multiprocess):

    def __init__(self,config_multiprocess,config_worker):
        multiprocess.__init__(self,config_multiprocess,config_worker)

    def build_worker(self,worker_name,inq,outq,config_worker):
        return globals()[worker_name](inq,outq,config_worker)

def run_spatial_smooth(raster_path,output_path,radius,nodata_val,num_procs=1):
    '''
    Smooths a geographic projection raster with a Gaussian filter based on the input radius of influence.
    
    @param raster_path: the path to the input raster file
    @param output_path: the full path for the smoothed output raster file
    @param radius: the radius of influence in radians
    @param nodata_val: the no data value of the raster.  no data grid cells will remain no data and not be considered in smoothing
    @param num_procs: the number of processes to use for parallel processing 
    '''
     
    input = input_raster(raster_path)
    lon,lat = input.x_y_arrays()
    a = input.readEntireRaster()
    
    col_m = int(input.cols/2.0)
    row_m = int(input.rows/2.0)
    
    #Estimate roughly how many grid cell steps out from the target should be used in the smoothing for each grid cell
    x=0 #number of grid cell steps
    while utlg.dist_ca(lon[row_m,col_m],lat[row_m,col_m], lon[row_m,col_m+x], lat[row_m,col_m+x])[1] <= radius:
        x+=1
    x+=2 #add buffer of 2 to x
    print "".join(["Radius: ",str(radius)," Window: ",str(x)])
    
    outhandle = output_handler_smooth(input, output_path, nodata_val)
    
    process_cfg = multiprocess_config()
    process_cfg.numProcs = num_procs
    process_cfg.workerName = "worker_smooth"
    process_cfg.inQueueLimit = None
    process_cfg.outputHandler = outhandle
    process_cfg.status_check_num = 10000
    
    worker_cfg = worker_smooth_config()
    worker_cfg.a = a
    worker_cfg.lat = lat
    worker_cfg.lon = lon
    worker_cfg.nodata_val = nodata_val
    worker_cfg.radius = radius
    worker_cfg.win_size = x
    
    multip = multiprocess_smooth(process_cfg, worker_cfg)
    outhandle.init_a_out()
    
    rows,cols = np.nonzero(a != nodata_val)
    
    row_grps = utlm.split_list(rows,rows.size/100)
    col_grps = utlm.split_list(cols,cols.size/100)
    
    for i in np.arange(len(row_grps)):
        
        multip.process((row_grps[i],col_grps[i]))
    
    multip.handleOutputs()
    multip.terminate()
    outhandle.write_raster()

def gaussian_filter(dists,radius):
    
    wghts = np.exp(-0.5*(dists/(radius*0.3989))**2)
    wghts[dists>radius] = 0.0
    #exp(-d^2/(2r))
    return wghts
    

def spatial_smooth(raster_path,output_path,radius,nodata_val):
    input = input_raster(raster_path)
    output = output_raster(output_path, input,noDataVal=nodata_val)
    cols,rows = input.col_row_arrays()
    lon,lat = input.x_y_arrays(cols,rows)
    a = input.readEntireRaster()
    a_out = np.ones((input.rows,input.cols),dtype=np.float32)*nodata_val
    
    col_m = int(input.cols/2.0)
    row_m = int(input.rows/2.0)
    x=0
    while utlg.dist_ca(lon[row_m,col_m],lat[row_m,col_m], lon[row_m,col_m+x], lat[row_m,col_m+x])[1] <= radius:
        x+=1
    x+=2 #add buffer of 2 to x
    
    rows_data,cols_data = np.nonzero(a != nodata_val)
    
    status = status_check(rows_data.size, 1000000)
    
    for i in np.arange(rows_data.size):
        
        row_start = rows_data[i]-x
        row_end = rows_data[i]+x+1
        col_start = cols_data[i]-x
        col_end = cols_data[i]+x+1
        
        row_start = row_start if row_start >= 0 else 0
        row_end = row_end if row_end < input.rows else input.rows - 1
        col_start = col_start if col_start >= 0 else 0
        col_end = col_end if col_end < input.cols else input.cols - 1
        
        vals = a[row_start:row_end,col_start:col_end]
        ca = utlg.dist_ca(lon[rows_data[i],cols_data[i]],lat[rows_data[i],cols_data[i]],lon[row_start:row_end,col_start:col_end],lat[row_start:row_end,col_start:col_end])[1]
        
        wgts = gaussian_filter(ca, radius)
        a_out[rows_data[i],cols_data[i]]=np.average(vals[vals!=nodata_val],weights=wgts[vals!=nodata_val])
        status.increment()
    
    output.writeDataArray(a_out, 0, 0)
            
if __name__ == '__main__':
    
    nodata_val = np.float64(-1.7e+308)
    num_procs = 12
    path_input = "/projects/daymet2/dem/us_30s_dem.img"
    output_path = "/projects/daymet2/dem/smoothed/ncdf/dem_3_0.tif"
    #output_path = "/home/jared.oyler/Desktop/smth_tst.tif"
    dem_prism = input_raster(path_input)
    mesh = dem_prism.res()[0]
    print mesh
    mesh_radians = mesh * (np.pi / 180.0)
    #x = 1.0771
    x = 3.0
    run_spatial_smooth(path_input,output_path,x*mesh_radians,nodata_val,num_procs)
    dem_smoothed = input_raster(output_path)
    dem_smoothed.to_ncdf("/projects/daymet2/dem/smoothed/ncdf/dem_3_0.nc", "elev", "f4")

    
    ##########################################################################################
#    nodata_val = np.float64(-1.7e+308)
#    num_procs = 15
#    path_input = "/projects/daymet2/dem/us_30s_dem.img"
#    prefix_output = "/projects/daymet2/dem/smoothed/dem_"
#    dem_prism = input_raster(path_input)
#    mesh = dem_prism.res()[0]
#    print mesh
#    mesh_radians = mesh * (np.pi / 180.0)
#    
#    mesh_multipliers = np.arange(2.5,10.5,.5)
#    
#    for x in mesh_multipliers:
#        
#        output_path = "".join([prefix_output,str(x).replace(".","_"),".tif"])
#        run_spatial_smooth(path_input,output_path,x*mesh_radians,nodata_val,num_procs)