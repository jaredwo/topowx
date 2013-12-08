'''
Created on Jul 5, 2011

@author: jared.oyler
'''
from utils.input_raster import input_raster
import numpy as np
import utils.util_geo as utlg

class dem_transform():
    
    def __init__(self,raster_path,ctopomet):
                
        self.raster = input_raster(raster_path)
        self.a = self.raster.readEntireRaster()
        self.nodata_val = np.float32(self.raster.data.GetNoDataValue())
        self.a[self.a==self.nodata_val] = np.nan

        self.rows = self.raster.rows
        self.cols = self.raster.cols
        
        self.lons,self.lats = self.raster.x_y_arrays()
        
        self.col_m = int(self.raster.cols/2.0)
        self.row_m = int(self.raster.rows/2.0)
        
        self.lon_m, self.lat_m = self.raster.getGeoLocation(self.col_m,self.row_m)
        
        self.ctopomet = ctopomet
    
    def get_win_size(self,radius_km):
        
        x=0 #number of grid cell steps
        lon, lat = self.raster.getGeoLocation(self.col_m+x,self.row_m)
        while utlg.dist_ca(self.lon_m,self.lat_m ,lon,lat)[0] <= radius_km:
            x+=1
            lon, lat = self.raster.getGeoLocation(self.col_m+x,self.row_m)
        return x+5 #add buffer of 5 to x
    
    def smooth(self,lon,lat,radius_km,win_size=None):
        
        col,row = self.raster.getGridCellOffset(lon, lat)
        
        if win_size is None:
            win_size = self.get_win_size(radius_km)
        
        row_start = row-win_size
        row_end = row+win_size+1
        col_start = col-win_size
        col_end = col+win_size+1
        
        row_start = row_start if row_start >= 0 else 0
        row_end = row_end if row_end < self.rows else self.rows - 1
        col_start = col_start if col_start >= 0 else 0
        col_end = col_end if col_end < self.cols else self.cols - 1
        
        vals = self.a[row_start:row_end,col_start:col_end]
        ca = utlg.dist_ca(self.lons[row,col],self.lats[row,col],self.lons[row_start:row_end,col_start:col_end],self.lats[row_start:row_end,col_start:col_end])[0]
        
        wgts = self.gaussian_filter(ca,radius_km)
        
        return np.average(vals[np.isfinite(vals)],weights=wgts[np.isfinite(vals)])

    def gaussian_filter(self,dists,radius):
        
        wghts = np.exp(-0.5*(dists/(radius*0.3989))**2)
        wghts[dists>radius] = 0.0
        #print "".join(["Radius: ",str(self.radius)," # Grid Cells: ",str(wghts[dists<=radius].size)])
        return wghts 

    def topo_disect(self,lons,lats,windows):
        
        cols = np.empty(lons.size,dtype=np.int32)
        rows = np.empty(lats.size,dtype=np.int32)
        
        for x in np.arange(lons.size):
            
            cols[x],rows[x] = self.raster.getGridCellOffset(lons[x], lats[x])
        
        return self.ctopomet.calc_topo_disect(self.a,windows,rows,cols)
               
if __name__ == '__main__':
    
    smthr = dem_transform("/projects/daymet2/dem/smoothed/dem_1_0.tif", np.float64(-1.7e+308),9.5)
