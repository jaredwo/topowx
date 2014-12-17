'''
Class for creating a topographic dissection index grid (TDI).
TDI describes the height of a grid cell relative to surrounding terrain
'''
from twx.utils.input_raster import input_raster
import twx.utils.util_geo as utlg
import numpy as np

class TopoDisectDEM(input_raster):
    '''
    classdocs
    '''
    SEARCH_WINDOW = 100
    
    def __init__(self, filePath,bandNum=1):
        '''
        Constructor
        '''
        
        input_raster.__init__(self, filePath, bandNum)
        
        self.a = self.readEntireRaster()
        self.ndata_mask = self.a != self.ndata
        #self.lon_grid,self.lat_grid = self.x_y_arrays()
        self.max_rownum = self.a.shape[0]
        self.max_colnum = self.a.shape[1]

    def get_tdi(self,row,col,tdi_dists):
        
        pt_lon,pt_lat = self.getGeoLocation(col, row)
        #pt_lon = self.lon_grid[row,col]
        #pt_lat = self.lat_grid[row,col]
        pt_elev = self.a[row,col]
        
        start_row = row - self.SEARCH_WINDOW
        end_row = row + self.SEARCH_WINDOW
        start_col = col - self.SEARCH_WINDOW
        end_col = col + self.SEARCH_WINDOW
        
        start_row = start_row if start_row > 0 else 0
        end_row = end_row if end_row < self.max_rownum else self.max_rownum
        
        start_col = start_col if start_col > 0 else 0
        end_col = end_col if end_col < self.max_colnum else self.max_colnum
        
        rcgrid = np.mgrid[start_row:end_row,start_col:end_col]
        rows = rcgrid[0]
        cols = rcgrid[1]
        lons,lats = self.getGeoLocation(cols, rows)
        lons = lons.ravel()
        lats = lats.ravel()
        
        #lons = self.lon_grid[start_row:end_row,start_col:end_col].ravel()
        #lats = self.lat_grid[start_row:end_row,start_col:end_col].ravel()
        elev = self.a[start_row:end_row,start_col:end_col].ravel()
        ndata = self.ndata_mask[start_row:end_row,start_col:end_col].ravel()
        
        lons = lons[ndata]
        lats = lats[ndata]
        elev = elev[ndata]
        tdi = 0.0
        
        if elev.size > 0 and np.sum(elev==0) != elev.size:
        
            pt_dists = utlg.grt_circle_dist(pt_lon, pt_lat, lons, lats)
            
            for adist in tdi_dists:
                
                elevs = elev[pt_dists <= adist]
                
                if elevs.size > 0:
                
                    min_elev = np.min(elevs)
                    max_elev = np.max(elevs)
                
                    if min_elev != max_elev:
                    
                        tdi += (pt_elev-min_elev)/(max_elev-min_elev)
        
        return tdi
        