'''
Created on Aug 19, 2013

@author: jared.oyler
'''
import interp.interp_tair as it
import cProfile
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


auxFpaths = ['/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_elev.nc',
             '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_tdi.nc',
             '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmax.nc',
             '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_lst_tmin.nc',
             '/projects/daymet2/dem/interp_grids/conus/ncdf/fnl_climdiv.nc']
        
ptInterper = it.PtInterpTair('/projects/daymet2/station_data/infill/infill_20130725/serial_tmin.nc',
                '/projects/daymet2/station_data/infill/infill_20130725/serial_tmax.nc',
                '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/interp.R',
                '/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_C/Release/libwxTopo_C', 
                auxFpaths)

def run_full_interp_pt():
        

    #mask = np.logical_and(ptInterper.days[YEAR] >= 1981,ptInterper.days[YEAR] <= 2010)
    tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-112.883456,48.684078)
    #print tmin_mean
    #tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-112.874890,48.684078)
    #print tmin_mean

tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-113.975000598,48.1249998008,fixInvalid=True)

SCALE_FACTOR = np.float32(0.01)
tmin_dly = np.round(tmin_dly,2).astype(np.float32)
tmax_dly = np.round(tmax_dly,2).astype(np.float32)

ds = Dataset('/stage/climate/topowx_tile_output/h05v01.1/h05v01_tmax.nc')
tmax_interp = ds.variables['tmax'][:,119,73][:]
tmax_interp = np.round(tmax_interp,2)
print ds.variables['lat'][119]
print ds.variables['lon'][73]

plt.plot(tmax_dly-tmax_interp)
plt.show()


#tmin_dly, tmax_dly, tmin_mean, tmax_mean, tmin_se, tmax_se, tmin_ci, tmax_ci,ninvalid = ptInterper.interpLonLatPt(-112.883456,48.684078)  
#cProfile.run('run_full_interp_pt()')