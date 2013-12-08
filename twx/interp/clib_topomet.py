'''

@author: jared.oyler

Python interface to the Topomet C library
'''
from ctypes import *
import numpy as np

class tair_struct(Structure):
    _fields_ = [("nstns",c_int),
           ("ndays",c_int),
           ("pt_x",c_double),
           ("pt_y",c_double),
           ("pt_z",c_double),
           ("X",POINTER(c_double)),
           ("stn_wgt_dist",POINTER(c_double)),
           ("stn_wgt_resid",POINTER(c_double)),
           ("stn_obs",POINTER(c_double)),
           ("predictvals",POINTER(c_double))]
    
class prcp_struct(Structure):
    _fields_ = [("nstns",c_int),
           ("ndays",c_int),
           ("pt_x",c_double),
           ("pt_y",c_double),
           ("pt_z",c_double),
           ("X",POINTER(c_double)),
           ("stn_wgt_dist",POINTER(c_double)),
           ("stn_wgt_resid",POINTER(c_double)),
           ("stn_obs",POINTER(c_double)),
           ("stn_sum_obs",POINTER(c_double)),
           ("pop_crit",c_double),
           ("predictvals",POINTER(c_double))]
    
class topo_disect_struct(Structure):
    _fields_ = [("dem",POINTER(c_double)),
           ("nrows",c_int),
           ("ncols",c_int),
           ("nwins",c_int),
           ("npts",c_int),
           ("rows",POINTER(c_int)),
           ("cols",POINTER(c_int)),
           ("windows",POINTER(c_int)),
           ("topo_disect",POINTER(c_double))]

class clib_topomet():
    
    def __init__(self,lib_path):
        
        self.c_lib = CDLL(lib_path)
    
    def predict_tair_gsl(self,x,y,z,stn_x,stn_y,stn_z,stn_wgt_dist,stn_wgt_resid,stn_obs):
        
        nstns = stn_x.size
        ndays = stn_obs.shape[0]
        X = np.require(np.column_stack((np.ones(nstns),stn_x,stn_y,stn_z)),dtype=np.float64,requirements=['C','A','W','O'])
        stn_wgt_dist = np.require(stn_wgt_dist,dtype=np.float64,requirements=['C','A','W','O'])
        stn_wgt_resid = np.require(stn_wgt_resid,dtype=np.float64,requirements=['C','A','W','O'])
        stn_obs = np.require(stn_obs,dtype=np.float64,requirements=['C','A','W','O'])
        
        predict_vals = np.zeros(ndays)
        
        a_tair_str = tair_struct(nstns,ndays,x,y,z,
                                     X.ctypes.data_as(POINTER(c_double)),
                                     stn_wgt_dist.ctypes.data_as(POINTER(c_double)),
                                     stn_wgt_resid.ctypes.data_as(POINTER(c_double)),
                                     stn_obs.ctypes.data_as(POINTER(c_double)),
                                     predict_vals.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.predict_tair(byref(a_tair_str))
        
        return predict_vals
    
    def predict_prcp_gsl(self,x,y,z,stn_x,stn_y,stn_z,stn_wgt_dist,stn_wgt_resid,stn_obs,stn_sum_obs,pop_crit):
        
        nstns = stn_x.size
        ndays = stn_obs.shape[0]
        X = np.require(np.column_stack((np.ones(nstns),stn_x,stn_y,stn_z)),dtype=np.float64,requirements=['C','A','W','O'])
        stn_wgt_dist = np.require(stn_wgt_dist,dtype=np.float64,requirements=['C','A','W','O'])
        stn_wgt_resid = np.require(stn_wgt_resid,dtype=np.float64,requirements=['C','A','W','O'])
        stn_obs = np.require(stn_obs,dtype=np.float64,requirements=['C','A','W','O'])
        stn_sum_obs = np.require(stn_sum_obs,dtype=np.float64,requirements=['C','A','W','O'])
        
        predict_vals = np.zeros(ndays)
        
        a_prcp_str = prcp_struct(nstns,ndays,x,y,z,
                                     X.ctypes.data_as(POINTER(c_double)),
                                     stn_wgt_dist.ctypes.data_as(POINTER(c_double)),
                                     stn_wgt_resid.ctypes.data_as(POINTER(c_double)),
                                     stn_obs.ctypes.data_as(POINTER(c_double)),
                                     stn_sum_obs.ctypes.data_as(POINTER(c_double)),
                                     pop_crit,
                                     predict_vals.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.predict_prcp(byref(a_prcp_str))
        
        return predict_vals
    
    def calc_topo_disect(self,a_dem,windows,rows,cols):
        
        a_dem = np.require(a_dem,dtype=np.float64,requirements=['C','A','W','O'])
        windows = np.require(windows,dtype=np.int32,requirements=['C','A','W','O'])
        rows = np.require(rows,dtype=np.int32,requirements=['C','A','W','O'])
        cols = np.require(cols,dtype=np.int32,requirements=['C','A','W','O'])
        
        topo_disects = np.zeros(rows.size)

        a_td_struct = topo_disect_struct(a_dem.ctypes.data_as(POINTER(c_double)),
                                         a_dem.shape[0],a_dem.shape[1],windows.size,rows.size,
                                         rows.ctypes.data_as(POINTER(c_int)),
                                         cols.ctypes.data_as(POINTER(c_int)),
                                         windows.ctypes.data_as(POINTER(c_int)),
                                         topo_disects.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.calc_topo_disect(byref(a_td_struct))
        
        return topo_disects
        