'''

@author: jared.oyler

Python interface to the wxTopo C functions
'''
from ctypes import *
import numpy as np
import os
import pickle

class prcomp_struct(Structure):
    _fields_ = [("nrows",c_int),
                ("ncols",c_int),
                ("nvars",c_int),
           ("pt_vals",POINTER(c_double)),     
           ("apply_mean",c_int),
           ("apply_std",c_int),
           ("pt_mean",c_double),
           ("pt_std",c_double),
           ("X",POINTER(c_double)),
           ("wgts",POINTER(c_double)),
           ("obs",POINTER(c_double)),
           ("returnvals",POINTER(c_double))]

class pca_struct(Structure):
    _fields_ = [("nrows",c_int),
                ("ncols",c_int),
                ("A",POINTER(c_double)), #the nrows*ncols matrix for the pca
           ("pc_loads",POINTER(c_double)), #the loadings for each PC ncols*ncols
           ("pc_scores",POINTER(c_double)), #the scores for each PC nrows*ncols
           ("var_explain",POINTER(c_double))] #variance explained by each PCA ncols vector

class boot_struct(Structure):
    _fields_ = [("nboots",c_int),
                ("nobs",c_int),
           ("pt_x",c_double),
           ("pt_y",c_double),
           ("pt_z",c_double),
           ("X",POINTER(c_double)),
           ("wgts",POINTER(c_double)),
           ("obs",POINTER(c_double)),
           ("returnvals",POINTER(c_double))]

class regress_struct(Structure):
    _fields_ = [("nobs",c_int),
           ("pt_x",c_double),
           ("pt_y",c_double),
           ("pt_z",c_double),
           ("X",POINTER(c_double)),
           ("wgts",POINTER(c_double)),
           ("obs",POINTER(c_double)),
           ("predictval",POINTER(c_double))]

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
    
class RepRegressStruct(Structure):
    _fields_ = [("nYRows",c_int),
           ("nYCols",c_int),
           ("nZLen",c_int),
           ("Y",POINTER(c_double)),
           ("Z",POINTER(c_double)),
           ("z",POINTER(c_double)),
           ("predictVals",POINTER(c_double)),
           ("fitVals",POINTER(c_double))]
    
class clib_wxTopo():
    
    def __init__(self):
                
        #get system path to twx
        twx_path = os.path.split(os.path.split(__file__)[0])[0]
        #get system path to libtwx_c.so
        lib_path = os.path.join(twx_path,'lib','libtwx_c.so')
                
        self.c_lib = CDLL(lib_path)
    
    def pca_gwpca(self,A,wgt):
        nrows,ncols = A.shape
        A = np.require(A,dtype=np.float64,requirements=['C','A','W','O'])
        
        #Subtract out weighted average
        A = A - np.average(A, axis=0, weights=wgt)
        
        wgt.shape = (wgt.size,1)
        #Calculate covariance matrix
        cov = np.dot(A.T,wgt * A)
        
        #Weighted standard deviations
        sigma = np.sqrt(np.diag(cov))
        
        #standardize with sigma
        A = A/sigma
        
        #put in weights
        A = A * np.sqrt(wgt)
        
        pc_loads = np.zeros((ncols,ncols))
        pc_scores = np.zeros((nrows,ncols))
        var_explain = np.zeros(ncols)
        
        a_pca_str = pca_struct(nrows,ncols,
                               A.ctypes.data_as(POINTER(c_double)),
                               pc_loads.ctypes.data_as(POINTER(c_double)),
                               pc_scores.ctypes.data_as(POINTER(c_double)),
                               var_explain.ctypes.data_as(POINTER(c_double)))
        
        error = self.c_lib.pca_basic(byref(a_pca_str))
        
        return pc_loads,pc_scores,var_explain,error
    
    def pca_basic(self,A,center=True,scale=False):
        
        nrows,ncols = A.shape
        A = np.require(A,dtype=np.float64,requirements=['C','A','W','O'])
        
        if center:
            A = A - np.mean(A,axis=0)
        
        if scale:
            A = A/np.std(A,axis=0,ddof=1)
        
        pc_loads = np.zeros((ncols,ncols))
        pc_scores = np.zeros((nrows,ncols))
        var_explain = np.zeros(ncols)
        
        a_pca_str = pca_struct(nrows,ncols,
                               A.ctypes.data_as(POINTER(c_double)),
                               pc_loads.ctypes.data_as(POINTER(c_double)),
                               pc_scores.ctypes.data_as(POINTER(c_double)),
                               var_explain.ctypes.data_as(POINTER(c_double)))
        
        error = self.c_lib.pca_basic(byref(a_pca_str))
        
        return pc_loads,pc_scores,var_explain,error
    
    def repRegress(self,X,Y,x,w):
        
        #pickle.dump((X,Y,x,w),open('/projects/daymet2/regress_test.pickle','wb'))
        
        x.shape = (x.shape[0],1)
        W = np.diag(w)
        
        X_t = np.transpose(X)
        
        #m1 = inverse(X_t * W * X)
        m1 = np.linalg.inv(np.dot(np.dot(X_t,W),X))
        #m2 = X_t * W
        m2 = np.dot(X_t,W)
        #m3 = m1 * m2
        m3 = np.dot(m1,m2)
        
        z = np.dot(np.transpose(x),m3)
        Z = np.dot(X,m3)
        
        #z = np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),W).ravel()
        
        Y = np.require(Y,dtype=np.float64,requirements=['C','A','W','O'])
        Z = np.require(Z,dtype=np.float64,requirements=['C','A','W','O'])
        z = np.require(z,dtype=np.float64,requirements=['C','A','W','O'])
        
        predictVals = np.zeros(Y.shape[0])
        fitVals = np.zeros((Y.shape[0],z.size))
        
        aRepRegressStr = RepRegressStruct(Y.shape[0],Y.shape[1],z.size,
                                          Y.ctypes.data_as(POINTER(c_double)),
                                          Z.ctypes.data_as(POINTER(c_double)),
                                          z.ctypes.data_as(POINTER(c_double)),
                                          predictVals.ctypes.data_as(POINTER(c_double)),
                                          fitVals.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.repRegress(byref(aRepRegressStr))
        
        x.shape = (x.shape[0],)
        
        return predictVals,fitVals
        
    
    def prcomp_interp(self,stn_design_mat,wgts,obs,pt_vals,pt_mean=0,pt_std=0,apply_mean=False,apply_std=False):
        
        nrows,ncols = obs.shape
        nvars = pt_vals.size
        
        X = np.require(stn_design_mat,dtype=np.float64,requirements=['C','A','W','O'])
        pt_vals = np.require(pt_vals,dtype=np.float64,requirements=['C','A','W','O'])
        wgts = np.require(wgts,dtype=np.float64,requirements=['C','A','W','O'])
        obs = np.require(obs,dtype=np.float64,requirements=['C','A','W','O'])
        
        returnvals = np.zeros(nrows)
                
        a_prcomp_str = prcomp_struct(nrows,ncols,nvars,pt_vals.ctypes.data_as(POINTER(c_double)),
                                     int(apply_mean),int(apply_std),pt_mean,pt_std,
                                     X.ctypes.data_as(POINTER(c_double)),
                                     wgts.ctypes.data_as(POINTER(c_double)),
                                     obs.ctypes.data_as(POINTER(c_double)),
                                     returnvals.ctypes.data_as(POINTER(c_double)))
        
        error = self.c_lib.prcomp_interp(byref(a_prcomp_str))
        
        return returnvals,error
    
    def bootstrap(self,stn_x,stn_y,stn_z,wgts,obs,pt_x,pt_y,pt_z,nboots=1000):
        
        nobs = obs.size
        X = np.require(np.column_stack((np.ones(nobs),stn_x,stn_y,stn_z)),dtype=np.float64,requirements=['C','A','W','O'])
        wgts = np.require(wgts,dtype=np.float64,requirements=['C','A','W','O'])
        obs = np.require(obs,dtype=np.float64,requirements=['C','A','W','O'])
        
        returnvals = np.zeros(4)
        
        a_boot_str = boot_struct(nboots,nobs,pt_x,pt_y,pt_z,
                                       X.ctypes.data_as(POINTER(c_double)),
                                       wgts.ctypes.data_as(POINTER(c_double)),
                                       obs.ctypes.data_as(POINTER(c_double)),
                                       returnvals.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.bootstrap(byref(a_boot_str))
        
        return returnvals
    
    def regress(self,stn_x,stn_y,stn_z,wgts,obs,pt_x,pt_y,pt_z):
    
        nobs = obs.size
        X = np.require(np.column_stack((np.ones(nobs),stn_x,stn_y,stn_z)),dtype=np.float64,requirements=['C','A','W','O'])
        wgts = np.require(wgts,dtype=np.float64,requirements=['C','A','W','O'])
        obs = np.require(obs,dtype=np.float64,requirements=['C','A','W','O'])
        
        predictval = np.zeros(1)
        
        a_regress_str = regress_struct(nobs,pt_x,pt_y,pt_z,
                                       X.ctypes.data_as(POINTER(c_double)),
                                       wgts.ctypes.data_as(POINTER(c_double)),
                                       obs.ctypes.data_as(POINTER(c_double)),
                                       predictval.ctypes.data_as(POINTER(c_double)))
        
        self.c_lib.regress(byref(a_regress_str))
        
        return predictval[0]
        
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
        