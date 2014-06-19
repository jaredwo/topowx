'''
Created on Feb 20, 2013

@author: jared.oyler
'''
import numpy as np
import twx.utils.util_geo as utlg
from twx.db.station_data import StationDataDb, STN_ID, LON, LAT,UTC_OFFSET,YMD,YEAR
from twx.utils.util_dates import MONTH, MTH_SRT_END_DATES, DAY
from datetime import timedelta
from scipy import stats
import matplotlib.pyplot as plt
import time

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

MAX_DISTANCE = 75 #in km
MIN_POR_OVERLAP = 0.90
MIN_DAILY_NGHBRS = 3

CHGPT_START = 'chgpt_start'
CHGPT_END = 'chgpt_end'

TMIN_CHGPT_START = "".join(['tmin',CHGPT_START])
TMIN_CHGPT_END = "".join(['tmin',CHGPT_END])
TMAX_CHGPT_START = "".join(['tmax',CHGPT_START])
TMAX_CHGPT_END = "".join(['tmax',CHGPT_END])

DTYPE_CHGPTS = [(STN_ID, "<S16"),(TMIN_CHGPT_START,np.int64),(TMIN_CHGPT_END,np.int64),
               (TMAX_CHGPT_START,np.int64),(TMAX_CHGPT_END,np.int64)]

QA_CHGPT_FLG = "C"

# critical levels (simulated by Khaliq and Ouarda 2007)
chgpt_critvals = np.array(([10,4.964,5.197,5.473,5.637,6.188,6.769],
                [12,5.288,5.554,5.876,6.068,6.729,7.459],
                [14,5.54,5.831,6.187,6.402,7.152,8.001],
                [16,5.749,6.059,6.441,6.674,7.492,8.44],
                [18,5.922,6.248,6.652,6.899,7.775,8.807],
                [20,6.07,6.41,6.83,7.089,8.013,9.113],
                [22,6.2,6.551,6.988,7.257,8.22,9.38],
                [24,6.315,6.675,7.123,7.4,8.4,9.609],
                [26,6.417,6.785,7.246,7.529,8.558,9.812],
                [28,6.509,6.884,7.353,7.643,8.697,9.993],
                [30,6.592,6.973,7.451,7.747,8.825,10.153],
                [32,6.669,7.056,7.541,7.841,8.941,10.3],
                [34,6.741,7.132,7.625,7.93,9.05,10.434],
                [36,6.803,7.201,7.699,8.009,9.143,10.552],
                [38,6.864,7.263,7.768,8.081,9.23,10.663],
                [40,6.921,7.324,7.835,8.151,9.317,10.771],
                [42,6.972,7.38,7.894,8.214,9.39,10.865],
                [44,7.022,7.433,7.951,8.273,9.463,10.957],
                [46,7.071,7.484,8.007,8.331,9.53,11.04],
                [48,7.112,7.529,8.054,8.382,9.592,11.116],
                [50,7.154,7.573,8.103,8.432,9.653,11.193],
                [52,7.194,7.616,8.149,8.48,9.711,11.259],
                [54,7.229,7.654,8.19,8.524,9.76,11.324],
                [56,7.264,7.69,8.23,8.566,9.81,11.382],
                [58,7.299,7.727,8.268,8.606,9.859,11.446],
                [60,7.333,7.764,8.308,8.647,9.906,11.498],
                [62,7.363,7.796,8.343,8.683,9.947,11.548],
                [64,7.392,7.827,8.375,8.717,9.985,11.599],
                [66,7.421,7.857,8.408,8.752,10.026,11.648],
                [68,7.449,7.886,8.439,8.784,10.067,11.692],
                [70,7.475,7.913,8.467,8.814,10.099,11.737],
                [72,7.499,7.938,8.496,8.844,10.134,11.776],
                [74,7.525,7.965,8.523,8.873,10.171,11.822],
                [76,7.547,7.989,8.548,8.898,10.2,11.858],
                [78,7.57,8.013,8.575,8.926,10.23,11.895],
                [80,7.591,8.035,8.599,8.951,10.259,11.928],
                [82,7.613,8.059,8.623,8.976,10.29,11.966],
                [84,7.634,8.079,8.647,9.001,10.315,11.995],
                [86,7.655,8.102,8.67,9.026,10.347,12.033],
                [88,7.673,8.121,8.691,9.047,10.37,12.059],
                [90,7.692,8.14,8.71,9.067,10.394,12.089],
                [92,7.711,8.16,8.732,9.09,10.417,12.12],
                [94,7.73,8.181,8.752,9.11,10.447,12.153],
                [96,7.745,8.196,8.77,9.127,10.465,12.175],
                [98,7.762,8.214,8.788,9.147,10.484,12.196],
                [100,7.778,8.231,8.807,9.167,10.507,12.228],
                [105,7.819,8.273,8.851,9.214,10.562,12.291],
                [110,7.856,8.312,8.892,9.255,10.608,12.343],
                [115,7.891,8.35,8.931,9.296,10.656,12.401],
                [120,7.921,8.38,8.963,9.33,10.694,12.446],
                [125,7.952,8.413,8.999,9.365,10.735,12.488],
                [130,7.983,8.446,9.032,9.4,10.772,12.538],
                [135,8.01,8.474,9.063,9.431,10.808,12.579],
                [140,8.038,8.501,9.092,9.462,10.845,12.621],
                [145,8.063,8.529,9.12,9.49,10.877,12.66],
                [150,8.086,8.554,9.147,9.519,10.906,12.694],
                [155,8.111,8.578,9.172,9.543,10.933,12.725],
                [160,8.133,8.601,9.195,9.569,10.966,12.759],
                [165,8.155,8.625,9.222,9.596,10.992,12.793],
                [170,8.174,8.643,9.241,9.615,11.016,12.82],
                [175,8.195,8.666,9.265,9.641,11.046,12.851],
                [180,8.214,8.685,9.283,9.658,11.062,12.872],
                [185,8.233,8.706,9.307,9.683,11.089,12.904],
                [190,8.252,8.725,9.325,9.701,11.11,12.93],
                [195,8.268,8.741,9.343,9.72,11.132,12.956],
                [200,8.286,8.761,9.364,9.741,11.156,12.982],
                [225,8.361,8.838,9.446,9.826,11.247,13.083],
                [250,8.429,8.908,9.516,9.898,11.329,13.175],
                [275,8.489,8.97,9.581,9.966,11.399,13.248],
                [300,8.54,9.022,9.635,10.02,11.46,13.326],
                [325,8.587,9.07,9.685,10.071,11.517,13.389],
                [350,8.633,9.117,9.732,10.118,11.565,13.44],
                [375,8.67,9.157,9.775,10.161,11.613,13.494],
                [400,8.706,9.193,9.814,10.202,11.654,13.542],
                [425,8.738,9.224,9.844,10.234,11.692,13.58],
                [450,8.771,9.26,9.882,10.272,11.73,13.623],
                [475,8.798,9.288,9.912,10.302,11.761,13.655],
                [500,8.828,9.317,9.939,10.33,11.795,13.69],
                [525,8.854,9.344,9.967,10.36,11.827,13.73],
                [550,8.878,9.369,9.995,10.386,11.854,13.751],
                [575,8.901,9.391,10.016,10.408,11.878,13.782],
                [600,8.923,9.414,10.04,10.431,11.904,13.813],
                [650,8.963,9.455,10.083,10.476,11.949,13.856],
                [700,9.001,9.493,10.119,10.511,11.986,13.904],
                [750,9.033,9.524,10.152,10.547,12.026,13.947],
                [800,9.063,9.557,10.187,10.58,12.059,13.975],
                [850,9.093,9.587,10.216,10.612,12.096,14.023],
                [900,9.119,9.614,10.244,10.64,12.12,14.041],
                [950,9.143,9.638,10.269,10.665,12.149,14.07],
                [1000,9.168,9.664,10.295,10.692,12.176,14.105],
                [1100,9.211,9.708,10.339,10.736,12.22,14.15],
                [1200,9.246,9.745,10.377,10.775,12.263,14.197],
                [1300,9.283,9.781,10.415,10.812,12.304,14.235],
                [1400,9.313,9.812,10.446,10.845,12.34,14.271],
                [1500,9.347,9.846,10.481,10.88,12.374,14.312],
                [1600,9.372,9.871,10.506,10.904,12.396,14.339],
                [2000,9.464,9.965,10.603,11.002,12.5,14.443],
                [2500,9.551,10.052,10.69,11.089,12.591,14.54],
                [3000,9.618,10.121,10.76,11.161,12.664,14.619],
                [3500,9.675,10.178,10.818,11.219,12.727,14.683],
                [4000,9.727,10.229,10.869,11.271,12.779,14.734],
                [4500,9.766,10.269,10.911,11.313,12.82,14.777],
                [5000,9.803,10.307,10.948,11.349,12.859,14.817],
                [7500,9.938,10.442,11.085,11.487,12.997,14.959],
                [10000,10.031,10.537,11.18,11.584,13.095,15.063],
                [15000,10.152,10.658,11.302,11.707,13.221,15.186],
                [20000,10.236,10.743,11.388,11.791,13.305,15.271],
                [50000,10.48,10.988,11.634,12.039,13.556,15.523]))

CHGPT_CRITVALS = np.empty(chgpt_critvals.shape[0],dtype=[('n',np.float),('clevel90.0',np.float),
                                                ('clevel92.0',np.float),('clevel94.0',np.float),
                                                ('clevel95.0',np.float),('clevel97.5',np.float),('clevel99.0',np.float)])
CHGPT_CRITVALS['n'] = chgpt_critvals[:,0]
CHGPT_CRITVALS['clevel90.0'] = chgpt_critvals[:,1]
CHGPT_CRITVALS['clevel92.0'] = chgpt_critvals[:,2]
CHGPT_CRITVALS['clevel94.0'] = chgpt_critvals[:,3]
CHGPT_CRITVALS['clevel95.0'] = chgpt_critvals[:,4]
CHGPT_CRITVALS['clevel97.5'] = chgpt_critvals[:,5]
CHGPT_CRITVALS['clevel99.0'] = chgpt_critvals[:,6]


def calc_ioa(x, y):
    '''
    Calculate the index of agreement (Durre et al. 2010; Legates and McCabe 1999) between x and y
    '''
    
    y_mean = np.mean(y)
    d = np.sum(np.abs(x - y_mean) + np.abs(y - y_mean))
    
    if d == 0.0:
        #The x and y series are exactly the same
        #Return a perfect ioa
        return 1.0
    
    ioa = 1.0 - (np.sum(np.abs(y - x)) / d)
    
    return ioa

class NghMatrix(object):
    '''
    A class for building a data matrix of surrounding neighbor station observations for a 
    target station and performing imputation to determine the statistical distribution of a station's
    observations over a set time period (e.g.mean and variance).
    '''

    def __init__(self, stn_id, stn_da,stns_mask, tair_var,min_dist= -1, max_dist=MAX_DISTANCE):
        '''
        
        @param stn_id: the stn_id of the target
        @param stn_da: a StationDataDb object
        @param tair_var: the tair variable (tmin, tmax)
        @param min_dist: the min distance for which to search for neighbors (exclusive)
        @param max_dist: the max distance for which to search for neighbors (inclusive)
        @param nnr_ds: a NNRds for loading reanalysis data from nearest grid cells
        @param tair_mask: a mask for which observations at the target should be set to nan (default: None)
        '''
    
        stn = stn_da.stns[stn_da.stn_ids == stn_id][0]
        
        target_tair = stn_da.load_all_stn_obs_var(np.array([stn_id]), tair_var,set_flagged_nan=True)[0]
        target_tair = target_tair.astype(np.float64)
        
        #Number of observations threshold for entire period that is being infilled
        nthres_all = np.round(MIN_POR_OVERLAP * target_tair.size)
                
        #Number of observations threshold just for the target's period of record
        valid_tair_mask = np.isfinite(target_tair)
        ntair_valid = np.nonzero(valid_tair_mask)[0].size
        nthres_target_por = np.round(MIN_POR_OVERLAP * ntair_valid)    
        
        #Make sure to not include the target station itself as a neighbor station
        stns_mask = np.logical_and(stn_da.stns[STN_ID] != stn_id,stns_mask)
        all_stns = stn_da.stns[stns_mask]
        
        dists = utlg.grt_circle_dist(stn[LON], stn[LAT], all_stns[LON], all_stns[LAT])
        mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
        
        while np.nonzero(mask_dists)[0].size == 0:
            max_dist += MAX_DISTANCE / 2.0
            mask_dists = np.logical_and(dists <= max_dist, dists > min_dist)
                
        ngh_stns = all_stns[mask_dists]
        ngh_dists = dists[mask_dists]
        
        ngh_ids = ngh_stns[STN_ID]
        ngh_tair = stn_da.load_all_stn_obs_var(ngh_ids, tair_var, set_flagged_nan=True)[0]
        ngh_tair = ngh_tair.astype(np.float64)
        
        if len(ngh_tair.shape) == 1:
            ngh_tair.shape = (ngh_tair.size, 1) 
        
        dist_sort = np.argsort(ngh_dists)
        ngh_stns = ngh_stns[dist_sort]
        ngh_dists = ngh_dists[dist_sort]
        ngh_tair = ngh_tair[:, dist_sort]
        
        overlap_mask_tair = np.zeros(ngh_stns.size, dtype=np.bool)
        ioa = np.zeros(ngh_stns.size)
        
        for x in np.arange(ngh_stns.size):
            
            valid_ngh_mask = np.isfinite(ngh_tair[:, x])
            
            nlap = np.nonzero(valid_ngh_mask)[0].size
            
            overlap_mask = np.logical_and(valid_tair_mask, valid_ngh_mask)
            
            nlap_stn = np.nonzero(overlap_mask)[0].size
            
            #if nlap >= nthres_all and nlap_stn >= nthres_target_por:
            if nlap_stn >= nthres_target_por:
                
                ioa[x] = calc_ioa(target_tair[overlap_mask], ngh_tair[:, x][overlap_mask])
                overlap_mask_tair[x] = True
        
        ioa = ioa[overlap_mask_tair]
        ngh_dists = ngh_dists[overlap_mask_tair]
        ngh_tair = ngh_tair[:, overlap_mask_tair]
        
        if ioa.size > 0:
            
            ioa_sort = np.argsort(ioa)[::-1]
            ioa = ioa[ioa_sort]
            ngh_dists = ngh_dists[ioa_sort]
            ngh_tair = ngh_tair[:, ioa_sort]
            
            target_tair.shape = (target_tair.size, 1)
            
            imp_tair_mat = np.hstack((target_tair, ngh_tair))
            ngh_dists = np.concatenate((np.zeros(1), ngh_dists))
            ioa = np.concatenate((np.ones(1), ioa))
            
            valid_imp_mask = np.isfinite(imp_tair_mat)
            
            nnghs_per_day = np.sum(valid_imp_mask , axis=1)
        
        else:
            
            target_tair.shape = (target_tair.size, 1)  
            imp_tair_mat = target_tair
            
            valid_tair_mask.shape = (valid_tair_mask.size, 1) 
            valid_imp_mask = valid_tair_mask
            
            ioa = np.ones(1)
            ngh_dists = np.zeros(1)
            
            nnghs_per_day = np.zeros(target_tair.shape[0])        
        
        #############################################################
        self.imp_tair_mat = np.array(imp_tair_mat, dtype=np.float64)
        self.valid_imp_mask = valid_imp_mask
        self.ngh_ioa = ioa
        self.ngh_dists = ngh_dists
        self.max_dist = max_dist
        self.stn_id = stn_id
        self.stn_da = stn_da
        self.tair_var = tair_var
        self.nnghs_per_day = nnghs_per_day
        self.stns_mask = stns_mask
        self.stn = stn
    
    def extend_ngh_radius(self, extend_by):
        '''
        Extends the search radius for neighbor stations. The minimum of the search radius
        is the previous max distance.
        @param extend_by: The amount (km) by which to extend the radius by
        '''
        
        min_dist = self.max_dist
        max_dist = self.max_dist + extend_by

        imp_matrix2 = NghMatrix(self.stn_id, self.stn_da,self.stns_mask, self.tair_var, min_dist, max_dist)

        self.merge(imp_matrix2)
        self.max_dist = imp_matrix2.max_dist

    def merge(self, matrix2):
        '''
        Merges this ImputeMatrix with another ImputeMatrix
        @param matrix2: a ImputeMatrix
        '''    
    
        self.imp_tair_mat = np.hstack((self.imp_tair_mat, matrix2.imp_tair_mat[:, 1:]))
        self.valid_imp_mask = np.hstack((self.valid_imp_mask, matrix2.valid_imp_mask[:, 1:]))
        self.ngh_ioa = np.concatenate((self.ngh_ioa, matrix2.ngh_ioa[1:]))
        self.ngh_dists = np.concatenate((self.ngh_dists, matrix2.ngh_dists[1:]))
        
        if self.ngh_ioa.size > 0:
            
            ioa_sort = np.argsort(self.ngh_ioa[1:])[::-1]
            ioa_sort = np.concatenate([np.zeros(1, dtype=np.int), ioa_sort + 1])
            
            self.imp_tair_mat = self.imp_tair_mat[:, ioa_sort]
            self.valid_imp_mask = self.valid_imp_mask[:, ioa_sort]
            self.ngh_ioa = self.ngh_ioa[ioa_sort]
            self.ngh_dists = self.ngh_dists[ioa_sort]
            
            self.nnghs_per_day = np.sum(self.valid_imp_mask[:, 1:], axis=1)
    
        else:
            
            self.nnghs_per_day = np.zeros(self.imp_tair_mat.shape[1])

    def has_min_daily_nghs(self, nnghs):
        '''
        Checks to see if there is a minimum number of neighbor observations each day
        '''
        
        trim_valid_mask = self.valid_imp_mask[:, 0:1 + nnghs]
        nnghs_per_day = np.sum(trim_valid_mask[:, 1:], axis=1)
        
        return np.min(nnghs_per_day) >= MIN_DAILY_NGHBRS

    def get_ngh_matrix(self, nnghs=MIN_DAILY_NGHBRS):
        
        trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]
        
        engh_dly_nghs = self.has_min_daily_nghs(nnghs)
        actual_nnghs = trim_imp_tair_mat.shape[1] - 1
        
        while actual_nnghs < nnghs or not engh_dly_nghs:
        
            if actual_nnghs == nnghs and not engh_dly_nghs:
                
                nnghs += 1
            
            else:
                
                self.extend_ngh_radius(MAX_DISTANCE / 2.0)

            trim_imp_tair_mat = self.imp_tair_mat[:, 0:1 + nnghs]
            engh_dly_nghs = self.has_min_daily_nghs(nnghs)
            actual_nnghs = trim_imp_tair_mat.shape[1] - 1

        #############################################################
         
        obs_tair = trim_imp_tair_mat[:, 0]
        
        return trim_imp_tair_mat[:,1:], obs_tair 

class ChgPtDaily(object):
    
    def __init__(self, stn_da,stns_mask_tmin,stns_mask_tmax):
        
        self.stn_da = stn_da
        self.stns_mask = {'tmin':stns_mask_tmin,'tmax':stns_mask_tmax}
    
    def find_stn_chg_pt(self,stn_id,tair_var):
        
        ngh_matrix = NghMatrix(stn_id, self.stn_da, self.stns_mask[tair_var], tair_var)
        ngh_tair,obs_tair = ngh_matrix.get_ngh_matrix()
        
        mask_fin = np.isfinite(obs_tair)
        
        ymd_fin = self.stn_da.days[YMD][mask_fin]
        ymd_start = ymd_fin[0]
        ymd_end = ymd_fin[-1]
        ymd_mask = np.logical_and(self.stn_da.days[YMD] >= ymd_start,self.stn_da.days[YMD] <= ymd_end)
        
        ngh_tair = ngh_tair[ymd_mask,:]
        obs_tair = obs_tair[ymd_mask]
            
        ymd = self.stn_da.days[YMD][ymd_mask]
        
#        T = r.run_snht(robjects.Array(ymd),robjects.Array(obs_tair),robjects.Matrix(ngh_tair))
#        #T = r.run_snht(robjects.Array(uniq_yrs),robjects.Array(obs_tair_yrly),robjects.Matrix(ngh_tair_yrly))
#        T = np.array(T)
#        ymd_T = np.int64(T[:,0])
#        T = T[:,1]
        
        ymd_T,T,maxT,idx_maxT,ymd_maxT,sig,means = snhtQ(obs_tair, ngh_tair, 0.5, ymd) #Tail 1096 days (3 years)
        
        obs_tair_nona = obs_tair[np.in1d(ymd, ymd_T, True)]
    
        self.stn_id = stn_id
        self.stn_T = T
        self.stn_obs = obs_tair#obs_tair_yrly#obs_tair
        self.stn_obs_nona = obs_tair_nona#obs_tair_yrly#obs_tair_nona
        self.stn_startend = (ymd_start,ymd_end)
        self.stn_maxT = maxT
        self.stn_maxT_ymd = ymd_maxT
        self.stn_maxT_idx = idx_maxT
        self.stn_ymd = ymd_T
        self.stn_tair_var = tair_var
        self.stn_sigchgpt = sig
        self.stn_meanschgpt = means
    
    def find_stn_chg_pt_tmin_tmax(self,stn_id):
        
        self.find_stn_chg_pt(stn_id,'tmin')
        
        stn_T_tmin = self.stn_T
        stn_obs_tmin = self.stn_obs
        stn_obs_nona_tmin = self.stn_obs_nona
        stn_startend_tmin = self.stn_startend
        stn_maxT_tmin = self.stn_maxT
        stn_maxT_ymd_tmin = self.stn_maxT_ymd
        stn_maxT_idx_tmin = self.stn_maxT_idx
        stn_ymd_tmin = self.stn_ymd
        stn_sigchgpt = self.stn_sigchgpt
        stn_meanschgpt = self.stn_meanschgpt
        
        self.find_stn_chg_pt(stn_id,'tmax')
        
        self.stn_T = (stn_T_tmin,self.stn_T)
        self.stn_obs = (stn_obs_tmin,self.stn_obs)
        self.stn_obs_nona = (stn_obs_nona_tmin,self.stn_obs_nona)
        self.stn_startend = (stn_startend_tmin,self.stn_startend)
        self.stn_maxT = (stn_maxT_tmin,self.stn_maxT)
        self.stn_maxT_ymd = (stn_maxT_ymd_tmin,self.stn_maxT_ymd)
        self.stn_maxT_idx = (stn_maxT_idx_tmin,self.stn_maxT_idx)
        self.stn_ymd = (stn_ymd_tmin,self.stn_ymd)
        self.stn_sigchgpt = (stn_sigchgpt,self.stn_sigchgpt)
        self.stn_meanschgpt = (stn_meanschgpt,self.stn_meanschgpt)
        
        self.stn_tair_var = 'both'
    
    def print_chg_pt(self):
        
        print self.stn_da.stns[self.stn_id==self.stn_da.stn_ids]
        
        if self.stn_tair_var == "both":
            
            print "|".join(['Tmin MaxT',str(self.stn_maxT[0]),str(self.stn_sigchgpt[0]),str(self.stn_meanschgpt[0]),str(self.stn_maxT_ymd[0])])
            print "|".join(['Tmax MaxT',str(self.stn_maxT[1]),str(self.stn_sigchgpt[1]),str(self.stn_meanschgpt[1]),str(self.stn_maxT_ymd[1])])
        
        else:
            
            print "|".join([self.stn_tair_var,' MaxT',str(self.stn_maxT),self.stn_sigchgpt,self.stn_meanschgpt,str(self.stn_maxT_ymd)])
        
    
    def plot_chg_pt(self,nrows=3,ncols=2,startnum=1):
        
        if self.stn_tair_var == "both":
        
            plt.subplot(nrows,ncols,startnum)
            plt.title('tmin')
            plt.plot(self.stn_T[0])
            plt.subplot(nrows,ncols,startnum+1)
            plt.title('tmax')
            plt.plot(self.stn_T[1])
            
            
            plt.subplot(nrows,ncols,startnum+2)
            plt.plot(self.stn_obs_nona[0])
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx[0], ymin, ymax,'r')
            
            plt.subplot(nrows,ncols,startnum+3)
            plt.plot(self.stn_obs_nona[1])
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx[1], ymin, ymax,'r')
            
            plt.subplot(nrows,ncols,startnum+4)
            plt.plot(self.stn_obs[0])
            
            plt.subplot(nrows,ncols,startnum+5)
            plt.plot(self.stn_obs[1])
        
        else:  
        
            plt.subplot(nrows,ncols,startnum)
            plt.title(self.stn_tair_var)
            plt.plot(self.stn_T)
            plt.subplot(nrows,ncols,startnum+1)
            plt.plot(self.stn_obs)
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx, ymin, ymax,'r')

class ChgPtMthly(object):
    
    MIN_OVERLAP = 2.0/3.0
    
    def __init__(self, stn_da,stns_mask_tmin,stns_mask_tmax):
        
        self.stn_da = stn_da
        self.stns_mask = {'tmin':stns_mask_tmin,'tmax':stns_mask_tmax}
    
    def find_stn_chg_pt(self,stn_id,tair_var):
        
        ngh_matrix = NghMatrix(stn_id, self.stn_da, self.stns_mask[tair_var], tair_var)
        ngh_tair,obs_tair = ngh_matrix.get_ngh_matrix()
        
        mask_fin = np.isfinite(obs_tair)
        
        ymd_fin = self.stn_da.days[YMD][mask_fin]
        ymd_start = ymd_fin[0]
        ymd_end = ymd_fin[-1]
        ymd_mask = np.logical_and(self.stn_da.days[YMD] >= ymd_start,self.stn_da.days[YMD] <= ymd_end)
        
        ngh_tair = ngh_tair[ymd_mask,:]
        obs_tair = obs_tair[ymd_mask]
        mask_fin = np.isfinite(obs_tair)
        mask_fin_ngh = np.isfinite(ngh_tair)
        
        uniq_yrs = np.unique(self.stn_da.days[YEAR][ymd_mask])
        days = self.stn_da.days[ymd_mask]

        obs_tair_mthly = np.zeros(uniq_yrs.size*12)*np.nan
        ngh_tair_mthly = np.zeros((uniq_yrs.size*12,ngh_tair.shape[1]))*np.nan
        
        x = 0
        yrmths = []
        for yr in uniq_yrs:
            
            for mth in np.arange(1,13):
                
                mask_yrmth = np.logical_and(days[YEAR]==yr,days[MONTH]==mth)
                ndays = np.sum(mask_yrmth)
                
                mask_yrmthfin = np.logical_and(mask_yrmth,mask_fin)
                
                n = np.sum(mask_yrmthfin)
                
                if ndays > 0 and n >= self.MIN_OVERLAP*ndays:
                    
                    obs_tair_mthly[x] = np.mean(obs_tair[mask_yrmthfin])
                      
                for i in np.arange(ngh_tair.shape[1]):
                    
                    mask_yrmth_ngh = np.logical_and(mask_yrmth,mask_fin_ngh[:,i])
                    
                    mask_overlap = np.logical_and(mask_yrmthfin,mask_yrmth_ngh)
                    
                    if np.sum(mask_overlap) > (self.MIN_OVERLAP*n):
                        
                        ngh_tair_mthly[x,i] = np.mean(ngh_tair[mask_overlap,i])
                     
                x+=1
                yrmths.append("".join([str(yr),'%02d'%(mth,)]))
                    
        yrmths = np.array(yrmths).astype(np.int64)   
        
        ymd_T,T,maxT,idx_maxT,ymd_maxT,sig,means = snhtQ(obs_tair_mthly, ngh_tair_mthly, 0.5, yrmths)
        
        obs_tair_nona = obs_tair_mthly[np.in1d(yrmths, ymd_T, True)]
    
        self.stn_id = stn_id
        self.stn_T = T
        self.stn_obs = obs_tair_mthly
        self.stn_obs_nona = obs_tair_nona
        self.stn_startend = (ymd_start,ymd_end)
        self.stn_maxT = maxT
        self.stn_maxT_ymd = ymd_maxT
        self.stn_maxT_idx = idx_maxT
        self.stn_ymd = ymd_T
        self.stn_tair_var = tair_var
        self.stn_sigchgpt = sig
        self.stn_meanschgpt = means
        
    def find_stn_chg_pt_tmin_tmax(self,stn_id):
        
        self.find_stn_chg_pt(stn_id,'tmin')
        
        stn_T_tmin = self.stn_T
        stn_obs_tmin = self.stn_obs
        stn_obs_nona_tmin = self.stn_obs_nona
        stn_startend_tmin = self.stn_startend
        stn_maxT_tmin = self.stn_maxT
        stn_maxT_ymd_tmin = self.stn_maxT_ymd
        stn_maxT_idx_tmin = self.stn_maxT_idx
        stn_ymd_tmin = self.stn_ymd
        stn_sigchgpt = self.stn_sigchgpt
        stn_meanschgpt = self.stn_meanschgpt
        
        self.find_stn_chg_pt(stn_id,'tmax')
        
        self.stn_T = (stn_T_tmin,self.stn_T)
        self.stn_obs = (stn_obs_tmin,self.stn_obs)
        self.stn_obs_nona = (stn_obs_nona_tmin,self.stn_obs_nona)
        self.stn_startend = (stn_startend_tmin,self.stn_startend)
        self.stn_maxT = (stn_maxT_tmin,self.stn_maxT)
        self.stn_maxT_ymd = (stn_maxT_ymd_tmin,self.stn_maxT_ymd)
        self.stn_maxT_idx = (stn_maxT_idx_tmin,self.stn_maxT_idx)
        self.stn_ymd = (stn_ymd_tmin,self.stn_ymd)
        self.stn_sigchgpt = (stn_sigchgpt,self.stn_sigchgpt)
        self.stn_meanschgpt = (stn_meanschgpt,self.stn_meanschgpt)
        
        self.stn_tair_var = 'both'
    
    def print_chg_pt(self):
        
        print self.stn_da.stns[self.stn_id==self.stn_da.stn_ids]
        
        if self.stn_tair_var == "both":
            
            print "|".join(['Tmin MaxT',str(self.stn_maxT[0]),str(self.stn_sigchgpt[0]),str(self.stn_meanschgpt[0]),str(self.stn_maxT_ymd[0])])
            print "|".join(['Tmax MaxT',str(self.stn_maxT[1]),str(self.stn_sigchgpt[1]),str(self.stn_meanschgpt[1]),str(self.stn_maxT_ymd[1])])
        
        else:
            
            print "|".join([self.stn_tair_var,' MaxT',str(self.stn_maxT),self.stn_sigchgpt,self.stn_meanschgpt,str(self.stn_maxT_ymd)])
        
    
    def plot_chg_pt(self,nrows=3,ncols=2,startrow=1,startnum=1):
        
        if self.stn_tair_var == "both":
        
            plt.subplot(nrows,ncols,startnum)
            plt.title('tmin')
            plt.plot(self.stn_T[0])
            plt.subplot(nrows,ncols,startnum+1)
            plt.title('tmax')
            plt.plot(self.stn_T[1])
            
            
            plt.subplot(nrows,ncols,startnum+2)
            plt.plot(self.stn_obs_nona[0])
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx[0], ymin, ymax,'r')
            
            plt.subplot(nrows,ncols,startnum+3)
            plt.plot(self.stn_obs_nona[1])
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx[1], ymin, ymax,'r')
            
            plt.subplot(nrows,ncols,startnum+4)
            plt.plot(self.stn_obs[0])
            
            plt.subplot(nrows,ncols,startnum+5)
            plt.plot(self.stn_obs[1])
        
        else:  
        
            plt.subplot(nrows,ncols,startnum)
            plt.title(self.stn_tair_var)
            plt.plot(self.stn_T)
            plt.subplot(nrows,ncols,startnum+1)
            plt.plot(self.stn_obs)
            ymin,ymax = plt.ylim()
            plt.vlines(self.stn_maxT_idx, ymin, ymax,'r')
        
        
def update_qa_flags(fpath_db,fpath_chgpts):

    stn_da = StationDataDb(fpath_db,mode="r+")
    ymd_start = stn_da.days[YMD][0]
    ymd_end = stn_da.days[YMD][-1]

    chgpts = np.loadtxt(fpath_chgpts, dtype=DTYPE_CHGPTS, delimiter=",",skiprows=1)
    
    for chgpt in chgpts:
        
        print chgpt
        for tairvar in ['tmin','tmax']:
            
            chgpt_start = chgpt["".join([tairvar,CHGPT_START])]
            chgpt_end = chgpt["".join([tairvar,CHGPT_END])]
            
            if chgpt_start == 0:
                chgpt_start = ymd_start
            if chgpt_end == 1:
                chgpt_end = ymd_end
                
            tair,qaflags = stn_da.load_all_stn_obs_var(chgpt[STN_ID], tairvar)
        
            mask_chgpt = np.logical_and(stn_da.days[YMD] >= chgpt_start,stn_da.days[YMD] <= chgpt_end)
            mask_fin = np.isfinite(tair)
            mask_flgs = qaflags == ""
            
            mask_chpt_flg = np.logical_and(np.logical_and(mask_chgpt,mask_fin),mask_flgs)
            idx_chgpt_flg = np.nonzero(mask_chpt_flg)[0]
            
            if idx_chgpt_flg.size > 0:
            
                idx_stnid = stn_da.stn_idxs[chgpt[STN_ID]]
                
                stn_da.ds.variables["".join(["qflag_",tairvar])][idx_chgpt_flg,idx_stnid] = QA_CHGPT_FLG
                stn_da.ds.sync()
                
                print "Updated flags for: "+chgpt[STN_ID]
            
            else:
                
                print "No flags to update for: "+chgpt[STN_ID]

def reset_qa_flags(fpath_db,fpath_chgpts):
    
    stn_da = StationDataDb(fpath_db,mode="r+")

    chgpts = np.loadtxt(fpath_chgpts, dtype=DTYPE_CHGPTS, delimiter=",",skiprows=1)
    
    for chgpt in chgpts:
        
        print chgpt
        for tairvar in ['tmin','tmax']:
                            
            tair,qaflags = stn_da.load_all_stn_obs_var(chgpt[STN_ID], tairvar)
        
            mask_flgs = qaflags == QA_CHGPT_FLG
            idx_chgpt_flg = np.nonzero(mask_flgs)[0]
            
            if idx_chgpt_flg.size > 0:
            
                idx_stnid = stn_da.stn_idxs[chgpt[STN_ID]]
                
                stn_da.ds.variables["".join(["qflag_",tairvar])][idx_chgpt_flg,idx_stnid] = ""
                stn_da.ds.sync()
                
                print "Updated flags for: "+chgpt[STN_ID]
            
            else:
                
                print "No flags to update for: "+chgpt[STN_ID]

def snhtQ(obs,nghs,min_cor,days,clevel=95.0,nomit_tails=5):
    
    cors = np.zeros(nghs.shape[1])
    
    obs_ma = np.ma.masked_array(obs,mask=np.isnan(obs))
    nghs_ma = np.ma.masked_array(nghs,mask=np.isnan(nghs))
    
    obs_dif_ma = np.diff(obs_ma)
    nghs_dif_ma = np.diff(nghs_ma,axis=0)
     
    for x in np.arange(cors.size):
        cors[x] = np.ma.corrcoef(obs_dif_ma, nghs_dif_ma[:,x],allow_masked=True)[0,1]
    nghs_ma = nghs_ma[:,cors>=min_cor]
    nghs = nghs_ma.data
    cors = cors[cors>=min_cor]
    #Calculate means for common period
    ##############################################
    
    
    obs.shape = (obs.size,1)
    all_obs = np.hstack((obs,nghs))
    mask_fin = np.isfinite(all_obs)
    nfin = np.sum(mask_fin,axis=1)
    mask_period = nfin == all_obs.shape[1]
    obs = np.ravel(obs)
    
    mean_obs = np.mean(obs[mask_period])
    mean_ngh = np.mean(nghs[mask_period,:],axis=0)
    ##############################################
    
#    mean_obs = np.ma.mean(obs_ma)
#    mean_ngh = np.ma.mean(nghs_ma,axis=0)
    
    tmp1 = cors**2 * (nghs_ma - mean_ngh + mean_obs)
    
#    tmp1 = np.zeros((obs.size,mean_ngh.size))
#    
#    for j in np.arange(mean_ngh.size):
#        for i in np.arange(obs.size):
#            tmp1[i,j] = cors[j]**2 * (nghs_ma[i,j] - mean_ngh[j] + mean_obs)
    
    #tmp1 = np.ma.masked_array(tmp1,mask=np.isnan(tmp1))
    
    Q = obs - np.ma.sum(tmp1,axis=1)/np.sum(cors**2)
    Q = Q.data
    
    mask_q = np.isfinite(Q)
    Q = Q[mask_q]
    days = days[mask_q]
    
    Qmean = np.mean(Q)
    Qstd = np.std(Q,ddof=1)
    
    Z = (Q-Qmean)/Qstd
    
#    plt.plot(Z)
#    plt.show()
    
    n = Q.size
    
    Tv = np.zeros(n-1)
    zmeans = np.zeros((n-1,2))
    
    for v in np.arange(n-1):
        
        z1 = np.mean(Z[0:v+1])
        z2 = np.mean(Z[v+1:])
        
        Tv[v] = ((v+1)*(z1**2)) + ((n-(v+1))*(z2**2))
        
        zmeans[v,:] = (z1,z2)
    
    # Test Statistic (omit tail ends +-5 years)
    T0  = np.max(Tv[nomit_tails:-nomit_tails-1])
    T0x = np.nonzero(Tv==T0)[0][0]
    T0xa = days[T0x]
    
    crit_idx = np.nonzero(n > CHGPT_CRITVALS['n'])[0][-1]+1
    if crit_idx == CHGPT_CRITVALS['n'].size:
        crit_idx = crit_idx - 1
    
    Tc = CHGPT_CRITVALS["".join(['clevel',str(clevel)])][crit_idx]
    
    sig = True if T0 > Tc else False
    
    mean1 = (zmeans[T0x,0])*Qstd+Qmean
    mean2 = (zmeans[T0x,1])*Qstd+Qmean
        
    return days,Tv,T0,T0x,T0xa,sig,(mean1,mean2)
     
if __name__ == '__main__':
    
    reset_qa_flags('/projects/daymet2/station_data/all/all_1948_2012.nc', 
                    '/projects/daymet2/station_data/all/qa_change_pts2.csv')
    
#    update_qa_flags('/projects/daymet2/station_data/all/all_1948_2012.nc', 
#                    '/projects/daymet2/station_data/all/qa_change_pts2.csv')