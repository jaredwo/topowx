'''
Created on Mar 15, 2011

@author: jared.oyler
'''
import processing
import sqlite3 as sql
import qa_prcp as qa
import utils.util_dates as utld
import numpy as np
import sys
from datetime import datetime
import cProfile
from db.station_data import station_data,STN_ID,STATE,TMIN,TMAX,TMIN_FLAG,TMAX_FLAG,STN_NAME,LON,LAT,ELEV,PRCP,PRCP_FLAG
import time
from utils.multiprocess import multiprocess_config,multiprocess,worker
#from Multiprocess import worker,MultiProcess_Config,MultiProcess

NA_STATE = np.array(["NA"],dtype="<U2")[0]
GHCN_FLAGS_MAP = {"D":np.array([qa.DUP,qa.QA_DUP_YEAR,qa.QA_DUP_MONTH,qa.QA_DUP_YEAR_MONTH,qa.QA_DUP_WITHIN_MONTH]),
               "G":np.array([qa.QA_GAP]),
               "I":np.array([qa.QA_INTERNAL_INCONSIST]),
               "K":np.array([qa.QA_STREAK,qa.QA_FREQUENT]),
               "M":np.array([qa.QA_MEGA_INCONSIST]),
               "N":np.array([qa.QA_NAUGHT]),
               "O":np.array([qa.QA_CLIM_OUTLIER]),
               "R":np.array([qa.QA_LAGRANGE_INCONSIST]),
               "S":np.array([qa.QA_SPATIAL_REGRESS,qa.QA_SPATIAL_CORROB]),
               "T":np.array([qa.QA_SPIKE_DIP]),
               "X":np.array([qa.QA_IMPOSS_VALUE])}


class multiprocess_qa(multiprocess):

    def __init__(self,config_multiprocess,config_worker):
        multiprocess.__init__(self,config_multiprocess,config_worker)

    def build_worker(self,worker_name,inq,outq,config_worker):
        return globals()[worker_name](inq,outq,config_worker)

class worker_qa(worker):

    def __init__(self,inq,outq,config):
        worker.__init__(self, inq, outq)
        
        self.stn_db_path = config.stn_db_path

        self.init_stn_db_conns()
    
    def init_stn_db_conns(self):
        
        CONN = sql.connect(self.stn_db_path)
        CURS = CONN.cursor()

        stn_da = station_data(CURS)
        stn_da.load_stns()
        
        self.stn_da = stn_da
        
    def do_work(self,stn):
        
        try:    
            obs = self.stn_da.load_all_stn_obs(np.array([stn[STN_ID]]),set_flagged_nan=False)
            
            if obs[PRCP][np.logical_and(np.isfinite(obs[PRCP]),obs[PRCP]>0)].size < 100:
                return stn,None
            
            flags_prcp = qa.run_qa(stn, self.stn_da, obs[PRCP], (obs[TMIN]+obs[TMAX])/2.0, self.stn_da.days)   
            
            flag_cnts_prcp = {}
            for flag in GHCN_FLAGS_MAP.keys():
                
                num_flagged = np.nonzero(np.in1d(flags_prcp, GHCN_FLAGS_MAP[flag]))[0].size
                num_orig_flagged = np.nonzero(np.in1d(obs[PRCP_FLAG],np.array([flag])))[0].size
                flag_cnts_prcp[flag] = (num_orig_flagged,num_flagged)
            
            return stn,flag_cnts_prcp
        except:
            print "".join(["Error stn: ",stn[STN_ID]])
            return stn,None

class worker_config():
    def __init__(self):
        self.stn_db_path = None       
        
class OutputHandler():
    
    def __init__(self,):
        pass
        
    def handleOutput(self,output):
        stn,flag_cnts_prcp = output
        
        if flag_cnts_prcp is not None:
        
            print "".join([stn[STN_ID],"***************************************"])
            
            for flag in flag_cnts_prcp.keys():
                num_orig_flagged,num_flagged = flag_cnts_prcp[flag]
                if num_orig_flagged != 0:
                    pct_dif = float(num_flagged-num_orig_flagged)/float(num_orig_flagged)*100.0
                else:
                    pct_dif = "NA"
                print "".join([PRCP,"|",flag,": ",str(flag_cnts_prcp[flag]),"--",str(pct_dif),"%"])
            
            print "**************************************************************\n"

if __name__ == '__main__':
    
    np.seterr(all='raise')
    np.seterr(under='ignore')
    NUM_WORKERS = 1
    DB_PATH= "/projects/daymet2/station_data/ghcn.db"
    CONN = sql.connect(DB_PATH)
    CURS = CONN.cursor()
    
    stn_da = station_data(CURS)
    stns = stn_da.load_stns()
    stns = stns[stns[STATE]=='MT']
    #obs = stn_da.load_all_stn_obs(stns[STN_ID],set_flagged_nan=False)
    
    output_handler = OutputHandler()
    
    processConfig = multiprocess_config()
    processConfig.numProcs = NUM_WORKERS
    processConfig.workerName = "worker_qa"
    processConfig.inQueueLimit = None
    processConfig.outputHandler = output_handler
    processConfig.status_check_num = -1
    
    workerConfig = worker_config()
    workerConfig.stn_db_path = DB_PATH
    
    multiProcess = multiprocess_qa(processConfig, workerConfig)
    
    for stn in stns:
        
        if stn[STN_ID] == "GHCN_USC00242266":
            #multiprocess pickle doesn't handle structured/record arrays--convert to dict stn[STN_ID],stn[STATE],stn[STN_NAME],stn[LON],stn[LAT],stn[ELEV],
            stn_dict = {STN_ID:stn[STN_ID],
                      STATE:NA_STATE if stn[STATE] == "" else stn[STATE],
                      STN_NAME:stn[STN_NAME],
                      LON:stn[LON],
                      LAT:stn[LAT],
                      ELEV:stn[ELEV]}
            multiProcess.process(stn_dict)
    
    multiProcess.handleOutputs()
    multiProcess.terminate()