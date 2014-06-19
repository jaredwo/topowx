'''
Driver for running quality assurance procedures in multiprocessing mode.

@author: jared.oyler
'''

import qa_prcp
import qa_temp
import numpy as np
from twx.db.station_data import StationDataDb, STN_ID, STATE, TMIN, TMAX, STN_NAME, LON, LAT, ELEV, PRCP, YMD, TMIN_FLAG, TMAX_FLAG, PRCP_FLAG
from twx.utils.multiprocess import multiprocess_config, multiprocess, worker
from netCDF4 import Dataset, date2num
import twx.utils.util_dates as utld
from twx.utils.util_misc import Unbuffered
import sys

sys.stdout=Unbuffered(sys.stdout)

NA_STATE = np.array(["NA"], dtype="<U2")[0]

GHCN_FLAGS_MAP = {qa_temp.QA_OK:"",
                  qa_temp.QA_MISSING:"",
                  qa_temp.DUP:"D",
                  qa_temp.QA_DUP_YEAR:"D",
                  qa_temp.QA_DUP_MONTH:"D",
                  qa_temp.QA_DUP_YEAR_MONTH:"D",
                  qa_temp.QA_DUP_WITHIN_MONTH:"D",
                  qa_temp.QA_GAP:"G",
                  qa_temp.QA_INTERNAL_INCONSIST:"I",
                  qa_temp.QA_STREAK:"K",
                  qa_prcp.QA_FREQUENT:"K",
                  qa_temp.QA_MEGA_INCONSIST:"M",
                  qa_temp.QA_NAUGHT:"N",
                  qa_temp.QA_CLIM_OUTLIER:"O",
                  qa_temp.QA_LAGRANGE_INCONSIST:"R",
                  qa_temp.QA_SPATIAL_REGRESS:"S",
                  qa_temp.QA_SPATIAL_CORROB:"S",
                  qa_temp.QA_SPIKE_DIP:"T",
                  qa_temp.QA_IMPOSS_VALUE:"X", }


TOPOMET_FLAGS_MAP = {"":qa_temp.QA_OK,
                     "D":qa_temp.DUP,
                     "G":qa_temp.QA_GAP,
                     "I":qa_temp.QA_INTERNAL_INCONSIST,
                     "K":qa_temp.QA_STREAK,
                     "M":qa_temp.QA_MEGA_INCONSIST,
                     "N":qa_temp.QA_NAUGHT,
                     "O":qa_temp.QA_CLIM_OUTLIER,
                     "R":qa_temp.QA_LAGRANGE_INCONSIST,
                     "S":qa_temp.QA_SPATIAL_REGRESS,
                     "T":qa_temp.QA_SPIKE_DIP,
                     "X":qa_temp.QA_IMPOSS_VALUE}

class multiprocess_qa(multiprocess):

    def __init__(self, config_multiprocess, config_worker):
        multiprocess.__init__(self, config_multiprocess, config_worker)

    def build_worker(self, worker_name, inq, outq, config_worker):
        return globals()[worker_name](inq, outq, config_worker)

class iter_flag_update:
       
    def __init__(self, stn_id, tmin_flags, tmax_flags, prcp_flags, ymd):
        self.stn_id = stn_id
        self.tmin_flags = tmin_flags
        self.tmax_flags = tmax_flags
        self.prcp_flags = prcp_flags
        self.ymd = ymd
        self.x = -1
        self.max = self.ymd.size - 1
        self.flag_map = GHCN_FLAGS_MAP

    def __iter__(self):
        return self

    def update_flags(self, curs):
        raise Exception('update_flags called on iter_flag_update')
        #curs.executemany("update observations set QFLAG_TMIN = ?, QFLAG_TMAX = ?, QFLAG_PRCP = ? where STN_ID = ? and YMD = ?",self)

    def next(self):
        self.x += 1
        if self.x > self.max:
            raise StopIteration
        else:
            return (self.flag_map[self.tmin_flags[self.x]], self.flag_map[self.tmax_flags[self.x]], self.flag_map[self.prcp_flags[self.x]], self.stn_id, long(self.ymd[self.x]))

class iter_multi_flag_update:
    
    def __init__(self):
        self.iters = []
        self.x = -1
    
    def __iter__(self):
        return self
    
    def add_iter(self, a_iter):
        self.iters.append(a_iter)
    
    def update_flags(self, db_path):
        self.max = len(self.iters) - 1
        self.x = 0
        self.cur_iter = self.iters[self.x]
        
        ds = Dataset(db_path, 'r+')
        stn_ids = ds.variables['stn_id'][:]
        time_units = ds.variables['time'].units
        
        print self.cur_iter.stn_id
        
        stn_id = ""
        stn_idx = 0
        while 1:
            
            try:
                #QFLAG_TMIN,QFLAG_TMAX,QFLAG_PRCP,STN_ID,YMD
                flag_row = self.next()
                
                if stn_id != flag_row[3]:
                    
                    stn_id = flag_row[3]
                    stn_idx = np.nonzero(stn_ids == stn_id)[0][0]
                 
                time_idx = int(date2num(utld.ymdL_to_date(flag_row[4]), units=time_units))
                
                ds.variables['qflag_tmin'][time_idx, stn_idx] = flag_row[0]
                ds.variables['qflag_tmax'][time_idx, stn_idx] = flag_row[1]
                ds.variables['qflag_prcp'][time_idx, stn_idx] = flag_row[2]
                
            except StopIteration:
                break
        
        ds.close()
        
        #curs.executemany("update observations set QFLAG_TMIN = ?, QFLAG_TMAX = ?, QFLAG_PRCP = ? where STN_ID = ? and YMD = ?",self)

    def next(self):
        try:
            return self.cur_iter.next()
        except StopIteration:
            self.x += 1
            if self.x > self.max:
                raise StopIteration
            else:
                while True:
                    try:
                        self.cur_iter = self.iters[self.x]
                        print self.cur_iter.stn_id
                        return self.cur_iter.next()
                    except StopIteration:
                        self.x += 1
                        if self.x > self.max:
                            raise StopIteration
    
class worker_qa(worker):

    def __init__(self, inq, outq, config):
        worker.__init__(self, inq, outq)
        
        self.stn_db_path = config.stn_db_path
        self.spatial_qa = config.spatial_qa
        self.init_db = False
    
    def init_stn_db_conns(self):

        stn_da = StationDataDb(self.stn_db_path)
        self.stn_da = stn_da
        self.init_db = True
    
    def create_update_iter(self, stn, flags_tmin, flags_tmax, flags_prcp, days, prev_flgs_tmin=np.array([]), prev_flgs_tmax=np.array([]), prev_flgs_prcp=np.array([])):
        
        mask_tmin = np.logical_not(np.logical_or(flags_tmin == qa_temp.QA_OK, flags_tmin == qa_temp.QA_MISSING))
        mask_tmax = np.logical_not(np.logical_or(flags_tmax == qa_temp.QA_OK, flags_tmax == qa_temp.QA_MISSING))
        mask_prcp = np.logical_not(np.logical_or(flags_prcp == qa_prcp.QA_OK, flags_prcp == qa_prcp.QA_MISSING))
        mask_all = np.logical_or(np.logical_or(mask_tmin, mask_tmax), mask_prcp)
        print "|".join([stn[STN_ID], str(np.nonzero(mask_all)[0].size)])
#        print "".join([stn[STN_ID]," ******************************"])
#        print "".join(["TMIN SPATIAL FLAG: ",str(np.nonzero(mask_tmin)[0].size)," | ",str(days[YMD][mask_tmin])])
#        print "".join(["TMAX SPATIAL FLAG: ",str(np.nonzero(mask_tmax)[0].size)," | ",str(days[YMD][mask_tmax])])
#        print "".join(["PRCP SPATIAL FLAG: ",str(np.nonzero(mask_prcp)[0].size)," | ",str(days[YMD][mask_prcp])])
        
        #make sure previous flags are not overwritten in the update
        flags_tmin = self.set_prev_flags(flags_tmin, prev_flgs_tmin)
        flags_tmax = self.set_prev_flags(flags_tmax, prev_flgs_tmax)
        flags_prcp = self.set_prev_flags(flags_prcp, prev_flgs_prcp)
        
        return iter_flag_update(stn[STN_ID], flags_tmin[mask_all], flags_tmax[mask_all], flags_prcp[mask_all], days[YMD][mask_all])
    
    def set_prev_flags(self, flags, prev_flgs):
        
        if prev_flgs.size > 0:
            
            uniq_flgs = np.unique(prev_flgs)
            
            for flag in uniq_flgs:
                
                if flag != "":
                
                    mask_flg = prev_flgs == flag
                    
                    flags[np.logical_and(mask_flg, np.logical_or(flags == qa_temp.QA_OK, flags == qa_temp.QA_MISSING))] = TOPOMET_FLAGS_MAP[flag]
        
        return flags
        
      
    def do_work(self, stn):
        
        if not self.init_db:
            self.init_stn_db_conns()
        
        while 1:
            try:    
                obs = self.stn_da.load_all_stn_obs(np.array([stn[STN_ID]]), set_flagged_nan=True)
                
                if self.spatial_qa:
                 
                    flags_tmin, flags_tmax = qa_temp.run_qa_spatial_only(stn, self.stn_da, obs[TMIN], obs[TMAX], self.stn_da.days)
                    flags_prcp = qa_prcp.run_qa_spatial_only(stn, self.stn_da, obs[PRCP], (obs[TMIN] + obs[TMAX]) / 2.0, self.stn_da.days)
                    #a_iter = self.create_update_iter(stn,flags_tmin,flags_tmax,flags_prcp,self.stn_da.days)
                
                else:
                    
                    flags_tmin, flags_tmax = qa_temp.run_qa_non_spatial(obs[TMIN], obs[TMAX], self.stn_da.days)
                    flags_prcp = qa_prcp.run_qa_non_spatial(obs[PRCP], (obs[TMIN] + obs[TMAX]) / 2.0, self.stn_da.days) 
        
                a_iter = self.create_update_iter(stn, flags_tmin, flags_tmax, flags_prcp, self.stn_da.days,
                                                 obs[TMIN_FLAG], obs[TMAX_FLAG], obs[PRCP_FLAG])
                return stn, a_iter
#            except RuntimeError:
#                print "".join(["HDF ERROR stn: ",stn[STN_ID],". Try again."])
#            except UnicodeDecodeError:
#                print "".join(["UNICODE ERROR stn: ",stn[STN_ID],". Try again."])
#            except FloatingPointError:
#                print "".join(["FloatingPointError ERROR stn: ",stn[STN_ID],". Try again."])
                
            except:
                print "".join(["Error stn: ", stn[STN_ID]])
                return stn, None

class worker_config():
    def __init__(self):
        self.stn_db_path = None
        self.spatial_qa = None       
        
class OutputHandler():
    
    def __init__(self, db_path):
        
        self.db_path = db_path
        self.iter_all = iter_multi_flag_update()
        
    def handleOutput(self, output):
        stn, a_iter = output
        
        if a_iter is not None:         
            self.iter_all.add_iter(a_iter)
    
    def commit_results(self):
        
        self.iter_all.update_flags(self.db_path)

#def update_results():
#    
#    DB_PATH= "/projects/daymet2/station_data/snotel_clean.db"
#    CONN = sql.connect(DB_PATH)
#    CURS = CONN.cursor()
#    file = open("/projects/daymet2/station_data/snotel_qa_spatial_results_error_stns.pyp")
#    iter_all = pickle.load(file)
#    iter_all.update_flags(CURS)
#    CONN.commit()

def run_qa_multiproc(spatial_qa, dp_path, nwrkrs=15):

    np.seterr(all='raise')
    np.seterr(under='ignore')
    
    stn_da = StationDataDb(dp_path)
    stns = stn_da.stns
    stn_da.ds.close()
    stn_da = None
    
    output_handler = OutputHandler(dp_path)
    
    processConfig = multiprocess_config()
    processConfig.numProcs = nwrkrs
    processConfig.workerName = "worker_qa"
    processConfig.inQueueLimit = None
    processConfig.outputHandler = output_handler
    processConfig.status_check_num = 50
    
    workerConfig = worker_config()
    workerConfig.stn_db_path = dp_path
    workerConfig.spatial_qa = spatial_qa
    
    multiProcess = multiprocess_qa(processConfig, workerConfig)
    
    for stn in stns:
        
        #if stn[STN_ID] == 'SNOTEL_21A09S':
        if stn[STN_ID].find("SNOTEL") != -1 or stn[STN_ID].find("GLAC") != -1 :
                
            #multiprocess pickle doesn't handle structured/record arrays--convert to dict stn[STN_ID],stn[STATE],stn[STN_NAME],stn[LON],stn[LAT],stn[ELEV],
            stn_dict = {STN_ID:stn[STN_ID],
                      STATE:NA_STATE if stn[STATE] == "" else stn[STATE],
                      STN_NAME:stn[STN_NAME],
                      LON:stn[LON],
                      LAT:stn[LAT],
                      ELEV:stn[ELEV]}
            multiProcess.process(stn_dict)
    
    multiProcess.handleOutputs()
    
#    for worker in multiProcess.workerProcs:
#        worker.stn_da.ds.close()
    
    multiProcess.terminate()
    output_handler.commit_results()
 
if __name__ == '__main__':
    
    run_qa_multiproc(True, '/projects/daymet2/station_data/all/all.nc', 15)
