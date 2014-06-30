'''
Utilities for analyzing and producing station period-of-record information.
'''
__all__ = ['build_valid_por_masks','load_por_csv','output_por_csv']

from station_data import STN_ID,PRCP,TMIN,TMAX,STATE,STN_NAME,LON,LAT,ELEV,MONTH,DAY,YMD,StationDataDb
import numpy as np
from datetime import timedelta,datetime
import matplotlib.mlab as mlab
from twx.utils import StatusCheck


MONTHS = np.arange(1,13)
MONTH_NUM_DAYS = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
MTH_SRT_END_DATES = {1:(datetime(2003,1,1),datetime(2003,1,31)),
                     2:(datetime(2003,2,1),datetime(2003,2,28)),
                     3:(datetime(2003,3,1),datetime(2003,3,31)),
                     4:(datetime(2003,4,1),datetime(2003,4,30)),
                     5:(datetime(2003,5,1),datetime(2003,5,31)),
                     6:(datetime(2003,6,1),datetime(2003,6,30)),
                     7:(datetime(2003,7,1),datetime(2003,7,31)),
                     8:(datetime(2003,8,1),datetime(2003,8,31)),
                     9:(datetime(2003,9,1),datetime(2003,9,30)),
                     10:(datetime(2003,10,1),datetime(2003,10,31)),
                     11:(datetime(2003,11,1),datetime(2003,11,30)),
                     12:(datetime(2003,12,1),datetime(2003,12,31))}


MIN_POR = 5 #in years
MIN_POR_PCT = 0.95

POR_TMIN_1 = "POR_TMIN_1"
POR_TMIN_2 = "POR_TMIN_2"
POR_TMIN_3 = "POR_TMIN_3"
POR_TMIN_4 = "POR_TMIN_4"
POR_TMIN_5 = "POR_TMIN_5"
POR_TMIN_6 = "POR_TMIN_6"
POR_TMIN_7 = "POR_TMIN_7"
POR_TMIN_8 = "POR_TMIN_8"
POR_TMIN_9 = "POR_TMIN_9"
POR_TMIN_10 = "POR_TMIN_10"
POR_TMIN_11 = "POR_TMIN_11"
POR_TMIN_12 = "POR_TMIN_12"

POR_TMAX_1 = "POR_TMAX_1"
POR_TMAX_2 = "POR_TMAX_2"
POR_TMAX_3 = "POR_TMAX_3"
POR_TMAX_4 = "POR_TMAX_4"
POR_TMAX_5 = "POR_TMAX_5"
POR_TMAX_6 = "POR_TMAX_6"
POR_TMAX_7 = "POR_TMAX_7"
POR_TMAX_8 = "POR_TMAX_8"
POR_TMAX_9 = "POR_TMAX_9"
POR_TMAX_10 = "POR_TMAX_10"
POR_TMAX_11 = "POR_TMAX_11"
POR_TMAX_12 = "POR_TMAX_12"

POR_PRCP_1 = "POR_PRCP_1"
POR_PRCP_2 = "POR_PRCP_2"
POR_PRCP_3 = "POR_PRCP_3"
POR_PRCP_4 = "POR_PRCP_4"
POR_PRCP_5 = "POR_PRCP_5"
POR_PRCP_6 = "POR_PRCP_6"
POR_PRCP_7 = "POR_PRCP_7"
POR_PRCP_8 = "POR_PRCP_8"
POR_PRCP_9 = "POR_PRCP_9"
POR_PRCP_10 = "POR_PRCP_10"
POR_PRCP_11 = "POR_PRCP_11"
POR_PRCP_12 = "POR_PRCP_12"

POR_DTYPE = [(STN_ID,"<U16"),(STATE,"<U2"),(STN_NAME,"<U30"),(LON,np.float64),(LAT,np.float64),(ELEV,np.float64),
           (POR_TMIN_1,np.int32),(POR_TMIN_2,np.int32),(POR_TMIN_3,np.int32),(POR_TMIN_4,np.int32),
           (POR_TMIN_5,np.int32),(POR_TMIN_6,np.int32),(POR_TMIN_7,np.int32),(POR_TMIN_8,np.int32),
           (POR_TMIN_9,np.int32),(POR_TMIN_10,np.int32),(POR_TMIN_11,np.int32),(POR_TMIN_12,np.int32),
           (POR_TMAX_1,np.int32),(POR_TMAX_2,np.int32),(POR_TMAX_3,np.int32),(POR_TMAX_4,np.int32),
           (POR_TMAX_5,np.int32),(POR_TMAX_6,np.int32),(POR_TMAX_7,np.int32),(POR_TMAX_8,np.int32),
           (POR_TMAX_9,np.int32),(POR_TMAX_10,np.int32),(POR_TMAX_11,np.int32),(POR_TMAX_12,np.int32),
           (POR_PRCP_1,np.int32),(POR_PRCP_2,np.int32),(POR_PRCP_3,np.int32),(POR_PRCP_4,np.int32),
           (POR_PRCP_5,np.int32),(POR_PRCP_6,np.int32),(POR_PRCP_7,np.int32),(POR_PRCP_8,np.int32),
           (POR_PRCP_9,np.int32),(POR_PRCP_10,np.int32),(POR_PRCP_11,np.int32),(POR_PRCP_12,np.int32)]

STN_CHK_SIZE = 250

CONUS_US_BOUNDS = (-126.0,-64.0,22.0,53.0)


def _get_mth_num_obs(obs,days,mth_buffer=0,clim_vars=[TMIN,TMAX,PRCP]):
    '''
    Calculates the number of observations for a variable for each month
    
    Parameters
    ----------
    obs : dict
        A dictionary of station observations 
        (obs[clim_var] = clim_var numpy time series).
    days : structured array
        A days array produced by twx.utils.get_days_metadata
    mth_buffer : int, optional
        The number of days the tails should be expanded when calculating
        the number of observations.
    clim_vars : sequence
        A list of variables that should be considered.
        
    Returns
    -------
    num_obs : dict
        A dict containing the # of obs in each month 
        (num_obs[clim_var][mth] = # of obs). 
    '''
    
    num_obs = {}
    for var in clim_vars:
        num_obs[var] = np.zeros(12)

    for mth in MONTHS:
        
        mth_date_str,mth_date_end = MTH_SRT_END_DATES[mth]
       
        str_date =  mth_date_str-timedelta(days=mth_buffer)
        end_date =  mth_date_end+timedelta(days=mth_buffer)
        
        mask_date = np.logical_and(days[MONTH]==str_date.month,days[DAY]>=str_date.day) 
        mask_date = np.logical_or(mask_date,days[MONTH]==mth)
        mask_date = np.logical_or(mask_date,np.logical_and(days[MONTH]==end_date.month,days[DAY]<=end_date.day))
        
        for var in clim_vars:
            mask_valid = np.logical_not(np.isnan(obs[var]))
            num_obs[var][mth-1] = np.nonzero(np.logical_and(mask_date,mask_valid))[0].size
    
    return num_obs


def output_por_csv(stn_da,stns,path):
    '''
    Produce an output csv file summarizing Tmin and Tmax observation 
    period-of-record information for a set of stations. TODO: Fully 
    remove prcp. This function used to return prcp period-of-record information. 
    It no longer does, but prcp columns are still written, so existing
    code is not broken. The prcp columns are just Tmin period-of-record.
    
    Parameters
    ----------
    stn_da : twx.db.StationDataDb
        A StationDataDb for the netCDF4 dataset containing
        the observations.
    stns : structured ndarray
        A structured array of stations from twx.db.StationDataDb.
        (can be a subset of all stations in dataset).
    set_flagged_nan : bool
        If true, any QA-flagged observations will be set to nan
    path : str
        The file path for the output csv file
    '''
    
    print "building period of records..."
    
    por_results = np.recarray(stns.size,dtype=POR_DTYPE)
    stat_chk = StatusCheck(stns.size,250)
    
    for x in np.arange(0,stns.size,STN_CHK_SIZE):
    
        if x + STN_CHK_SIZE < stns.size:
            nchk = STN_CHK_SIZE
        else:
            nchk = stns.size - x
    
        stns_chk = stns[x:x+nchk]

        tmin = stn_da.load_all_stn_obs_var(stns_chk[STN_ID],'tmin')[0]
        tmax = stn_da.load_all_stn_obs_var(stns_chk[STN_ID],'tmax')[0]
        prcp = stn_da.load_all_stn_obs_var(stns_chk[STN_ID],'tmin')[0]
    
        for i in np.arange(stns_chk.size):
            
            stn = stns_chk[i]
            stn_obs = {}
            stn_obs[TMIN] = tmin[:,i]
            stn_obs[TMAX] = tmax[:,i]
            stn_obs[PRCP] = prcp[:,i]
            
            stn_name = stn[STN_NAME]
            stn_name = stn_name.replace(","," ")
            
            num_obs = _get_mth_num_obs(stn_obs, stn_da.days)
            por_results[x+i] = (stn[STN_ID],stn[STATE],stn_name,stn[LON],stn[LAT],stn[ELEV],
                              num_obs[TMIN][0],num_obs[TMIN][1],num_obs[TMIN][2],num_obs[TMIN][3],
                              num_obs[TMIN][4],num_obs[TMIN][5],num_obs[TMIN][6],num_obs[TMIN][7],
                              num_obs[TMIN][8],num_obs[TMIN][9],num_obs[TMIN][10],num_obs[TMIN][11],
                              num_obs[TMAX][0],num_obs[TMAX][1],num_obs[TMAX][2],num_obs[TMAX][3],
                              num_obs[TMAX][4],num_obs[TMAX][5],num_obs[TMAX][6],num_obs[TMAX][7],
                              num_obs[TMAX][8],num_obs[TMAX][9],num_obs[TMAX][10],num_obs[TMAX][11],
                              num_obs[PRCP][0],num_obs[PRCP][1],num_obs[PRCP][2],num_obs[PRCP][3],
                              num_obs[PRCP][4],num_obs[PRCP][5],num_obs[PRCP][6],num_obs[PRCP][7],
                              num_obs[PRCP][8],num_obs[PRCP][9],num_obs[PRCP][10],num_obs[PRCP][11])
            stat_chk.increment()
    
    print "writing period of records..."
    mlab.rec2csv(por_results,path) 

def build_valid_por_masks(por_results,min_por=MIN_POR,loc_bounds=CONUS_US_BOUNDS):
    '''
    Build masks of stations that have minimum required of years of observations 
    for Tmin and Tmax in each month.
    
    Parameters
    ----------
    por_results : structured ndarray
        Period-of-record information from load_por_csv.
    min_por : int, optional
        The minimum period of record in years. The function tests
        whether a station has at least min_por years of data in each month.
    loc_bounds : tuple of 4 floats, optional
        The lat/lon bounds for stations that should be considered. If a station is
        outside the bounds, it will be marked has not meeting the period-of-record
        requirements. Bounds = (min lon, max lon, min lat, max lat)
        
    Returns
    -------
    mask_por_tmin : bool ndarray
        A station mask for Tmin. TRUE = period of record 
        meets minimum requirement.
    mask_por_tmax : bool ndarray
        A station mask for Tmax. TRUE = period of record 
        meets minimum requirement.  
    '''
    
    min_por1 = MONTH_NUM_DAYS[1]*min_por
    min_por2 = MONTH_NUM_DAYS[2]*min_por
    min_por3 = MONTH_NUM_DAYS[3]*min_por
    min_por4 = MONTH_NUM_DAYS[4]*min_por
    min_por5 = MONTH_NUM_DAYS[5]*min_por
    min_por6 = MONTH_NUM_DAYS[6]*min_por
    min_por7 = MONTH_NUM_DAYS[7]*min_por
    min_por8 = MONTH_NUM_DAYS[8]*min_por
    min_por9 = MONTH_NUM_DAYS[9]*min_por
    min_por10 = MONTH_NUM_DAYS[10]*min_por
    min_por11 = MONTH_NUM_DAYS[11]*min_por
    min_por12 = MONTH_NUM_DAYS[12]*min_por
    
    one_two = np.logical_and(por_results[POR_TMIN_1]>=min_por1,por_results[POR_TMIN_2]>=min_por2)
    three_four = np.logical_and(por_results[POR_TMIN_3]>=min_por3,por_results[POR_TMIN_4]>=min_por4)
    five_six = np.logical_and(por_results[POR_TMIN_5]>=min_por5,por_results[POR_TMIN_6]>=min_por6)
    seven_eight = np.logical_and(por_results[POR_TMIN_7]>=min_por7,por_results[POR_TMIN_8]>=min_por8)
    nine_ten = np.logical_and(por_results[POR_TMIN_9]>=min_por9,por_results[POR_TMIN_10]>=min_por10)
    eleven_twelve = np.logical_and(por_results[POR_TMIN_11]>=min_por11,por_results[POR_TMIN_12]>=min_por12)
    mask_por_tmin = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(one_two,three_four),five_six),seven_eight),nine_ten),eleven_twelve)
    
    one_two = np.logical_and(por_results[POR_TMAX_1]>=min_por1,por_results[POR_TMAX_2]>=min_por2)
    three_four = np.logical_and(por_results[POR_TMAX_3]>=min_por3,por_results[POR_TMAX_4]>=min_por4)
    five_six = np.logical_and(por_results[POR_TMAX_5]>=min_por5,por_results[POR_TMAX_6]>=min_por6)
    seven_eight = np.logical_and(por_results[POR_TMAX_7]>=min_por7,por_results[POR_TMAX_8]>=min_por8)
    nine_ten = np.logical_and(por_results[POR_TMAX_9]>=min_por9,por_results[POR_TMAX_10]>=min_por10)
    eleven_twelve = np.logical_and(por_results[POR_TMAX_11]>=min_por11,por_results[POR_TMAX_12]>=min_por12)
    mask_por_tmax = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(one_two,three_four),five_six),seven_eight),nine_ten),eleven_twelve)
    
#     one_two = np.logical_and(por_results[POR_PRCP_1]>=min_por1,por_results[POR_PRCP_2]>=min_por2)
#     three_four = np.logical_and(por_results[POR_PRCP_3]>=min_por3,por_results[POR_PRCP_4]>=min_por4)
#     five_six = np.logical_and(por_results[POR_PRCP_5]>=min_por5,por_results[POR_PRCP_6]>=min_por6)
#     seven_eight = np.logical_and(por_results[POR_PRCP_7]>=min_por7,por_results[POR_PRCP_8]>=min_por8)
#     nine_ten = np.logical_and(por_results[POR_PRCP_9]>=min_por9,por_results[POR_PRCP_10]>=min_por10)
#     eleven_twelve = np.logical_and(por_results[POR_PRCP_11]>=min_por11,por_results[POR_PRCP_12]>=min_por12)
#     mask_por_prcp = np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(one_two,three_four),five_six),seven_eight),nine_ten),eleven_twelve)
    
    #Temporary hack to not include SNOTEL AK stations, and non-GHCN Canada and Mexico stations
    mask_stnids = np.logical_and(por_results[STATE] != 'AK',
                                 np.logical_and(np.logical_not(np.char.startswith(por_results[STN_ID], prefix="CA_")),
                                                np.logical_not(np.char.startswith(por_results[STN_ID], prefix="MX_"))))
    
    if loc_bounds is not None:
        mask_loc = np.logical_and(np.logical_and(por_results[LON] >= loc_bounds[0], por_results[LON] <= loc_bounds[1]),np.logical_and(por_results[LAT] >= loc_bounds[2], por_results[LAT] <= loc_bounds[3])) 
    else:
        mask_loc = np.ones(mask_por_tmin.size,dtype=np.bool)
    
    mask_por_tmin = np.logical_and(np.logical_and(mask_por_tmin,mask_stnids),mask_loc)
    mask_por_tmax = np.logical_and(np.logical_and(mask_por_tmax,mask_stnids),mask_loc)
    
    #For now mask out RAWS stations for prcp due to possible freezing tipping bucket issues in winter
    #mask_raws = np.logical_not(np.char.startswith(por_results[STN_ID], prefix="RAWS_"))
    #mask_por_prcp = np.logical_and(np.logical_and(np.logical_and(mask_por_prcp,mask_stnids),mask_loc),mask_raws)
    
    return mask_por_tmin,mask_por_tmax#,mask_por_prcp

def load_por_csv(path):
    '''
    Loads a csv period of record summary file produced by output_por_csv into memory
    
    @param path: the file path for the output csv file
    '''

    por_results = np.genfromtxt(path, dtype=POR_DTYPE, comments=None, delimiter=",",skip_header=1)
    return por_results
    

if __name__ == '__main__':
    
    outpath = "/projects/daymet2/station_data/all/tairHomog_por_1948_2012.csv"
    stn_da = StationDataDb("/projects/daymet2/station_data/all/tairHomog_1948_2012.nc",
                               startend_ymd=(19480101,20121231))
    stns = stn_da.stns

    output_por_csv(stn_da, stns, outpath)
    
