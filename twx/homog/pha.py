'''
Utilities for running the external Pairwise Homogenization Algorithm (PHA) software:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.

Copyright 2014, Jared Oyler.

This file is part of TopoWx.

TopoWx is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

TopoWx is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.
'''


__all__ = ['setup_pha', 'run_pha', 'HomogDaily', 'InsertHomog',
           'load_snotel_sensor_hist', 'get_pha_adj_df']

from datetime import datetime
from dateutil.relativedelta import relativedelta
from twx.db import LON, LAT, STN_ID, ELEV, STATE, STN_NAME, MISSING, DTYPE_STNOBS
from twx.utils import get_mth_metadata, YEAR, MONTH, YMD, DATE, DAY
import glob
import numpy as np
import os
import pandas as pd
import subprocess
import twx

INCL_LINE_BEGIN_YR = '        parameter (begyr = 1895)\n'
INCL_LINE_END_YR = '        parameter (endyr = 2015)\n'
INCL_LINE_N_STNS = '        parameter (maxstns = 7720)\n'
CONF_LINE_END_YR = 'endyr=1999\n'
CONF_LINE_MAX_YRS = 'maxyrs=500000\n'
CONF_LINE_ELEMS = 'elems="tavg"\n'

DTYPE_PHA_ADJ = [(STN_ID, "<S50"), ('ymd_start',np.int),('ymd_end',np.int),('adj', np.float64)]


def setup_pha(fpath_pha_tar, path_out_src, path_out_run, yr_begin, yr_end, stns,
              tair, varname, stnhist=None):
    '''
    Perform setup for running external Pairwise Homogenization Algorithm (PHA) software
    from NCDC. Setup has been tested against PHA v52i downloaded from 
    ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/v3/software/. Reference:
    
    Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
    via pairwise comparisons. J. Climate, 22, 1700-1717.

    Parameters
    ----------
    fpath_pha_tar : str
        File path to the main PHA tar.gz file (eg phav52i.tar.gz) downloaded from NCDC
    path_out_src : str
        File path to where PHA source code should be written.
    path_out_run : str
        File path where PHA will be executed
    yr_begin : int
        The start year for the PHA run
    yr_end : int
        The end year for the PHA run
    stns : structured ndarray
        Stations for which to run PHA. Structured station array must contain at
        least the following fields: STN_ID, LAT, LON and can be obtained from
        twx.db.StationDataDb
    tair : MaskedArray
        A 2-D numpy MaskedArray of monthly temperature observations of shape P*N
        where P is the number of months between yr_begin and yr_end and N is the
        number of stations. Each column is a station's time series and must be in
        same order as stns.
    varname : str
        Temperature variable name (tmin or tmax)
    stnhist : pandas.DataFrame, optional
        DataFrame of station history metadata: station_id as index and
        change_date column specifying the date of a station change.
    '''
        
    n_stns = stns.size
    
    n_stnyrs = (yr_end - yr_begin + 1) * n_stns
    yrs = np.arange(yr_begin, yr_end + 1)

    print "Uncompressing PHA..."
    subprocess.call(["tar", "-xzvf", fpath_pha_tar, '-C', path_out_src])

    fpath_incl = os.path.join(path_out_src, 'phav52i', 'source_expand',
                              'parm_includes', 'inhomog.parm.MTHLY.TEST.incl')
    f_incl = open(fpath_incl, 'r')
    incl_lines = f_incl.readlines()
    f_incl.close()

    for x in np.arange(len(incl_lines)):

        a_line = incl_lines[x]

        if a_line == INCL_LINE_BEGIN_YR:
            a_line = a_line.replace('1895', "%d" % (yr_begin,))
            incl_lines[x] = a_line
        elif a_line == INCL_LINE_END_YR:
            a_line = a_line.replace('2015', "%d" % (yr_end,))
            incl_lines[x] = a_line
        elif a_line == INCL_LINE_N_STNS:
            a_line = a_line.replace('7720', "%d" % (n_stns,))
            incl_lines[x] = a_line

    f_incl = open(fpath_incl, 'w')
    f_incl.writelines(incl_lines)
    f_incl.close()

    path_src = os.path.join(path_out_src, 'phav52i')
    os.chdir(path_src)

    print "Compiling PHA..."
    subprocess.call(['make', 'install', 'INSTALLDIR=%s' % (path_out_run,)])

    _write_conf(os.path.join(path_out_run, "world1.conf"), yr_end, n_stnyrs, varname)
    _write_conf(os.path.join(path_out_run, "data", "world1.conf"), yr_end, n_stnyrs, varname)

    print "Writing input station data ASCII files..."
    _write_input_station_data(path_out_run, varname, stns, tair, yrs, stnhist)
    
    
def run_pha(path_run, varname):
    '''
    Run a PHA instance

    Parameters
    ----------
    path_run : str
        The PHA run path from setup_pha
    varname : str
        Temperature variable name (tmin or tmax)
    '''

    pha_cmd = 'bash ' + os.path.join(path_run, 'testv52i-pha.sh') + "  world1 %s raw 0 0 P" % (varname,)
    print "Running PHA for %s..." % (varname,)
    subprocess.call(pha_cmd, shell=True)
    
    path_log = os.path.join(path_run,'data','benchmark','world1','output','PHAv52i.FAST.MLY.TEST.*.%s.world1.r00.out.gz'%(varname,))
    path_out_log = os.path.join(path_run,'data','benchmark','world1','output','pha_adj_%s.log'%(varname,))
    
    print "Writing log of PHA adjustments: "+path_out_log
    cmd = " ".join(["zgrep 'Adj write'",path_log,">",path_out_log])
    subprocess.call(cmd,shell=True)


class HomogDaily():
    '''
    Class for homogenizing daily station data based on monthly homogenization
    results from a PHA run.
    '''
    
    def __init__(self,stnda,path_pha_run,varname):
        '''
        Parameters
        ----------
        stnda : twx.db.StationDataDb
            A StationDataDb object pointing to the daily netCDF database
            that was used as input to the PHA run
        path_run : str
            The PHA run path from setup_pha
        varname : str
            Temperature variable name (tmin or tmax)
        '''
        
        self.stnda = stnda
        self.varname = varname

        self.mthly_data = np.ma.masked_invalid(self.stnda.xrds['_'.join([varname, 'mth'])][:].values)
        self.miss_data = self.stnda.ds.variables['_'.join([varname,'mthmiss'])][:]
        
        path_adj_log = os.path.join(path_pha_run,'data','benchmark','world1','output','pha_adj_%s.log'%(varname,))
        self.pha_adjs = _parse_pha_adj(path_adj_log)        
        
        self.path_FLs_data = os.path.join(path_pha_run,'data','benchmark','world1','monthly','FLs.r00')
        
        self.mths = get_mth_metadata(self.stnda.days[YEAR][0], self.stnda.days[YEAR][-1])
        
        self.dly_yrmth_masks = []
        
        yrs = np.unique(self.stnda.days[YEAR])
        
        for yr in yrs:
        
            for mth in np.arange(1,13):
                
                self.dly_yrmth_masks.append(np.logical_and(stnda.days[YEAR]==yr,stnda.days[MONTH]==mth))
        
        self.ndays_per_mth = np.zeros(len(self.dly_yrmth_masks))
        
        for x in np.arange(self.ndays_per_mth.size):
            self.ndays_per_mth[x] = np.sum(self.dly_yrmth_masks[x])
        
        self.mthly_yr_masks = {}
        for yr in yrs:
            self.mthly_yr_masks[yr] = self.mths[YEAR] == yr
           
    def homog_stn(self,stn_id):
        '''
        Build time series of homogenized daily temperature observations for a station
        
        Parameters
        ----------
        stn_id : str
            The ID of the station for which to build the homogenized time series

        Returns
        -------
        dly_vals_homog : ndarray
            The homogenized time series of daily temperature observations for the station.
        '''
        
        fstn_id = _format_stnid(stn_id)
        
        file_homog_mth = open(os.path.join(self.path_FLs_data,"%s.FLs.r00.%s"%(fstn_id,self.varname)))
                
        mthvals_homog = np.ones(self.mths.size,dtype=np.float)*-9999
        
        for aline in file_homog_mth.readlines():
            
            yr = int(aline[12:17])
            yrmthvals = np.array([aline[17:17+5],aline[26:26+5],aline[35:35+5],aline[44:44+5],
                       aline[53:53+5],aline[62:62+5],aline[71:71+5],aline[80:80+5],
                       aline[89:89+5],aline[98:98+5],aline[107:107+5],aline[116:116+5]],dtype=np.float)
            
            mthvals_homog[self.mthly_yr_masks[yr]] = yrmthvals
        
        mthvals_homog = np.ma.masked_array(mthvals_homog,mthvals_homog==-9999)
        mthvals_homog = mthvals_homog/100.0
        mthvals_homog = np.round(mthvals_homog,2)

        dly_vals = self.stnda.load_all_stn_obs_var(stn_id,self.varname)[0].astype(np.float64)
        dly_vals = np.ma.masked_array(dly_vals,np.isnan(dly_vals))        
        mth_vals = np.ma.round(self.mthly_data[:,self.stnda.stn_idxs[stn_id]].astype(np.float64),2)
        miss_cnts = self.miss_data[:,self.stnda.stn_idxs[stn_id]]
        dly_vals_homog = dly_vals.copy()
        
        stn_pha_adj = self.pha_adjs[self.pha_adjs[STN_ID]==fstn_id]
        stn_pha_adj = stn_pha_adj[np.argsort(stn_pha_adj['ymd_start'])]
        
        dif_cnt = 0
        
        for x in np.arange(mth_vals.size):
            
            if not np.ma.is_masked(mth_vals[x]) and not np.ma.is_masked(mthvals_homog[x]):
            
                if mth_vals[x] != mthvals_homog[x]:

                    delta = mthvals_homog[x] - mth_vals[x]                    
                    dly_vals_homog[self.dly_yrmth_masks[x]] = dly_vals_homog[self.dly_yrmth_masks[x]] + delta
                    dif_cnt+=1
            
            elif miss_cnts[x] < self.ndays_per_mth[x] and not np.ma.is_masked(mthvals_homog[x]):
                
                ymd = self.mths[YMD][x]
                
                if ymd < stn_pha_adj['ymd_start'][0]:
                    #before all change points. assume it falls under the earliest change point
                    delta = -stn_pha_adj['adj'][0]
                else:
                    
                    mask_adj = np.logical_and(stn_pha_adj['ymd_start'] <= ymd, stn_pha_adj['ymd_end'] >= ymd)
                    sum_mask = np.sum(mask_adj)
                    
                    if sum_mask == 0:
                        
                        #don't do anything. past the last change point which is theoretically 0
                        delta = 0
                    
                    elif sum_mask == 1:
                        
                        delta = -stn_pha_adj[mask_adj]['adj'][0]
                    
                    else:
                        
                        raise Exception("Falls within more than one change point")
                        
                dly_vals_homog[self.dly_yrmth_masks[x]] = dly_vals_homog[self.dly_yrmth_masks[x]] + np.round(delta,2)
                            
        return dly_vals_homog

class InsertHomog(twx.db.Insert):
    '''
    Class for inserting stations and observations that have been homogenized.
    Loads stations and observations from current netCDF
    station dataset and performs homogenization adjustments.
    '''
    
    def __init__(self,stnda,homog_dly_tmin,homog_dly_tmax,path_pha_run_tmin,path_pha_run_tmax):
        '''
        Parameters
        ----------
        stnda : twx.db.StationDataDb
            A StationDataDb object pointing to the daily netCDF database
            that was used as input to the PHA run
        homog_dly_tmin : HomogDaily
            A HomogDaily object for homogenizing daily Tmin observations
        homog_dly_tmax : HomogDaily
            A HomogDaily object for homogenizing daily Tmax observations
        path_pha_run_tmin : str
            The PHA run path from setup_pha for Tmin
        path_pha_run_tmax : str
            The PHA run path from setup_pha for Tmax
        '''
        
        twx.db.Insert.__init__(self,stnda.days[DATE][0],stnda.days[DATE][-1])
        
        self.homog_tmin = homog_dly_tmin
        self.homog_tmax = homog_dly_tmax
        self.stnda = stnda
        
        #Get stn_ids for which homogenization could not be conducted
        unuse_tmin_ids = self.__load_input_not_stnlist(path_pha_run_tmin)
        unuse_tmax_ids = self.__load_input_not_stnlist(path_pha_run_tmax)
        
        fmt_ids = np.array([_format_stnid(stnid) for stnid in stnda.stn_ids])
        
        mask_stns_tmin = ~np.in1d(fmt_ids, unuse_tmin_ids, True)
        mask_stns_tmax = ~np.in1d(fmt_ids, unuse_tmax_ids, True)
                
        self.stns_tmin = stnda.stns[mask_stns_tmin]
        self.stns_tmax = stnda.stns[mask_stns_tmax]
        
        uniq_ids = np.unique(np.concatenate((self.stns_tmin[STN_ID],self.stns_tmax[STN_ID])))
        
        self.stns_all = stnda.stns[np.in1d(stnda.stn_ids, uniq_ids, True)]
        
        self.stn_list = [(stn[STN_ID],stn[LAT],stn[LON],stn[ELEV],stn[STATE],stn[STN_NAME]) for stn in self.stns_all]
        
        self.empty_obs = np.ones(stnda.days.size)*MISSING
        self.empty_qa = np.zeros(stnda.days.size,dtype=np.str)
    
    def __load_input_not_stnlist(self,run_path):
        
        fpath_corr = os.path.join(run_path,'data','benchmark','world1','corr')
        fnames = np.array(os.listdir(fpath_corr))
        fnames_input_not_stnlist = fnames[np.char.endswith(fnames,'input_not_stnlist')]
        
        stnids_all = []
        
        for a_fname in fnames_input_not_stnlist:
            
            fname_input_not_stnlist = os.path.join(fpath_corr,a_fname)
            stn_ids = np.atleast_1d(np.loadtxt(fname_input_not_stnlist, dtype=np.str, usecols=[0]))
            stnids_all.extend(stn_ids)
        
        stnids_all = np.sort(np.array(stnids_all))
        
        return stnids_all
        
    def get_stns(self):
        return self.stn_list
                            
    def parse_stn_obs(self,stn_id):
        
        if stn_id in self.stns_tmin[STN_ID]:
            tmin_homog = self.homog_tmin.homog_stn(stn_id)
            tmin_homog = np.ma.filled(tmin_homog, MISSING)
        else:
            tmin_homog = self.empty_obs
        
        if stn_id in self.stns_tmax[STN_ID]:
            tmax_homog = self.homog_tmax.homog_stn(stn_id)
            tmax_homog = np.ma.filled(tmax_homog, MISSING)
        else:
            tmax_homog = self.empty_obs
                
        obs = np.empty(self.stnda.days.size, dtype=DTYPE_STNOBS)                
        obs['year'] = self.stnda.days[YEAR]
        obs['month'] = self.stnda.days[MONTH]
        obs['day'] = self.stnda.days[DAY]
        obs['ymd'] = self.stnda.days[YMD]
        obs['tmin'] = tmin_homog
        obs['tmax'] = tmax_homog
        obs['prcp'] = self.empty_obs
        obs['swe'] = self.empty_obs
        obs['qflag_tmin'] = self.empty_qa
        obs['qflag_tmax'] = self.empty_qa
        obs['qflag_prcp'] = self.empty_qa
                
        return obs

def _write_conf(fpath_conf, endyr, n_stnyrs, varname):
    '''
    Write an updated PHA configuration file
    
    Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
    via pairwise comparisons. J. Climate, 22, 1700-1717.

    Parameters
    ----------
    fpath_conf : str
        File path to the existing PHA config file (eg world1.conf)
    endyr : int
        Last year to run PHA
    n_stnyrs : int
        Number of station years (# of stations * # of years)
    varname : str
        Temperature variable name (tmin or tmax)
    '''

    f_conf = open(fpath_conf, 'r')
    conf_lines = f_conf.readlines()
    f_conf.close()

    for x in np.arange(len(conf_lines)):

        a_line = conf_lines[x]

        if a_line == CONF_LINE_END_YR:
            a_line = a_line.replace('1999', "%d" % (endyr,))
            conf_lines[x] = a_line
        elif a_line == CONF_LINE_MAX_YRS:
            a_line = a_line.replace('500000', "%d" % (n_stnyrs,))
            conf_lines[x] = a_line
        elif a_line == CONF_LINE_ELEMS:
            a_line = a_line.replace('tavg', varname)
            conf_lines[x] = a_line
            
    f_conf = open(fpath_conf, 'w')
    f_conf.writelines(conf_lines)
    f_conf.close()

def _write_input_station_data(path_pha_run, varname, stns, tair, yrs,
                              stnhist=None):
    '''
    Write station data to GHCN format for input to PHA
    '''
        
    _write_stn_list(stns, os.path.join(path_pha_run, 'data', 'benchmark',
                                       'world1', 'meta',
                                       'world1_stnlist.%s' % (varname,)))
    
    os.remove(os.path.join(path_pha_run, 'data', 'benchmark', 'world1', 'meta',
                           'world1_stnlist.tavg'))
    fpath_metafile = os.path.join(path_pha_run, 'data', 'benchmark', 'world1',
                                  'meta', 'world1_metadata_file.txt')
    
    with open(fpath_metafile, 'w') as f:
        
        if stnhist is not None:
            
            for stn_id, a_date in zip(stnhist.station_id,
                                      pd.DatetimeIndex(stnhist.change_date).strftime('%Y%m')):
                
                stn_id = _format_stnid(stn_id)
                f.write("  %s %s 1\n"%(stn_id, a_date))
        
    path_stn_obs = os.path.join(path_pha_run, 'data', 'benchmark', 'world1',
                                'monthly', 'raw')
    rm_stn_files = glob.glob(os.path.join(path_stn_obs, "*.tavg"))
    for a_fpath in rm_stn_files:
        os.remove(a_fpath)
        
    _write_stn_obs_files(stns, tair, yrs, varname, path_stn_obs)
        

def _write_stn_list(stns, fpath_out):
    '''
    Write GHCN format station list ASCII file
    '''

    fout = open(fpath_out, "w")

    for stn in stns:

        outid = _format_stnid(stn[STN_ID])

        if stn[LAT] < 0 or stn[LON] >= 0:
            raise Exception("Only handles formating of positive Lats and negative Lons.")

        outLat = "{0:0<8.5F}".format(stn[LAT])

        if np.abs(stn[LON]) < 100:
            fmtLon = "{0:0<9.5F}"
        else:
            fmtLon = "{0:0<9.4F}"

        outLon = fmtLon.format(stn[LON])

        fout.write(" ".join([outid, outLat, outLon, "\n"]))

def _format_stnid(stnid):
    '''
    Format station id for PHA
    '''

    if stnid.startswith("GHCND_"):

        outid = stnid.split("_")[1]

    elif stnid.startswith("NRCS_"):

        outid = stnid.split("_")[1]
        outid = "".join(["SNT", "{0:0>8}".format(outid)])

    elif stnid.startswith("RAWS_"):

        outid = stnid.split("_")[1]
        outid = "".join(["WRC", "{0:0>8}".format(outid)])
        
    elif stnid.startswith("USH"):

        outid = stnid

    else:

        raise Exception("Do not recognize stn id prefix for stnid: " + stnid)

    return outid

def _write_stn_obs_files(stns, data, yrs, varname, path_out):
    '''
    Write individual GHCN format station observation ASCII files 
    '''

    for stn, x in zip(stns, np.arange(stns.size)):

        outId = _format_stnid(stn[STN_ID])

        fout = open(os.path.join(path_out, "".join([outId, ".raw.", varname])), 'w')

        tair = data[:, x]

        for yr, i in zip(yrs, np.arange(0, yrs.size * 12, 12)):

            outLine = " ".join([outId, str(yr)])
            tairYr = tair[i:i + 12]

            if np.ma.isMA(tairYr):
                ttairYr = tairYr.data
                validMask = np.logical_not(tairYr.mask)
                ttairYr[validMask] = ttairYr[validMask] * 100.0
                ttairYr[tairYr.mask] = -9999
                tairYr = ttairYr
            else:
                tairYr = tairYr * 100.0

            for aVal in tairYr:
                outLine = "".join([outLine, " {0:>5.0f}".format(aVal), "   "])
            outLine = "".join([outLine, "\n"])
            fout.write(outLine)

        fout.close()

def _parse_pha_adj(path_adj_log):
    '''
    Parse a log file of PHA adjustments and return as a structured array
    with dtype DTYPE_PHA_ADJ
    '''
    
    f = open(path_adj_log)
        
    vals_adj = []
    
    for aline in f.readlines():
        
        stnid = aline[10:21]
        yrmth_start = aline[25:31]
        yrmth_end = aline[45:51]
        val_adj = np.float(aline[75:81])
        
        date_start = datetime(np.int(yrmth_start[0:4]),np.int(yrmth_start[-2:]),1)
        ymd_start = np.int("%d%02d%02d"%(date_start.year,date_start.month,date_start.day))
        
        date_end = datetime(np.int(yrmth_end[0:4]),np.int(yrmth_end[-2:]),1)
        ymd_end = np.int("%d%02d%02d"%(date_end.year,date_end.month,date_end.day))
         
        #if val_adj != 0.0:

        vals_adj.append((stnid,ymd_start,ymd_end,val_adj))
    
    vals_adj = np.array(vals_adj,dtype=DTYPE_PHA_ADJ)
    
    return vals_adj

def load_snotel_sensor_hist(station_ids=None):
    '''Load SNOTEL metadata history for YSI extended range sensor installs.
    
    Metadata extracted from NRCS station sensor history pages:
    e.g.: http://wcc.sc.egov.usda.gov/nwcc/sensorhistory?sitenum=542
    
    Parameters
    ----------
    station_ids : list-like, optional
        Station IDs for which sensor history should be loaded.

    Returns
    -------
    stnhist : pandas.DataFrame
        DataFrame with station_id as index and change_date column specifying
        the date that the YSI extended range sensor was installed
    '''

    path_root = os.path.dirname(__file__)
    fpath_stnhist = os.path.join(path_root, 'data', 'snotel_sensor_installs.csv')
    
    stnhist = pd.read_csv(fpath_stnhist, index_col='station_id')
    stnhist.index = "NRCS_"+stnhist.index
    stnhist['date_new_sensor'] = pd.to_datetime(stnhist['date_new_sensor'])
    
    if station_ids is not None:
        stnhist = stnhist.loc[station_ids].dropna()
    
    stnhist = stnhist.reset_index().rename(columns={'index':'station_id',
                                                    'date_new_sensor':'change_date'})
    return stnhist
    
def get_pha_adj_df(fpath_pha_adj_log, stns, elem):
    '''Build DataFrame of PHA adjustments.
    
    Parameters
    ----------
    fpath_pha_adj_log : str
        File path to PHA adjustment log file generated by a run_pha.
        e.g.: [pha_run_path]/run/data/benchmark/world1/output/pha_adj_tmin.log
    stns : structured ndarray
        Stations for which PHA was run. Structured station array must contain at
        least the following fields: STN_ID, STN_NAME, LAT, LON and can be obtained from
        twx.db.StationDataDb
    elem : str
        Element for which PHA was run (e.g.: tmin, tmax)

    Returns
    -------
    stnhist : pandas.DataFrame
        DataFrame with following columns:
        YEAR_MONTH_START
        YEAR_MONTH_END
        ADJ(C)
        VARIABLE
        STN_ID
        NAME
        LON
        LAT
        ELEV(m)
    '''
        
    stnids = np.array([_format_stnid(stnid) for stnid in stns[STN_ID]])
    
    stn_meta = {}
    for i in np.arange(stnids.size):
        stn_meta[stnids[i]] = stns[i]
    
    pha_adj = _parse_pha_adj(fpath_pha_adj_log)
    pha_adj = pha_adj[np.in1d(pha_adj[STN_ID], stnids, assume_unique=False)]
    pha_adj = pha_adj[np.abs(pha_adj['adj']) > 0]
    pha_adj['adj'] = -pha_adj['adj']
    
    a_month = relativedelta(months=1)
    dates_start_chgpt = [datetime.strptime(str(a_ymd),"%Y%m%d") + a_month
                         for a_ymd in pha_adj['ymd_start']]
    dates_end_chgpt = [datetime.strptime(str(a_ymd),"%Y%m%d") + a_month
                       for a_ymd in pha_adj['ymd_end']]
    
    mthyr_start_chgpt = ['%d%02d'%(a_date.year,a_date.month)
                         for a_date in dates_start_chgpt]
    mthyr_end_chgpt = ['%d%02d'%(a_date.year,a_date.month)
                       for a_date in dates_end_chgpt]
    
    stnelev_chgpt = np.round([stn_meta[a_stnid][ELEV]
                              for a_stnid in pha_adj[STN_ID]]).astype(np.int)
    stnlon_chgpt = [stn_meta[a_stnid][LON] for a_stnid in pha_adj[STN_ID]]
    stnlat_chgpt = [stn_meta[a_stnid][LAT] for a_stnid in pha_adj[STN_ID]]
    stnname_chgpt = [stn_meta[a_stnid][STN_NAME] for a_stnid in pha_adj[STN_ID]]
    stnid_chgpt = [stn_meta[a_stnid][STN_ID] for a_stnid in pha_adj[STN_ID]]
    varnames = [elem]*pha_adj.size
    
    df = pd.DataFrame({'YEAR_MONTH_START':mthyr_start_chgpt,
                       'YEAR_MONTH_END':mthyr_end_chgpt,
                       'ADJ(C)':pha_adj['adj'], 'VARIABLE':varnames,
                       'STN_ID':stnid_chgpt, 'NAME':stnname_chgpt,
                       'LON':stnlon_chgpt, 'LAT':stnlat_chgpt,
                       'ELEV(m)':stnelev_chgpt})
    
    df = df[['YEAR_MONTH_START', 'YEAR_MONTH_END', 'ADJ(C)', 'VARIABLE',
             'STN_ID','NAME','LON','LAT','ELEV(m)']]
    df = df.set_index('STN_ID')
    return df     
