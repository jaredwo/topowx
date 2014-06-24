'''
Utilities for running the external Pairwise Homogenization Algorithm (PHA) software:

Menne, M.J., and C.N. Williams, Jr., 2009: Homogenization of temperature series 
via pairwise comparisons. J. Climate, 22, 1700-1717.

'''
__all__ = ['setup_pha', 'run_pha']

import numpy as np
import subprocess
import os
import glob
from twx.db import LON, LAT, STN_ID

INCL_LINE_BEGIN_YR = '        parameter (begyr = 1895)\n'
INCL_LINE_END_YR = '        parameter (endyr = 2015)\n'
INCL_LINE_N_STNS = '        parameter (maxstns = 7720)\n'
CONF_LINE_END_YR = 'endyr=1999\n'
CONF_LINE_MAX_YRS = 'maxyrs=500000\n'
CONF_LINE_ELEMS = 'elems="tavg"\n'


def setup_pha(fpath_pha_tar, path_out_src, path_out_run, yr_begin, yr_end, stns, tair, varname):
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
    '''

    n_stns = stns.size
    n_stnyrs = (yr_end - yr_begin + 1) * n_stns
    yrs = np.arange(yr_begin, yr_end + 1)

    print "Uncompressing PHA..."
    subprocess.call(["tar", "-xzvf", fpath_pha_tar, '-C', path_out_src])

    fpath_incl = os.path.join(path_out_src, 'phav52i', 'source_expand', 'parm_includes', 'inhomog.parm.MTHLY.TEST.incl')
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
    _write_input_station_data(path_out_run, varname, stns, tair, yrs)

def run_pha(path_run, varname):
    '''
    Run a PHA instance setup by setup_pha

    Parameters
    ----------
    path_run : str
        The PHA run path from setup_pha
    varname : str
        Temperature variable name (tmin or tmax)
    '''

    pha_cmd = os.path.join(path_run, 'testv52i-pha.sh') + "  world1 %s raw 0 0 P" % (varname,)
    print "Running PHA for %s..." % (varname,)
    subprocess.call(pha_cmd, shell=True)

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

def _write_input_station_data(path_pha_run, varname, stns, tair, yrs):
    '''
    Write station data to GHCN format for input to PHA
    '''

    _write_stn_list(stns, os.path.join(path_pha_run, 'data', 'benchmark', 'world1', 'meta', 'world1_stnlist.%s' % (varname,)))

    os.remove(os.path.join(path_pha_run, 'data', 'benchmark', 'world1', 'meta', 'world1_stnlist.tavg'))
    fpath_metafile = os.path.join(path_pha_run, 'data', 'benchmark', 'world1', 'meta', 'world1_metadata_file.txt')
    f = open(fpath_metafile, 'w')
    f.close()

    path_stn_obs = os.path.join(path_pha_run, 'data', 'benchmark', 'world1', 'monthly', 'raw')
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

    if stnid.startswith("GHCN_"):

        outid = stnid.split("_")[1]

    elif stnid.startswith("SNOTEL_"):

        outid = stnid.split("_")[1]
        outid = "".join(["SNT", "{0:0>8}".format(outid)])

    elif stnid.startswith("RAWS_"):

        outid = stnid.split("_")[1]
        outid = "".join(["RAW", "{0:0>8}".format(outid)])

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
