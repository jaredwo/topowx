'''
Utility functions for automated downloading RAWS data from the WRCC website
http://www.raws.dri.edu/

@author: jared.oyler
'''
import urllib
import urllib2
import sys
import numpy as np
from datetime import datetime
from calendar import monthrange
import os

class Unbuffered:
    def __init__(self, stream):
        self.stream = stream
    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
    def __getattr__(self, attr):
        return getattr(self.stream, attr)
sys.stdout=Unbuffered(sys.stdout)

STN_TIME_SERIES_URL = "http://www.raws.dri.edu/cgi-bin/wea_dysimts.pl?"
STN_METADATA_URL = "http://www.raws.dri.edu/cgi-bin/wea_info.pl?"
STN_TIME_SERIES2_URL = 'http://www.raws.dri.edu/cgi-bin/wea_dysimts2.pl'

def dms2decimal(degrees, minutes, seconds):
    """ Converts degrees, minutes, and seconds to the equivalent
    number of decimal degrees. If parameter 'degrees' is negative,
    then returned decimal-degrees will also be negative.
    
    Example:
    >>> dms2decimal(121, 8, 6)
    121.13500000000001
    >>> dms2decimal(-121, 8, 6)
    -121.13500000000001
    
    """
    
    decimal = 0.0
    if (degrees >= 0):
        decimal = degrees + float(minutes)/60 + float(seconds)/3600
    else:
        decimal = degrees - float(minutes)/60 - float(seconds)/3600
        
    return decimal

def build_stnid_list(path_stnlst_pages,out_path):
    
    afile = open(path_stnlst_pages)
    stn_ids = []
    
    for line in afile.readlines():
        
        req = urllib2.Request(line.strip())
        response = urllib2.urlopen(req)
        plines = response.readlines()
        for pline in plines:
            
            if "rawMAIN.pl" in pline:
                
                stn_id = pline.split("?")[1][0:6]
                print stn_id
                stn_ids.append(stn_id)
    
    stn_ids = np.unique(stn_ids)
    print "Total # of stn_ids: "+str(stn_ids.size)
    
    fo = open(out_path,"w")
    for stn_id in stn_ids:
        fo.write("".join([stn_id,"\n"]))
   
def subset_ghcn_raws(raws_stnid_fpath,ghcn_stn_fpath,out_path):
    
    raws_ids_orig = np.loadtxt(raws_stnid_fpath,dtype="<S6")
    raws_ids = np.array([x[2:] for x in raws_ids_orig],dtype="<S4")
    
    ghcn_stns = open(ghcn_stn_fpath)
    
    ghcn_raws = []
    for aline in ghcn_stns.readlines():
        
        stn_id = aline[0:11].strip()
        
        #prefix for a raws station
        if stn_id[0:3] == "USR":
            ghcn_raws.append(stn_id[-4:])
            
    ghcn_raws = np.array(ghcn_raws,dtype="<S4")
    
    fnl_ids = raws_ids_orig[np.in1d(raws_ids, ghcn_raws, True)]
    np.savetxt(out_path, fnl_ids, "%s") 

def build_stn_metadata(path_stnids,path_out):
    
    afile = open(path_stnids)
    fo = open(path_out,"w")
    
    for stn_id in afile.readlines():
        
        try:
            stn_id = stn_id.strip()
            
            response = urllib2.urlopen("".join([STN_TIME_SERIES_URL,stn_id]))
            plines = response.readlines()
            
            start_date = None
            end_date = None
            stn_name = None
            read_stn_name = False
            
            for pline in plines:
                
                if start_date is None or end_date is None or stn_name is None:
                    
                    if read_stn_name:
                        stn_name = pline.split(">")[1][0:-4]
                        read_stn_name = False
                    
                    if "Station:" in pline:
                        read_stn_name = True
                    
                    if "Earliest available data" in pline:
                        start_date = datetime.strptime(pline.split(":")[1][1:-2],"%B %Y")
                        
                    if "Latest available data" in pline:
                        end_date = datetime.strptime(pline.split(":")[1][1:-2],"%B %Y")
                        end_date = end_date.replace(day=monthrange(end_date.year,end_date.month)[1]) #set to last day of month
            
            response = urllib2.urlopen("".join([STN_METADATA_URL,stn_id]))
            plines = response.readlines()
            lon,lat,elev = [None]*3
            
            for x in np.arange(len(plines)):
                
                pline = plines[x]
                
                if "Latitude" in pline:
                    lat = parse_decdegrees(plines[x+2])
    
                if "Longitude" in pline:
                    lon = parse_decdegrees(plines[x+2])
                
                if "Elevation" in pline:
                    elev = float(plines[x+2].split()[0])*0.3048 #convert from feet to meters
            
            stn_ln = " ".join([stn_id[2:],start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"),
                               "%.5f"%lat,"%.5f"%lon,"%.2f"%elev,stn_name,"\n"])
            
            print stn_ln 
            fo.write(stn_ln)
        except:
            print "COULD NOT LOAD METADATA FOR: ",stn_id
    fo.close()

def save_dly_series(stn_id, start_date, end_date, out_path):
    
    values = {'smon' : start_date.strftime("%m"),
              'sday' : start_date.strftime("%d"),
              'syea' : start_date.strftime("%y"),
              'emon' : end_date.strftime("%m"),
              'eday' : end_date.strftime("%d"),
              'eyea' : end_date.strftime("%y"),
              'qBasic' : "ON",
              'unit' : 'M',
              'Ofor' : 'A',
              'Datareq' : 'A',
              'qc' : 'Y',
              'miss' : '08',#-9999
              "Submit Info" : "Submit Info",
              'stn' : stn_id,
              'WsMon' : '01',
              'WsDay' : '01',
              'WeMon' : '12',
              'WeDay' : '31'}

    data = urllib.urlencode(values)
    req = urllib2.Request(STN_TIME_SERIES2_URL, data)
    response = urllib2.urlopen(req)
    
    fo = open("".join([out_path,stn_id,".txt"]),"w")
    
    for pline in response.readlines():
        fo.write(pline)
    fo.close()

def save_all_dly_series(path_meta,out_path):

    fmeta = open(path_meta)
    
    for line in fmeta.readlines():

        stn_id = line[0:4]
        start_date = datetime.strptime(line[5:15],"%Y-%m-%d")
        end_date = datetime.strptime(line[16:26],"%Y-%m-%d")
        
        if not os.path.exists("".join([out_path,stn_id,".txt"])):
        
            print stn_id,start_date,end_date
            save_dly_series(stn_id,start_date,end_date,out_path)

def parse_decdegrees(a_str):
    
    vals = a_str.split("&")
    
    deg = float(vals[0])
    
    vals = vals[1].split()
    
    minute = float(vals[1][0:-1])
    sec = float(vals[2])
    
    return dms2decimal(deg,minute,sec)

if __name__ == '__main__':

    #Build a list of raws stn ids    
#    build_stnid_list('/projects/daymet2/station_data/raws/raws_stnlst_pages.txt',
#                     '/projects/daymet2/station_data/raws/raws_stnids.txt')
   
    #Build all metadata for stations
    #build_stn_metadata('/projects/daymet2/station_data/raws/raws_ghcn_stnids.txt','/projects/daymet2/station_data/raws/raws_meta.txt')
   
    #Build daily time series for each station
    save_all_dly_series('/projects/daymet2/station_data/raws/raws_meta.txt','/projects/daymet2/station_data/raws/raws_data/')
