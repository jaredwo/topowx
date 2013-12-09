'''
Created on Oct 10, 2011
Utilities for quality assurance of station location metadata (lon, lat, and elev). In most cases, elev
is correct, but lon/lat are either imprecise or incorrect.
@author: jared.oyler
'''

import urllib, urllib2
from xml.dom import minidom
from twx.db.station_data import STN_ID, STATE, LON, LAT, ELEV,station_data_ncdb
import numpy as np
import time
import twx.infill.obs_por as obs_por
from netCDF4 import Dataset
from twx.utils.input_raster import input_raster

# DEM service URLs
URL_USGS_NED = 'http://gisdata.usgs.gov/XMLWebServices/TNM_Elevation_Service.asmx/getElevation'
URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'
USRNAME_GEONAMES = "jaredntsg"

class loc_fixer():
    '''
    loc_fixer
    '''
    
    def __init__(self,dem_rast,n,nstep,nmax,dif_thres):
        
        self.dem = dem_rast.readEntireRaster()
        self.rdem = dem_rast
        self.n = n
        self.nstep = nstep
        self.nmax = nmax
        self.dif_thres = dif_thres
        
        
    def find_pt(self,lon,lat,elev):
        
        x,y = self.rdem.getGridCellOffset(lon, lat)
        
        min_dif = self.dif_thres + 1
        cn = self.n
        
        while min_dif > self.dif_thres and cn <= self.nmax:
            print cn
            grid = np.mgrid[y-cn:y+cn+1,x-cn:x+cn+1]
            ys = grid[0,:,:].ravel()
            xs = grid[1,:,:].ravel()
        
            difs = np.abs(self.dem[ys,xs] - elev)
            min_dif = np.min(difs)
            mask = difs == min_dif
            
            lats,lons = self.rdem.getLatLon(xs[mask],ys[mask],transform=False)
            cn+=self.nstep
        
        return lats,lons,min_dif
        
        
        
    


def load_locqa(path_locqa):
    '''
    Load an entire location qa file into memory as a dictionary
    A location qa file has the following headers: STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
    
    @param path_locqa: the path to the location qa file
    @return: locs a dictionary of file lines where stn ids are keys
    '''
    
    afile = open(path_locqa)
    afile.readline()
    
    locs = {}
    for line in afile.readlines():
        
        #STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        stn_id = line.split(",")[0].strip()
        locs[stn_id] = line
        
    return locs

def load_locs_fixed_all(path_locqa):
    '''
    Load an entire location qa file into memory as a dictionary, but where values are either the fixed 
    lon,lat, and elev of a station or its original lon, lat, and elev.
    A location qa file has the following headers: STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
    
    @param path_locqa: the path to the location qa file
    @return: locs a dictionary of lon/lat/elev values where stn ids are keys
    '''
    
    afile = open(path_locqa)
    afile.readline()
    
    locs = {}
    for line in afile.readlines():
        
        #STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        vals = line.split(",")
        vals = [x.strip() for x in vals]
        
        #If fixed/new values are not empty, use these instead of originals
        if vals[7] != '' and vals[8] != '':
        
            locs[vals[0]] = [float(vals[7]),float(vals[8]),float(vals[9])]

        else:
        
            locs[vals[0]] = [float(vals[2]),float(vals[3]),float(vals[4])]
    
    return locs

def load_locs_fixed(path_locqa):
    '''
    Load only fixed station locations in a location qa file into memory as a dictionary
    A location qa file has the following headers: STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
    
    @param path_locqa: the path to the location qa file
    @return: locs a dictionary of lon/lat/elev values where stn ids are keys
    '''
    
    afile = open(path_locqa)
    afile.readline()
    
    locs = {}
    for line in afile.readlines():
        
        #STN_ID,ST,LON,LAT,ELEV,DEM,DIF,LON_NEW,LAT_NEW,ELEV_NEW
        vals = line.split(",")
        
        #If fixed/new values are not empty, load station
        if vals[7] != '' and vals[8] != '':
        
            locs[vals[0]] = [float(vals[7]),float(vals[8]),float(vals[9])]
    
    return locs

def combine_locqa(old_locqa,new_locqa,path_fout):
    '''
    Combines to location qa files into a single file. If the 2 files have overlapping stations,
    the new file takes precedence over the old file
    
    @param old_locqa: a path to a location qa csv file
    @param new_locqa: a path to a location qa csv file
    @param path_fout: the path to the new combined location qa csv file
    '''

    locs_old = load_locqa(old_locqa)
    locs_new = load_locqa(new_locqa)
    
    stnids_old = np.array(locs_old.keys())
    stnids_new = np.array(locs_new.keys())
    
    stnids_old = stnids_old[np.logical_not(np.in1d(stnids_old, stnids_new, assume_unique=True))]
    
    fout = open(path_fout,"w")
    fout.write(",".join(["STN_ID", "ST", "LON", "LAT", "ELEV", "DEM", "DIF","LON_NEW","LAT_NEW","ELEV_NEW","\n"]))
    
    for stn_id in stnids_old:
        
        fout.write(locs_old[stn_id])
    
    for stn_id in stnids_new:
        
        fout.write(locs_new[stn_id])
        
    fout.close()
    

def get_elev_geonames(lon, lat, url=URL_GEONAMES_SRTM):
    '''
    Get elev value from geonames web sevice (SRTM or ASTER).
    @param lon:
    @param lat:
    '''
    
    while 1:
        #?lat=50.01&lng=10.2&username=demo
        # url GET args
        values = {'lat' : lat,
        'lng' : lon,
        'username' : USRNAME_GEONAMES}
       
        # encode the GET arguments
        data = urllib.urlencode(values)
       
        # make the URL into a qualified GET statement    
        get_url = "".join([url, "?", data])
       
        req = urllib2.Request(url=get_url)
        response = urllib2.urlopen(req)
        elev = float(response.read().strip())
        
        if elev == -32768.0 and url == URL_GEONAMES_SRTM:
            #Try ASTER instead
            url = URL_GEONAMES_ASTER
        else:
            break
            
    #print "".join(["Geonames Elev: ",str(elev)])
    return elev

def get_elev(stn):
    '''
    Get the elevation of a station's lon/lat from a DEM online datasource
    @param stn: a station record (ID,NAME,STATE,LON,LAT,ELEV)
    @return elev_dem: elev in meters
    '''
    
    #determine if a us station or not. currently, only stations outside us are in mexico or canada
    us_stn = not (stn[STN_ID].find('CA') != -1 or stn[STN_ID].find('MX') != -1 or (stn[STATE] == '' or stn[STATE] == 'AK' or stn[STATE] == 'HI'))
        
    while 1:
        
        try:
        
            if us_stn:
                elev_dem = get_elev_usgs(stn[LON], stn[LAT])
            else:
                elev_dem = get_elev_geonames(stn[LON], stn[LAT])
                
            return elev_dem
        
        except urllib2.URLError:
            print "Error in connection. sleep and try again..."
            time.sleep(5)
    

def get_elev_usgs(lon, lat):
    '''
    Get elev value from USGS NED 1/3 arc-sec DEM.  Code directly modeled from:
    http://casoilresource.lawr.ucdavis.edu/drupal/node/610
    @param lon:
    @param lat:
    '''
   
    #   NED 1/3rd arc-second: Eastern United States    NED.CONUS_NED_13E    -99.0006,24.9994,
    #-65.9994,49.0006
    #NED 1/3rd arc-second: Western United States    NED.CONUS_NED_13W    -125.0006,25.9994,
    #-98.9994,49.0006
   
    if lon <= -98.9994:
        src_layer = 'NED.CONUS_NED_13W'
    else:
        src_layer = 'NED.CONUS_NED_13E'
   
   
    # url GET args
    values = {'X_Value' : lon,
    'Y_Value' : lat,
    'Elevation_Units' : 'meters',
    'Source_Layer' : src_layer,
    'Elevation_Only' : '1', }
   
    # make some fake headers, with a user-agent that will
    # not be rejected by bone-headed servers
    user_agent = 'Mozilla/4.0 (compatible; MSIE 5.5; Windows NT)'
    headers = {'User-Agent' : user_agent}
   
    # encode the GET arguments
    data = urllib.urlencode(values)
   
    # make the URL into a qualified GET statement:
    get_url = URL_USGS_NED + '?' + data
   
    # make the request: note that by ommitting the url arguments
    # we force a GET request, instead of a POST
    req = urllib2.Request(url=get_url, headers=headers)
    response = urllib2.urlopen(req)
    the_page = response.read()
   
    try:
    
        # convert the HTML back into plain XML
        for entity, char in (('lt', '<'), ('gt', '>'), ('amp', '&')):
            the_page = the_page.replace('&%s;' % entity, char)
    
        # clean some cruft... XML won't parse with this stuff in there...
        the_page = the_page.replace('<string xmlns="http://gisdata.usgs.gov/XMLWebServices/">', '')
        the_page = the_page.replace('<?xml version="1.0" encoding="utf-8"?>\r\n', '')
        the_page = the_page.replace('</string>', '')
        the_page = the_page.replace('<!-- Elevation Values of -1.79769313486231E+308 (Negative Exponential Value) may mean the data source does not have values at that point.  --> <USGS_Elevation_Web_Service_Query>', '')
       
        # parse the cleaned XML
        dom = minidom.parseString(the_page)
        children = dom.getElementsByTagName('Elevation_Query')[0]
       
        # extract the interesting parts
        elev = float(children.getElementsByTagName('Elevation')[0].firstChild.data)
        data_source = children.getElementsByTagName('Data_Source')[0].firstChild.data
       
        # print to stdout
        #print "%f,%f,%f,%s" % (lon, lat, elev, data_source)
        
        return elev
    
    except:
        
        print "".join(["ERROR: ", str(lon), ",", str(lat)])
        return np.nan

def update_stn_locs(path_db,path_locqa):
    '''
    Updates the lon,lat,elev of stations in a ncdf database based on corrected locations in a
    qa station location csv file
    @param path_db: the path to the ncdf database
    @param path_locqa: The path to the station location qa csv file
    '''
    
    locs_fixed = load_locs_fixed(path_locqa)
    
    ds = Dataset(path_db,'r+')
    var_lon = ds.variables['lon']
    var_lat = ds.variables['lat']
    var_elev = ds.variables['elev']
    stn_ids = ds.variables['stn_id'][:]
    
    cnt = 0
    print "Total # of station locs to fix: "+str(len(locs_fixed.keys()))
    for stn_id in locs_fixed.keys():
        
        try:
            x = np.nonzero(stn_ids==stn_id)[0][0]
        except IndexError:
            print "Could not update, since station not in database: "+stn_id
            continue
        
        lon,lat,elev = locs_fixed[stn_id]
        var_lon[x] = lon
        var_lat[x] = lat
        var_elev[x] = elev
        cnt+=1
        ds.sync()
    print "Total # of stations locs updated: "+str(cnt)

def qa_stn_locs(path_db,path_locqa,path_out,path_obspor):
    '''
    Performs quality assurance of station locations by retrieving the DEM elevation for a station's lon/lat
    from a DEM online datasource and then comparing the DEM elevation with that listed for the station. The user
    must then decide if there is a location issue that warrants manually relocating the station. Only stations
    in the db that have location values different/missing from the current location qa file and have long enough
    period-of-records as specified by infill.obs_por will be checked and outputed.
    
    @param path_db: the path to the ncdf database
    @param path_locqa: the path to the most current location qa csv file
    @param path_out: the output path for a new location qa csv file
    @param path_obspor: the path to a obs period-of-record file generated with infill.obs_por
    
    @return: Outputs a station location csv file at the path specified by path_out.
    The headers of the file are: ["STN_ID", "ST", "LON", "LAT", "ELEV", "DEM", "DIF","LON_NEW","LAT_NEW","ELEV_NEW","\n"]
    '''
    
    stn_da = station_data_ncdb(path_db)
    stns = stn_da.stns
    stn_da = None
    
    locs_fixed = load_locs_fixed_all(path_locqa)
    
    por = obs_por.load_por_csv(path_obspor)
    mask_por_tmin,mask_por_tmax,mask_por_prcp = obs_por.build_valid_por_masks(por)

    stns = stns[np.logical_or(np.logical_or(mask_por_tmin,mask_por_tmax),mask_por_prcp)]
    
    print "Total num of stations for location qa: "+str(stns.size)
    
    fout = open(path_out,"w")
    fout.write(",".join(["STN_ID", "ST", "LON", "LAT", "ELEV", "DEM", "DIF","LON_NEW","LAT_NEW","ELEV_NEW","\n"]))
    
    for stn in stns:
        
        chk_stn = False
        
        if locs_fixed.has_key(stn[STN_ID]):
            
            lon,lat,elev = locs_fixed[stn[STN_ID]]
            if np.round(stn[LON],4) != np.round(lon,4) or np.round(stn[LAT],4) != np.round(lat,4) or np.round(stn[ELEV],0) != np.round(elev,0):
                chk_stn = True

        else:
            chk_stn = True
            
        if chk_stn:
            
            elev_dem = get_elev(stn)
            dif = stn[ELEV] - elev_dem
            print dif,stn
            fout.write(",".join([stn[STN_ID], stn[STATE], str(stn[LON]), str(stn[LAT]), str(stn[ELEV]), str(elev_dem), str(dif),"","","","\n"]))
    
    fout.close()
            
if __name__ == '__main__':
    
    #STEP 1: Update station locations of stations that had their locations previously fixed
#    update_stn_locs("/projects/daymet2/station_data/all/all_1948_2012.nc", 
#                    "/projects/daymet2/station_data/all/qa_elev_final.csv")
    
    #STEP 2: Run new location qa check on stations
#    qa_stn_locs("/projects/daymet2/station_data/all/all_1948_2012.nc",
#                "/projects/daymet2/station_data/all/qa_elev_final.csv",
#                "/projects/daymet2/station_data/all/qa_elev_20130328.csv",
#                path_obspor='/projects/daymet2/station_data/all/all_por_1948_2012.csv') 

    #STEP 3: Combine new and old location qa csv files
#    combine_locqa("/projects/daymet2/station_data/all/qa_elev_final.csv", 
#                  "/projects/daymet2/station_data/all/qa_elev_20130328.csv",
#                  "/projects/daymet2/station_data/all/qa_elev_combine.csv")

    #STEP 4: Update station locations with any new fixed locations
    update_stn_locs("/projects/daymet2/station_data/all/all_1948_2012.nc", 
                    "/projects/daymet2/station_data/all/qa_elev_final.csv")
