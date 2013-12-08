'''
Utility functions to download weather station data.  Current main weather station datasources are:

1.) Global Historical Climate Network Daily (GHCN-Daily)
2.) SNOTEL Current Tab Card Files (tab delimited files)
3.) SNOTEL Historical. No download function written as this does not appear to be updated regularly. Downloaded manually
from ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/snothist/.
4.) Glacier National Park Alpine Network. Provided on occasion by USGS NOROCK
5.) Remote Automated Weather Stations (RAWS). A web crawler (db.raws) was written to download RAWS data from WRCC website
(http://www.raws.dri.edu/). It is not yet integrated into this download module.

@author: jared.oyler
'''
import os
import tarfile
import numpy as np

RPATH_SNOTEL_TABDATA = 'ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/cards/'
RPATH_GHCN = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/'
RPATH_GHCN_BYYEAR = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'

def mirror_snotel_tabdata(local_path):
    '''
    Uses wget to mirror SNOTEL current tab card files to the path provided. This could take a while since there
    are many individual files.
    
    @param local_path: The local mirror path 
    '''
    
    wget_cmd = 'wget --no-verbose --mirror --directory-prefix=' + local_path + ' ' + RPATH_SNOTEL_TABDATA
    os.system(wget_cmd)
    
def download_ghcn_data(local_path,backup=True):
    '''
    Downloads and unzips the latest GHCN data
    
    @param local_path: the local bath to which the data should be downloaded
    @param backup: if true, take all current files in local path and mv to bak directory. 
    If false, all current files are deleted
    '''
    
    os.chdir(local_path)
    
    if backup:
        os.mkdir("".join([local_path,"bak"]))
        os.system("mv * "+"bak")
    else:
        os.system("".join(["rm -rf ",local_path,"*"]))
    
    os.system("".join(["wget ",RPATH_GHCN,"ghcnd-version.txt"]))
    os.system("".join(["wget ",RPATH_GHCN,"status.txt"]))
    os.system("".join(["wget ",RPATH_GHCN,"readme.txt"]))
    os.system("".join(["wget ",RPATH_GHCN,"ghcnd-inventory.txt"]))
    os.system("".join(["wget ",RPATH_GHCN,"ghcnd-stations.txt"]))
    os.system("".join(["wget ",RPATH_GHCN,"ghcnd_all.tar.gz"]))
    
    print "Extracting GHCN tar file..."
    ghcn_tar = tarfile.open('ghcnd_all.tar.gz')
    ghcn_tar.extractall()
    
def download_ghcn_byyr_data(local_path,yrs):
    os.chdir(local_path)
    for yr in yrs:
        os.system("".join(["wget ",RPATH_GHCN_BYYEAR,str(yr),".csv.gz"]))
    

if __name__ == '__main__':
    #mirror_snotel_tabdata('/projects/daymet2/station_data/snotel/current/')
    #download_ghcn_data('/projects/daymet2/station_data/ghcn/',backup=True)
    download_ghcn_byyr_data('/projects/daymet2/station_data/ghcn/ghcnByYr/', np.arange(1948,2013))