'''
Created on Nov 17, 2012

@author: jaredwo
'''
'''
    h11v05
    h10v05
    h09v06
    h13v04*
    h09v04-
    h07v06
    h10v06
    h09v05
    h07v05*
    h11v06*
    h08v04
    h12v05
    h10v04-
    h11v04
    h08v05
    h12v04
    h08v06
    '''
TILES_CONUS = ['h11v05','h10v05','h09v06','h07v06','h10v06','h09v05','h08v04','h12v05','h11v04',
               'h08v05','h12v04']#h09v04,h10v04
from ftplib import FTP
import numpy as np
import os

FULL_FTP_PATH = 'ftp://e4ftl01.cr.usgs.gov/MOLA/MYD11A2.005/'
FULL_HTTP_PATH = 'http://e4ftl01.cr.usgs.gov/MOLA/MYD11A2.005/'
FULL_MCD12Q1_FTP_PATH = 'ftp://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.051/2011.01.01/'

def download_tiles(tiles,out_path):
    
    ddirs = get_date_dirs()
    
    for tile in tiles:
        
        tile_path = "".join([out_path,"MYD11A2.005.",tile,"/"])
        if not os.path.exists(tile_path):
            os.mkdir(tile_path)
            
        os.chdir(tile_path)
        
        for adir in ddirs:
            
            cmd = "".join(["wget -nv ",FULL_FTP_PATH,adir,"/","MYD11A2*",tile,"*hdf"])
            os.system(cmd)

def download_mcd12q1_tiles(tiles,out_path):
    
    os.chdir(out_path)
    
    for tile in tiles:
        cmd = "".join(["wget -nv ",FULL_MCD12Q1_FTP_PATH,"MCD12Q1*",tile,"*hdf"])
        os.system(cmd)
    

def download_tile_datedirs(tiles,out_path,datedirs):
    
    for tile in tiles:
        
        os.chdir("".join([out_path,'MYD11A2.005.',tile]))
        
        for adir in datedirs:
            cmd = "".join(['wget -r -l1 --no-parent -A "MYD11A2*',tile,'*hdf" -nd ',FULL_HTTP_PATH,adir])
            print cmd
            os.system(cmd)

def get_date_dirs():
    
    not_connected = True
    nconns = 0
    while not_connected:
        try:
            ftp = FTP('e4ftl01.cr.usgs.gov')
            not_connected = False
        except EOFError, e:
            if nconns < 20:
                nconns+=1
                print "Error connecting. Trying again"
    
    ftp.login()
    ftp.cwd("MOLA/MYD11A2.005")
    dirs = []
    ftp.dir(dirs.append)
    dirs = np.array(dirs)
    dirs = dirs[np.char.startswith(dirs,"d")] 
    dirs = [x.split()[-1] for x in dirs]
    ftp.close()
    
    return dirs
    

def download_tile(tile,out_path):
    os.chdir(out_path)
    not_connected = True
    nconns = 0
    while not_connected:
        try:
            ftp = FTP('e4ftl01.cr.usgs.gov')
            not_connected = False
        except EOFError, e:
            if nconns < 20:
                nconns+=1
                print "Error connecting. Trying again"
    
    ftp.login()
    ftp.cwd("MOLA/MYD11A2.005")
    dirs = []
    ftp.dir(dirs.append)
    dirs = np.array(dirs)
    dirs = dirs[np.char.startswith(dirs,"d")] 
    dirs = [x.split()[-1] for x in dirs]
    ftp.close()
    
    for adir in dirs:
        
        cmd = "".join(["wget -nv ",FULL_FTP_PATH,adir,"/","MYD11A2*",tile,"*hdf"])
        os.system(cmd)
#        ftp.cwd(adir)
#        files = np.array(ftp.nlst())
#        tile_files = files[np.logical_and(np.char.find(files, tile) != -1,np.char.endswith(files,"hdf"))]
#        print tile_files
#        ftp.cwd('..')

if __name__ == '__main__':
        
    TILES = ['h07v05',
            'h07v06',
            'h08v04',
            'h08v05',
            'h08v06',
            'h09v03',
            'h09v04',
            'h09v05',
            'h09v06',
            'h10v03',
            'h10v04',
            'h10v05',
            'h10v06',
            'h11v03',
            'h11v04',
            'h11v05',
            'h11v06',
            'h12v03',
            'h12v04',
            'h12v05',
            'h13v03',
            'h13v04',
            'h14v03',
            'h14v04']
    
    #DATE_DIRS = ['2012.11.08','2012.11.16','2012.11.24','2012.12.02','2012.12.10','2012.12.18','2012.12.26']
    
    #download_tile_datedirs(TILES,'/projects/daymet2/climate_office/modis/MYD11A2/',DATE_DIRS)
    
    download_mcd12q1_tiles(TILES, '/projects/daymet2/climate_office/modis/MCD12Q1.051/')
    #download_tiles(TILES, '/projects/daymet2/climate_office/modis/MYD11A2/')
    
    #download_tiles(['h11v03','h10v03'], '/projects/daymet2/climate_office/modis/')
    #download_tiles(['h13v04','h07v05','h11v06'], '/projects/daymet2/climate_office/modis/')
    #download_tiles(TILES_CONUS, '/Users/jaredwo/Downloads/wxtopo_data/pymodis_tst/')
    #download_tile("h09v04","/projects/daymet2/climate_office/modis/MYD11A2.005.h09v04")