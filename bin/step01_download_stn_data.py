'''
Created on Jun 16, 2014

@author: jaredwo
'''
import twx
import numpy as np
import os

if __name__ == '__main__':

    PROJECT_ROOT = "/projects/topowx"

    # Download GHCN-D data from NCDC
    twx.db.ghcnd_download_data(os.path.join(PROJECT_ROOT, 'station_data',
                                           'ghcn'))

    # Download GHCN-D data from NCDC in "by year" format for 1948-2013
    twx.db.ghcnd_download_byyr_data(os.path.join(PROJECT_ROOT, 'station_data',
                                                'ghcn', 'ghcn_yrly'),
                                   yrs=np.arange(1948, 2014))

    # Mirror SNOTEL tab-delimited data from NRCS FTP
    twx.db.snotel_mirror_tabdata(os.path.join(PROJECT_ROOT, 'station_data',
                                              'snotel', 'current'))
