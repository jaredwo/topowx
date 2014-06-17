'''
Created on Jun 16, 2014

@author: jaredwo
'''
import twx

'''
if __name__ == '__main__':
    #mirror_snotel_tabdata('/projects/daymet2/station_data/snotel/current/')
    #download_ghcn_data('/projects/daymet2/station_data/ghcn/',backup=True)
    download_ghcn_byyr_data('/projects/daymet2/station_data/ghcn/ghcnByYr/', np.arange(1948,2013))
'''

if __name__ == '__main__':
    twx.db.mirror_snotel_tabdata('/Users/jaredwo/Documents/twx/station_data/snotel')