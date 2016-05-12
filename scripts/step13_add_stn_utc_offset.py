'''
Script to add UTC offset attribute to each homogenized station location
'''

import os
from twx.db import add_utc_offset
from twx.utils import TwxConfig

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    add_utc_offset(twx_cfg.fpath_stndata_nc_tair_homog,
                   geonames_usrname=twx_cfg.username_geonames)
