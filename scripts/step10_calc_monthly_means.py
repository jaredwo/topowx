from twx.db import add_monthly_means
from twx.utils import TwxConfig
import os

if __name__ == '__main__':

    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    add_monthly_means(twx_cfg.fpath_stndata_nc_tair_tobs_adj, 'tmin')
    add_monthly_means(twx_cfg.fpath_stndata_nc_tair_tobs_adj, 'tmax')