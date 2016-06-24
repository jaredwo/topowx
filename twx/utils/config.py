from ConfigParser import ConfigParser
from twx.utils import ymdL, mkdir_p
import numpy as np
import os
import pandas as pd

class TwxConfig():
    '''Class to load and access TopoWx configuration settings in a INI file.
    
    Upon initialization, also creates necessary sub-directories in the TopoWx
    data root directory if they do not exist.
    
    Example TopoWx INI File:
    
    [TOPOWX_CONFIG]
    # Path to TopoWx data root
    TWX_DATA_ROOT=[a path]
    # Lon/lat bounding box for station observations
    STN_BBOX=-126.0,22.0,-64.0,53.0
    # Start date for which to process station observations
    OBS_START_DATE=1895-01-01
    # End data for which to process station observations
    OBS_END_DATE=2016-03-29
    # Start date for interpolation
    INTERP_START_DATE=1948-01-01
    # End date for interpolation
    INTERP_END_DATE=2015-12-31
    # Station observation elements to process
    OBS_ELEMS=tmin,tmax,prcp,tobs_tmin,tobs_tmax,tobs_prcp
    # Primary station observation elements
    OBS_MAIN_ELEMS=tmin,tmax,prcp
    # Station chunk size for which to load and process station observations
    STN_READ_CHUNK_GHCND=100
    STN_READ_CHUNK_SNOTEL=20
    STN_READ_CHUNK_RAWS=20
    # Station chunk size for loading and writing to netcdf file
    STN_WRITE_CHUNK_NC=100
    # Station chunk size for creating aggregated data (e.g.--monthly from daily)
    STN_AGG_CHUNK=1000
    # A geonames username for accessing DEM elevation services
    USERNAME_GEONAMES=[a username]
    '''
    
    def __init__(self, fpath_ini):
    
        cfg = ConfigParser()
        cfg.read(fpath_ini)
        
        self.twx_data_root = cfg.get('TOPOWX_CONFIG', 'twx_data_root')
        self.obs_start_date = pd.Timestamp(cfg.get('TOPOWX_CONFIG',
                                                   'obs_start_date'))
        self.obs_end_date = pd.Timestamp(cfg.get('TOPOWX_CONFIG',
                                                 'obs_end_date'))
        self.interp_start_date = pd.Timestamp(cfg.get('TOPOWX_CONFIG',
                                                      'interp_start_date'))
        self.interp_end_date = pd.Timestamp(cfg.get('TOPOWX_CONFIG',
                                                    'interp_end_date'))
        
        bbox_str = cfg.get('TOPOWX_CONFIG', 'stn_bbox')
        self.stn_bbox = tuple([np.float(i) for i in bbox_str.split(',')])
        
        self.obs_elems = tuple(cfg.get('TOPOWX_CONFIG', 'obs_elems').split(','))
        self.obs_main_elems = tuple(cfg.get('TOPOWX_CONFIG',
                                            'obs_main_elems').split(','))
        self.stn_read_chunk_ghcnd = int(cfg.get('TOPOWX_CONFIG',
                                                'stn_read_chunk_ghcnd'))
        self.stn_read_chunk_snotel = int(cfg.get('TOPOWX_CONFIG',
                                                 'stn_read_chunk_snotel'))
        self.stn_read_chunk_raws = int(cfg.get('TOPOWX_CONFIG',
                                               'stn_read_chunk_raws'))
        self.stn_write_chunk_nc = int(cfg.get('TOPOWX_CONFIG',
                                              'stn_write_chunk_nc'))
        self.stn_agg_chunk = int(cfg.get('TOPOWX_CONFIG',
                                         'stn_agg_chunk'))
        self.username_geonames = cfg.get('TOPOWX_CONFIG',
                                         'username_geonames')
        self.fpath_log_daily_infill = cfg.get('TOPOWX_CONFIG',
                                              'fpath_log_daily_infill')
        self.twx_data_version = cfg.get('TOPOWX_CONFIG',
                                        'twx_data_version')
        
        # Make TopoWx data directory for local storage of station observations
        self.path_stndata = os.path.join(self.twx_data_root, 'station_data')
        mkdir_p(self.path_stndata)
        
        fname_stndata_hdf_ghcnd = 'obs_ghcnd_%d_%d.hdf' % (ymdL(self.obs_start_date),
                                                         ymdL(self.obs_end_date))
        self.fpath_stndata_hdf_ghcnd = os.path.join(self.path_stndata,
                                                    fname_stndata_hdf_ghcnd)
        
        fname_stndata_hdf_snotel = 'obs_snotel_%d_%d.hdf' % (ymdL(self.obs_start_date),
                                                           ymdL(self.obs_end_date))
        self.fpath_stndata_hdf_snotel = os.path.join(self.path_stndata,
                                                     fname_stndata_hdf_snotel)
        
        fname_stndata_hdf_raws = 'obs_raws_%d_%d.hdf' % (ymdL(self.obs_start_date),
                                                       ymdL(self.obs_end_date))
        self.fpath_stndata_hdf_raws = os.path.join(self.path_stndata,
                                                   fname_stndata_hdf_raws)
        
        fname_stndata_nc_all = 'obs_all_%d_%d.nc' % (ymdL(self.obs_start_date),
                                                   ymdL(self.obs_end_date))
        self.fpath_stndata_nc_all = os.path.join(self.path_stndata,
                                                 fname_stndata_nc_all)
        
        fname_stndata_nc_tair_tobs_adj = 'tair_tobs_adj_%d_%d.nc' % (ymdL(self.obs_start_date),
                                                                   ymdL(self.obs_end_date))
        self.fpath_stndata_nc_tair_tobs_adj = os.path.join(self.path_stndata,
                                                           fname_stndata_nc_tair_tobs_adj)
        
        fname_stndata_nc_tair_homog = 'tair_homog_%d_%d.nc' % (ymdL(self.obs_start_date),
                                                             ymdL(self.obs_end_date))
        self.fpath_stndata_nc_tair_homog = os.path.join(self.path_stndata,
                                                        fname_stndata_nc_tair_homog)
        
        self.fpath_locqa_hdf = os.path.join(self.path_stndata, 'locqa.hdf')
        self.fpath_locqa_fail_csv = os.path.join(self.path_stndata, 'locqa_fail.csv')
        
        # Make TopoWx data directory for PHA-based homogenization
        self.path_homog_pha = os.path.join(self.path_stndata, 'homog')
        mkdir_p(self.path_homog_pha)
        self.fpath_pha_tgz = os.path.join(self.path_homog_pha, 'phav52i.tar.gz')
        
        # Make TopoWx data directories for reanalysis data
        self.path_reanalysis_data = os.path.join(self.twx_data_root,
                                                 'reanalysis_data')
        mkdir_p(self.path_reanalysis_data)
        self.path_reanalysis_namerica = os.path.join(self.path_reanalysis_data,
                                                     'n_america_subset')
        mkdir_p(self.path_reanalysis_namerica)
        
        # Make TopoWx data directory for infilled station observations
        self.path_stndata_infill = os.path.join(self.path_stndata, 'infill')
        mkdir_p(self.path_stndata_infill)
        self.fpath_xval_infill_nc = os.path.join(self.path_stndata_infill,
                                                 'xval_infill_tair.nc')
        self.fpath_stndata_nc_infill_tmin = os.path.join(self.path_stndata_infill,
                                                         'infill_tmin.nc')
        self.fpath_stndata_nc_infill_tmax = os.path.join(self.path_stndata_infill,
                                                         'infill_tmax.nc')
        self.fpath_flagged_bad_stns = os.path.join(self.path_stndata_infill,
                                                   'bad_stns.csv')
        self.fpath_stndata_nc_serial_tmin = os.path.join(self.path_stndata_infill,
                                                         'serial_tmin.nc')
        self.fpath_stndata_nc_serial_tmax = os.path.join(self.path_stndata_infill,
                                                         'serial_tmax.nc')
                
        # Make data directories for storing interp param optimization files
        # Temperature normals
        self.path_interp_optim_norms = os.path.join(self.path_stndata_infill,
                                                    'optim_norm')
        mkdir_p(self.path_interp_optim_norms)
        # Daily anomalies
        self.path_interp_optim_anoms = os.path.join(self.path_stndata_infill,
                                                    'optim_anom')
        mkdir_p(self.path_interp_optim_anoms)
        self.fpath_xval_interp_nc_tmin = os.path.join(self.path_stndata_infill,
                                                      'xval_interp_tmin.nc')
        self.fpath_xval_interp_nc_tmax = os.path.join(self.path_stndata_infill,
                                                      'xval_interp_tmax.nc')
        
        # Make TopoWx data directory for raster data
        self.path_rasters = os.path.join(self.twx_data_root, 'rasters')
        mkdir_p(self.path_rasters)
        self.path_predictor_rasters = os.path.join(self.path_rasters,
                                                   'conus_interp_grids', 'ncdf')
        mkdir_p(self.path_predictor_rasters)
        
        # Make TopoWx data directory for writing output tiles
        self.path_tile_out = os.path.join(self.twx_data_root, 'tile_output')
        mkdir_p(self.path_tile_out)
        
        
        # Make TopoWx log directory
        self.path_logs = os.path.join(self.twx_data_root, 'logs')
        mkdir_p(self.path_logs)
        
        ##################################
        # Make TopoWx data directory for final outputs
        ##################################
        
        self.path_final_output = os.path.join(self.twx_data_root, 'final_output_data')
        mkdir_p(self.path_final_output)
        
        # Final auxiliary data directories
        self.path_aux_data = os.path.join(self.path_final_output, 'auxiliary_data')
        mkdir_p(self.path_aux_data)
        self.path_aux_stndata = os.path.join(self.path_aux_data, 'station_data')
        mkdir_p(self.path_aux_stndata)
        self.fpath_stndata_nc_aux_tmin = os.path.join(self.path_aux_stndata,
                                                         'stn_obs_tmin.nc')
        self.fpath_stndata_nc_aux_tmax = os.path.join(self.path_aux_stndata,
                                                         'stn_obs_tmax.nc')
        self.fpath_pha_adj_aux = os.path.join(self.path_aux_stndata, 'homog_adjust.csv')
        self.path_aux_grids = os.path.join(self.path_aux_data, 'auxiliary_grids')
        mkdir_p(self.path_aux_grids)
        
        # Final TopoWx output mosaics for normals, daily, and monthly data
        self.path_mosaic_norms = os.path.join(self.path_final_output, 'normals')
        mkdir_p(self.path_mosaic_norms)
        self.path_mosaic_daily = os.path.join(self.path_final_output, 'daily')
        mkdir_p(self.path_mosaic_daily)
        self.path_mosaic_monthly = os.path.join(self.path_final_output, 'monthly')
        mkdir_p(self.path_mosaic_monthly)
