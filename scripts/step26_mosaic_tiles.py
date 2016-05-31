'''
Script to mosaic TopoWx tiles into single CONUS-wide netCDF files
'''
from twx.interp import TileMosaic
from twx.utils import TwxConfig, mkdir_p
import fnmatch
import os

if __name__ == '__main__':
    
    twx_cfg = TwxConfig(os.getenv('TOPOWX_INI'))
    
    fpath_mask = os.path.join(twx_cfg.path_predictor_rasters, 'mask.nc')
    tile_names = fnmatch.filter(os.listdir(twx_cfg.path_tile_out),
                                "h[0-9][0-9]v[0-9][0-9]")
    elems = ['tmin', 'tmax']
        
    mtwx = TileMosaic(fpath_mask, tile_size_y=250, tile_size_x=250,
                      chk_size_y=50, chk_size_x=50)
    
    # Create normals mosaics
    print "Creating normals mosaics..."
    for a_elem in elems:
         
        fpath_mosaic_out = os.path.join(twx_cfg.path_mosaic_norms,
                                        'normals_%s.nc'%a_elem) 
          
        print "Mosaicing %d %s tiles to: %s" % (len(tile_names), a_elem, fpath_mosaic_out)
          
        mtwx.create_normals_mosaic(tile_names, a_elem, twx_cfg.path_tile_out,
                                   fpath_mosaic_out, twx_cfg.twx_data_version)
    
    # Create daily mosaics
    print "Creating daily mosaics..."
    for a_elem in elems:
        
        path_mosaic_out = os.path.join(twx_cfg.path_mosaic_daily, a_elem)
        mkdir_p(path_mosaic_out)
                 
        mtwx.create_dly_ann_mosaics(tile_names, a_elem, twx_cfg.path_tile_out,
                                    path_mosaic_out, twx_cfg.interp_start_date.year,
                                    twx_cfg.interp_end_date.year, twx_cfg.twx_data_version,
                                    chunk_cache_size = 250000000.0) #250 MB