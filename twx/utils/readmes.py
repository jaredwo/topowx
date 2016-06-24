'''
Functions for creating TopoWx data readmes.
'''

import os
import subprocess

def create_main_readme(fpath_out, start_yr, end_yr, twx_version, dataset_version):
    '''Create the main TopoWx data readme
    
    Parameters
    ----------
    fpath_out : str
        File path for the output readme
    start_yr : int
        Start year for TopoWx dataset 
    end_yr : int
        Start year for TopoWx dataset
    twx_version : str
        TopoWx code version
    dataset_version : str
        TopoWx dataset version
    '''
    
    path_root = os.path.dirname(__file__)

    with open(os.path.join(path_root, 'data', 'readme_main.txt')) as afile:
        
        readme_txt = afile.read()
        readme_txt = readme_txt%(start_yr, end_yr, twx_version, dataset_version)
    
    with open(fpath_out, 'w') as afile:
        
        afile.write(readme_txt)
        
def create_stnobs_readme(fpath_out, start_yr, end_yr, fpath_obs_tmin, fpath_obs_tmax):
    '''Create the station observation data readme
    
    Parameters
    ----------
    fpath_out : str
        File path for the output readme
    start_yr : int
        Start year for TopoWx dataset 
    end_yr : int
        Start year for TopoWx dataset
    fpath_obs_tmin : str
        File path to auxiliary netcdf file of tmin observations
    fpath_obs_tmax : str
        File path to auxiliary netcdf file of tmax observations
    '''
    
    path_root = os.path.dirname(__file__)

    with open(os.path.join(path_root, 'data', 'readme_stnobs.txt')) as afile:
        
        readme_txt = afile.read()
        readme_txt = readme_txt%(start_yr, end_yr)
    
    header_line = ''.join(['#']*80)
    for a_fpath in [fpath_obs_tmin, fpath_obs_tmax]:
        
        fname = os.path.split(a_fpath)[-1]
        
        nctxt = subprocess.check_output('ncdump '+'-h '+a_fpath, shell=True)
        nctxt = '\n'.join([header_line, fname, header_line, nctxt,''])
        
        readme_txt = ''.join([readme_txt, nctxt])
    
    with open(fpath_out, 'w') as afile:
        
        afile.write(readme_txt)
    
def create_auxgrid_readme(fpath_out, fpaths_nc_grids):
    '''Create the auxiliary grid data readme
    
    Parameters
    ----------
    fpath_out : str
        File path for the output readme
    fpaths_nc_grids : list of str
        File paths for all auxiliary grids
    '''
    
    path_root = os.path.dirname(__file__)

    with open(os.path.join(path_root, 'data', 'readme_auxgrids.txt')) as afile:
        
        readme_txt = afile.read()
    
    header_line = ''.join(['#']*80)
    for a_fpath in fpaths_nc_grids:
        
        fname = os.path.split(a_fpath)[-1]
        
        nctxt = subprocess.check_output('ncdump '+'-h '+a_fpath, shell=True)
        nctxt = '\n'.join([header_line, fname, header_line, nctxt,''])
        
        readme_txt = ''.join([readme_txt, nctxt])
    
    with open(fpath_out, 'w') as afile:
        
        afile.write(readme_txt)