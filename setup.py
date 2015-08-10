from setuptools import setup, find_packages
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))
#Required external R libraries
r_libs_require = ['changepoint',
                  'gstat',
                  'norm',
                  'pcaMethods',
                  'sp']

#Check python version
if sys.version_info.major > 2:
    print "Sorry, Python 3 is not yet supported"
    sys.exit(1)

#Check for rpy2 and required R libraries
import rpy2.robjects
r = rpy2.robjects.r

for a_rlib in r_libs_require:
    r('library(%s)'%a_rlib)

setup(
    name='topowx',
    
    version='1.1.0',
    
    packages=find_packages(),
    
    description='TopoWx ("Topography Weather") Project',
    
    long_description='TopoWx is a project for producing high resolution\
    gridded climate datasets. Corresponding publication: http://dx.doi.org/10.1002/joc.4127',
    
    url='http://www.ntsg.umt.edu/project/TopoWx',
    
    author='Jared W. Oyler',
    
    author_email='jaredwo@gmail.com',

    license='GPL',

    classifiers=[
        'Development Status :: 5 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Scientific',
        'License :: OSI Approved :: GNU General Public License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
    ],

    keywords='climate interpolation',
    
    package_data={
        'twx.db': ['data/*.txt'],
        'twx.infill': ['data/*.txt','rpy/*.R'],
        'twx.interp': ['rpy/*.R'],    
    },
    
    install_requires=['fiona',
                      'gdal',
                      'matplotlib',
                      'mpi4py',
                      'netCDF4',
                      'numpy',
                      'pandas',
                      'pyproj',
                      'rpy2',
                      'scipy',
                      'shapely',
                      'statsmodels',
                      'tzwhere',
                      'xray'
    ] 
)