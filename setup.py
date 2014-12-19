from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name='topowx',
    
    version='1.0',
    
    packages=find_packages(),
    
    description='TopoWx ("Topography Weather") Project',
    
    long_description='TopoWx is a project for producing high resolution\
    gridded climate datasets. Corresponding publication: http://dx.doi.org/10.1002/joc.4127',
    
    url='http://www.ntsg.umt.edu/project/TopoWx',
    
    author='Jared W. Oyler',
    
    author_email='jared.oyler@ntsg.umt.edu',

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
    }
)
