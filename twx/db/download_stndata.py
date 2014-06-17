'''
Utility functions to download weather station data.  Current main weather
station datasources are:

1.) Global Historical Climate Network Daily (GHCN-Daily)

2.) SNOTEL Current Tab Card Files (tab delimited files)

3.) SNOTEL Historical. No download function written as this does not appear to
    be updated regularly. Downloaded manually from
    ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/snothist/.

4.) Remote Automated Weather Stations (RAWS) at the WRCC website.
'''
import os
import tarfile
import subprocess
from urlparse import urljoin

RPATH_SNOTEL_TABDATA = 'ftp://ftp.wcc.nrcs.usda.gov/data/snow/snotel/cards/'
RPATH_GHCN = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/'
RPATH_GHCN_BYYEAR = 'http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/by_year/'


def snotel_mirror_tabdata(local_path, remote_path=RPATH_SNOTEL_TABDATA):
    '''
    Use wget to mirror SNOTEL current tab card files. This takes a while
    since there are many individual files.

    Parameters
    ----------
    local_path : str
        The local mirror path to mirror to.
    remote_path : str, optional
        The remote path to mirror from.
    '''

    subprocess.call(['/usr/local/bin/wget', '--no-verbose', '--mirror',
                     '--directory-prefix=' + local_path, remote_path])


def ghcnd_download_data(local_path, remote_path=RPATH_GHCN, extract_tar=True):
    '''
    Use wget to download and unzip the latest GHCN-Daily data

    Parameters
    ----------
    local_path : str
        The local path to download to.
    remote_path : str, optional
        The remote path to download from.
    extract_tar : bool, optional
        Extract the ghcnd_all.tar.gz file
    '''

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-version.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'status.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'readme.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-inventory.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd-stations.txt')])

    subprocess.call(['wget', '--directory-prefix=' + local_path,
                     urljoin(remote_path, 'ghcnd_all.tar.gz')])

    if extract_tar:
        print "Extracting GHCN-Daily tar.gz file..."
        ghcn_tar = tarfile.open(os.path.join(local_path, 'ghcnd_all.tar.gz'))
        ghcn_tar.extractall(local_path)


def ghcnd_download_byyr_data(local_path, yrs, remote_path=RPATH_GHCN_BYYEAR):
    '''
    Use wget to download the latest GHCN-Daily data in "by year" format

    Parameters
    ----------
    local_path : str
        The local path to download to.
    yrs: sequence of int
        A list of years for which to download
    remote_path : str, optional
        The remote path to download from.
    '''

    for yr in yrs:

        subprocess.call(['wget', '--directory-prefix=' + local_path,
                         urljoin(remote_path, "%s.csv.gz" % (yr, ))])
