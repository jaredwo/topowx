'''
Class for selecting and weighting stations and loading corresponding observation data around an interpolation point.

@author: jared.oyler
'''
from twx.db.station_data import LON,LAT,STN_ID
import numpy as np
import twx.utils.util_geo as utlg

   
class StationSelect(object):
    '''
    classdocs
    '''

    def __init__(self,stn_da,stn_mask=None,rm_zero_dist_stns=False):
        '''
        Constructor
        '''
        
        if stn_mask is None:
            self.stns = stn_da.stns
        else:
            self.stns = stn_da.stns[stn_mask]
            
        self.stn_da = stn_da
        self.rm_zero_dist_stns = rm_zero_dist_stns
        self.mask_all = np.ones(self.stns.size,dtype=np.bool)
        
        #Cached data for a specific point
        self.pt_lat = None
        self.pt_lon = None
        self.pt_stns_rm = None
        self.pt_mask_stns_rm = None
        self.pt_stn_dists = None
        self.pt_dist_sort = None
        self.pt_sort_stn_dists = None
        self.pt_sort_stns = None
            
    def __set_pt(self,lat,lon,stns_rm=None):
        
        if isinstance(stns_rm, str) or isinstance(stns_rm, unicode):
            stns_rm = np.array([stns_rm])
        elif not isinstance(stns_rm, np.ndarray) and not stns_rm is None:
            raise Exception("stns_rm must  be str, unicode, or numpy array of str/unicode")
        
        do_set_pt = True
        
        if self.pt_lat == lat and self.pt_lon == lon:
            
            try:
                if self.pt_stns_rm == None and stns_rm == None:
                    do_set_pt = False
                elif np.alltrue(self.pt_stns_rm==stns_rm):
                    do_set_pt = False
            except:
                pass
                
        if do_set_pt:        
            
            stn_dists = utlg.grt_circle_dist(lon,lat,self.stns[LON],self.stns[LAT])
            
            fnl_stns_rm = stns_rm if stns_rm is not None else np.array([])
                    
            if self.rm_zero_dist_stns:
                #Remove any stations that are at the same location (dist == 0)
                fnl_stns_rm = np.unique(np.concatenate((fnl_stns_rm,self.stns[STN_ID][stn_dists==0])))
            
            if fnl_stns_rm.size > 0: 
                mask_rm = np.logical_not(np.in1d(self.stns[STN_ID], fnl_stns_rm, assume_unique=True))
            else:
                mask_rm = self.mask_all
            
            self.pt_lat = lat
            self.pt_lon = lon
            self.pt_stns_rm = stns_rm
            self.pt_mask_stns_rm = mask_rm
            self.pt_stn_dists = stn_dists
            self.pt_dist_sort = np.argsort(self.pt_stn_dists)
            self.pt_sort_stn_dists = np.take(self.pt_stn_dists, self.pt_dist_sort)
    
            self.pt_sort_stns = np.take(self.stns, self.pt_dist_sort)
            mask_rm = np.take(self.pt_mask_stns_rm, self.pt_dist_sort)
                    
            mask_rm = np.nonzero(mask_rm)[0]
            self.pt_sort_stn_dists = np.take(self.pt_sort_stn_dists, mask_rm)
            self.pt_sort_stns = np.take(self.pt_sort_stns, mask_rm)
                        
    def set_ngh_stns(self, lat, lon, nnghs, load_obs=True, obs_mth=None,stns_rm=None):
        
        self.__set_pt(lat, lon, stns_rm)
        
        stn_dists = self.pt_sort_stn_dists
        stns = self.pt_sort_stns
        
        #get the distance bandwidth using the the nnghs + 1
        dbw = stn_dists[nnghs]
        ngh_stns = stns[0:nnghs]
        dists = stn_dists[0:nnghs]
        
        #bisquare weighting
        wgt = np.square(1.0-np.square(dists/dbw))
        
        #Gaussian
        #wgt = np.exp(-.5*((dists/dbw)**2))
        
        #wgt = ((1.0+np.cos(np.pi*(dists/dbw)))/2.0)
        #wgt = 1.0/(dists**2)
        #wgt = wgt/np.sum(wgt)
                                    
        #Sort by stn id
        stnid_sort = np.argsort(ngh_stns[STN_ID])
        interp_stns = np.take(ngh_stns,stnid_sort)
        wgt = np.take(wgt,stnid_sort)
        dists = np.take(dists,stnid_sort)
        
        if load_obs:
            ngh_obs = self.stn_da.load_obs(ngh_stns[STN_ID],mth=obs_mth)
        else:
            ngh_obs = None
            
        self.ngh_stns = interp_stns
        self.ngh_obs = ngh_obs
        self.ngh_dists = dists
        self.ngh_wgt = wgt
        