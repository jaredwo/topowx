'''
Class for selecting stations around an interpolation/predicting point and weighting them according to the SYMAP
interpolation algorithm

References:

Frei, C., and C. Schar. 1998. A precipitation climatology of the Alps from high-resolution rain-gauge observations. 
International Journal of Climatology 18:873-900

Shepard, D. 1968. A two-dimensional interpolation function for irregularly-spaced data. 
Pages 517-524 Proceedings of the 1968 23rd ACM national conference.

Willmott, C. J., C. M. Rowe, and W. D. Philpot. 1985. 
Small-scale climate maps: A sensitivity analysis of some common assumptions associated with 
grid-point interpolation and contouring. Cartography and Geographic Information Science 12:5-16.

@author: jared.oyler
'''
from twx.db.station_data import LON,LAT,STN_ID, NEON
import numpy as np
import twx.utils.util_geo as utlg

#http://en.wikipedia.org/wiki/Contiguous_United_States
AREA_CONUS = 8080464.3 #km-2

AREA_NEON_0 = 435754.46 #northeast
AREA_NEON_2 = 351629.25 #mid atlantic
AREA_NEON_3 = 429730.85 #southeast + atlantic neotropcial
AREA_NEON_5 = 545054.48 #great lakes
AREA_NEON_6 = 651362.51 #prairie peninsula
AREA_NEON_7 = 305979.32 #appalachians / cumberland plateau
AREA_NEON_8 = 682243.42 #ozarks complex
AREA_NEON_9 = 866811.05 #northern plains
AREA_NEON_10 = 453509.73 #central plains
AREA_NEON_11 = 536254.24 #southern plains
AREA_NEON_12 = 326671.41 #northern rockies
AREA_NEON_13 = 643100.22 #southern rockies / colorado plateau
AREA_NEON_14 = 437129.90 #desert southwest
AREA_NEON_15 = 842958.90 #great basin
AREA_NEON_16 = 192976.95 #pacific northwest
AREA_NEON_17 = 240936.39 #pacific southwest

NEON_AREAS = {0:AREA_NEON_0,2:AREA_NEON_2,3:AREA_NEON_3,5:AREA_NEON_5,6:AREA_NEON_6,7:AREA_NEON_7,
              8:AREA_NEON_8,9:AREA_NEON_9,10:AREA_NEON_10,11:AREA_NEON_11,12:AREA_NEON_12,13:AREA_NEON_13,
              14:AREA_NEON_14,15:AREA_NEON_15,16:AREA_NEON_16,17:AREA_NEON_17}
        
   
class station_select(object):
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
        
        self.lat = None
        self.lon = None
        self.mask_rm = None
        self.stn_dists = None
        self.stn_dists_sort = None
        self.bw_nstns = None
    
    def set_pt(self,lat,lon,stns_rm=None):
        
        stn_dists = utlg.grt_circle_dist(lon,lat,self.stns[LON],self.stns[LAT])
        
        fnl_stns_rm = stns_rm if stns_rm is not None else np.array([])
                
        if self.rm_zero_dist_stns:
            #Remove any stations that are at the same location (dist == 0)
            fnl_stns_rm = np.unique(np.concatenate((fnl_stns_rm,self.stns[STN_ID][stn_dists==0])))
        
        if fnl_stns_rm.size > 0: 
            mask_rm = np.logical_not(np.in1d(self.stns[STN_ID], fnl_stns_rm, assume_unique=True))
        else:
            mask_rm = self.mask_all
        
        self.lat = lat
        self.lon = lon
        self.mask_rm = mask_rm
        self.stn_dists = stn_dists
        self.dist_sort = np.argsort(self.stn_dists)
        self.sort_stn_dists = np.take(self.stn_dists, self.dist_sort)

        self.sort_stns = np.take(self.stns, self.dist_sort)
        mask_rm = np.take(self.mask_rm, self.dist_sort)
                
        mask_rm = np.nonzero(mask_rm)[0]
        self.sort_stn_dists = np.take(self.sort_stn_dists, mask_rm)
        self.sort_stns = np.take(self.sort_stns, mask_rm)
                
    def set_params(self,bw_nstns):
        
        self.bw_nstns = bw_nstns
        
    def set_ngh_stns(self,load_obs=True, obs_mth=None):
                        
        stn_dists = self.sort_stn_dists
        stns = self.sort_stns
        
        #get the distance bandwidth using the the bw_nstns + 1
        dbw = stn_dists[self.bw_nstns]
        ngh_stns = stns[0:self.bw_nstns]
        dists = stn_dists[0:self.bw_nstns]
        
        #bisquare weighting, do not normalize for gwr?
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
        
class station_select_neon(station_select):
    
    def __init__(self,stns,stn_da,fpath_neon_params,max_nngh_delta=.2):
        station_select.__init__(self, stns, stn_da, 30, 30)
        
        neon_params = np.loadtxt(fpath_neon_params,skiprows=1,delimiter=",")
        
        a_neon = np.arange(0,np.max(neon_params[:,0])+1)
        self.params = np.zeros((a_neon.size,4),dtype=np.int)
        mask = np.in1d(a_neon, neon_params[:,0], True)
        
        self.params[mask,0] = neon_params[:,0] #neon number
        self.params[mask,1] = neon_params[:,1] #min nngh
        self.params[mask,2] = self.params[mask,1]+np.round(self.params[mask,1]*max_nngh_delta) #max nngh
        self.params[mask,3] = neon_params[:,2] #p for geo-wgts

    def get_interp_stns(self,lat,lon,neon,stns_rm=None,load_obs=True):
        
        neon = np.int(neon)
        self.set_init_radius(self.params[neon,1], self.params[neon,2])
        interp_stns,interp_obs,dists,wgt = station_select.get_interp_stns(self,lat, lon, stns_rm, load_obs)
        
        fin_mask = np.isfinite(interp_stns[NEON])
        
        p_stns = interp_stns[fin_mask]
        p_wgt = wgt[fin_mask]/np.sum(wgt[fin_mask])
        p_neon = p_stns[NEON].astype(np.int)
        
        min_stns = np.round(np.average(self.params[p_neon,1],weights=p_wgt))
        max_stns = np.round(np.average(self.params[p_neon,2],weights=p_wgt))
        p_scaler = np.average(self.params[p_neon,3],weights=p_wgt)
        
        self.set_init_radius(min_stns, max_stns)
        interp_stns,interp_obs,dists,wgt = station_select.get_interp_stns(self,lat, lon, stns_rm, load_obs)
        
        return interp_stns,interp_obs,dists,wgt,p_scaler
        
        
def get_wgts(stns,rad_buf=1):
    
    wgts = np.zeros((stns.size,stns.size))
    
    for x in np.arange(stns.size):
        
        dists = utlg.dist_ca(stns[LON][x], stns[LAT][x], stns[LON],stns[LAT])[0]
        radius = np.max(dists) + rad_buf
    
        wgt = (1.0+np.cos(np.pi*(dists/radius)))/2.0
        wgt = wgt/np.sum(wgt)
        
        wgts[x,:] = wgt
    
    return wgts
    
    
    
