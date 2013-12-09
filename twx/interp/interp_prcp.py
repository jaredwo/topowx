'''
Classes and functions for performing prcp interpolation
'''
import numpy as np
from twx.db.station_data import LON,LAT,ELEV,YMD,PRCP,station_data_infill,STN_ID
import matplotlib.mlab as mlab
from twx.interp.clibs import clib_wxTopo
from twx.interp.station_select import station_select
import scipy.stats as stats

#rpy2
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.numpy2ri import numpy2ri
robjects.conversion.py2ri = numpy2ri
r = robjects.r

PO_THRESHS = np.arange(0.0001, 0.9999, .0001)
MIN_PRCP_AMT = 0.01 #min amt in cm

def quick_interp_prcp(lon,lat,elev,ngh=38,sigma=0.02507643,df=17585.,
                      path_prcp='/projects/daymet2/station_data/infill/old_infill/infill_prcp.nc'):
    
    stn_da = station_data_infill(path_prcp, 'prcp')
    
    mod = modeler_clib_prcp()
    prcp_interper = interp_prcp(mod)
    ainput = prcp_input()
    aoutput = prcp_output()
    
    stn_slct = station_select(stn_da.stns,ngh,ngh+10)
    stns,wgts,rad = stn_slct.get_interp_stns(lat, lon)
    stn_obs = stn_da.load_obs(stns[STN_ID])
    
    ainput.stns = stns
    ainput.stn_wgts = wgts
    ainput.stn_obs = stn_obs
    ainput.set_pt(lon, lat, elev)
    ainput.sigma = sigma
    ainput.df = df
    
    prcp_interper.model_prcp(ainput,aoutput)
    
    return ainput,aoutput

class prcp_input(object):
    
    def __init__(self):
        
        self.stns = None
        self.stn_wgts = None
        self.stn_obs = None
        self.sigma = None
        self.df = None
        
        self.a_pt = np.empty(1, dtype=[(LON, np.float64), (LAT, np.float64), (ELEV, np.float64)])
        
    def set_pt(self,lon,lat,elev):
        
        self.a_pt[LON] = lon
        self.a_pt[LAT] = lat
        self.a_pt[ELEV] = elev
        
class prcp_output(object):
    
    def __init__(self):
        
        self.prcp = None
        self.mean = None
        self.se = None
        self.ci = None
        
    def to_csv(self,days,fname):
        
        a_out = np.recarray(days.size,dtype=[(YMD,np.int),(PRCP,np.float)])
        a_out[YMD] = days[YMD]
        a_out[PRCP] = self.prcp
        mlab.rec2csv(a_out, fname, formatd={PRCP:mlab.FormatFloat(4)})
        
    def to_str(self,prefix=""):
        
        return "|".join([prefix,str(self.mean),str(self.se),str(self.ci),str(((self.ci[1] - self.ci[0])/self.mean)*100.)])

class modeler(object):
    
    def __init__(self):
        pass
    
    def model(self,stns,stn_wgts,stn_obs,a_pt):
        pass
    
class modeler_R_po(modeler):
    
    def __init__(self,r_src='/home/jared.oyler/ecl_helios_workspace/daymet2/topomet.R'):
        
        modeler.__init__(self)
        r.source(r_src)
        
    def model(self,stns,stn_wgts,stn_obs,a_pt):
        
        r_stns = robjects.DataFrame({'LON':stns[LON],'LAT':stns[LAT],'ELEV':stns[ELEV],'WGT':stn_wgts})
        r_obs = robjects.Matrix(stn_obs)
        
        interp_rslt = r.interp_po(r_stns,r_obs,a_pt[LON],a_pt[LAT],a_pt[ELEV])

        interp_vals = np.array(interp_rslt)
        
        return interp_vals

class modeler_clib_po(modeler):
    
    def __init__(self,clib_lib='/Users/jaredwo/Documents/workspace/wxTopo_C/Release/libwxTopo_C'):
        
        modeler.__init__(self)
        self.aclib = clib_wxTopo(clib_lib) 

    def model(self,stns,stn_wgts,stn_obs,a_pt):
        
        interp_vals,error = self.aclib.prcomp_interp(stns[LON], stns[LAT], stns[ELEV], stn_wgts, stn_obs, a_pt[LON], a_pt[LAT], a_pt[ELEV])
        
        if error:
            raise SVD_Exception("SVD did not converge.")
               
        return interp_vals

class modeler_clib_prcp(modeler):
    
    def __init__(self,clib_lib='/home/jared.oyler/ecl_helios_workspace/wxTopo/Release/libwxTopo',thres=0.5):
        
        modeler.__init__(self)
        self.aclib = clib_wxTopo(clib_lib)
        self.thres = thres 

    def model(self,stns,stn_wgts,stn_obs,a_pt,sigma=None,df=None):
        
        stn_obs_po = np.copy(stn_obs)
        stn_obs_po[stn_obs_po > 0] = 1
        
        po,error = self.aclib.prcomp_interp(stns[LON], stns[LAT], stns[ELEV], stn_wgts, stn_obs_po, a_pt[LON], a_pt[LAT], a_pt[ELEV])
        
        if error:
            raise SVD_Exception("SVD did not converge.")
        
        po_mask = po >= self.thres
        
        stn_obs_prcp = np.array(stn_obs[po_mask,:],dtype=np.float64)
        
        stn_obs_prcp_mean = np.mean(stn_obs_prcp,axis=0)
        stn_obs_prcp_std = np.std(stn_obs_prcp,axis=0,ddof=1)
        
        pt_mean = self.aclib.regress(stns[LON], stns[LAT], stns[ELEV], stn_wgts, stn_obs_prcp_mean, a_pt[LON], a_pt[LAT], a_pt[ELEV])
        pt_std = self.aclib.regress(stns[LON], stns[LAT], stns[ELEV], stn_wgts, stn_obs_prcp_std, a_pt[LON], a_pt[LAT], a_pt[ELEV])
        
        stn_obs_prcp = stn_obs_prcp - stn_obs_prcp_mean
        stn_obs_prcp = stn_obs_prcp/stn_obs_prcp_std
        
        prcp,error = self.aclib.prcomp_interp(stns[LON], stns[LAT], stns[ELEV], 
                                              stn_wgts, stn_obs_prcp, a_pt[LON], 
                                              a_pt[LAT], a_pt[ELEV], pt_mean, pt_std, apply_mean=True, apply_std=True)
        
        prcp[prcp < MIN_PRCP_AMT] = MIN_PRCP_AMT
        
        if error:
            raise SVD_Exception("SVD did not converge.")
        
        
        prcp_fnl = np.zeros(po.size)
        prcp_fnl[po_mask] = prcp
        
        stn_obs_mean = np.mean(stn_obs,axis=0,dtype=np.float64)
        pt_ttl_mean = self.aclib.regress(stns[LON], stns[LAT], stns[ELEV], stn_wgts, stn_obs_mean, a_pt[LON], a_pt[LAT], a_pt[ELEV])
        
        ###################################################
        #calculate mean value and C.I.
        #TODO: Move this to C code?
        ###################################################
        x = np.array([1.,a_pt[LON],a_pt[LAT],a_pt[ELEV]])
        x.shape = (x.shape[0],1)
    
        X = np.column_stack((np.ones(stns.size),stns[LON],stns[LAT],stns[ELEV]))
        W = np.diag(stn_wgts)
        Y = stn_obs_mean
        Y.shape = (Y.shape[0],1)
        
        mean_val = np.float(np.dot(np.dot(np.dot(np.dot(np.transpose(x),np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))),np.transpose(X)),W),Y))
        
        if sigma != None and df != None: 
            inverse = np.linalg.inv(np.dot(np.dot(np.transpose(X),W),X))
            S = np.float(np.dot(np.dot(np.dot(np.dot(np.dot(np.dot(np.transpose(x),inverse),np.transpose(X)),np.square(W)),X),inverse),x))
            ci_r = np.abs((sigma*((1.0+S)**.5))*stats.t.ppf(0.025,df)) 
            ci_l = mean_val - ci_r
            ci_u = mean_val + ci_r
            se = np.nan
        else:
            ci_l = np.nan
            ci_u = np.nan
            se = np.nan
        ###################################################
        
        prcp_fnl = prcp_fnl*(pt_ttl_mean/np.mean(prcp_fnl))        

        return prcp_fnl,se,mean_val,(ci_l,ci_u)

class SVD_Exception(Exception):
    pass

class interp_po(object):
    '''
    classdocs
    '''

    def __init__(self,a_modeler):
        '''
        Constructor
        '''
        self.a_modeler = a_modeler
    
    def model_po(self,stns,stn_wgts,stn_obs,a_pt,thres=None):
        
        interp_vals = self.a_modeler.model(stns,stn_wgts,stn_obs,a_pt)        
        
        if thres is not None:
            
            mask_thres = interp_vals >= thres
            interp_vals[mask_thres] = 1
            interp_vals[np.logical_not(mask_thres)] = 0
        
        return interp_vals

class interp_prcp(object):
    '''
    classdocs
    '''

    def __init__(self,a_modeler):
        '''
        Constructor
        '''
        self.a_modeler = a_modeler
    
    def model_prcp(self,input,output):

        interp_vals,se,mean_val,ci = self.a_modeler.model(input.stns,input.stn_wgts,input.stn_obs,input.a_pt,input.sigma,input.df)       
        
        output.prcp = interp_vals
        output.se = se
        output.mean = mean_val
        output.ci = ci
        
        return output

def calc_hss(obs_po, mod_po):
    '''
    Calculates heidke skill score of modeled prcp occurrence
    See http://www.wxonline.info/topics/verif2.html
    @param obs: array of observed occurrences (1s and 0s)
    @param mod: array of modeled occurrences (1s and 0s)
    @return hss: heidke skill score
    '''
    
    #model_obs
    true_true = mod_po[np.logical_and(obs_po == 1, mod_po == 1)].size
    false_false = mod_po[np.logical_and(obs_po == 0, mod_po == 0)].size
    true_false = mod_po[np.logical_and(obs_po == 0, mod_po == 1)].size
    false_true = mod_po[np.logical_and(obs_po == 1, mod_po == 0)].size
    
    a = float(true_true)
    b = float(true_false)
    c = float(false_true)
    d = float(false_false)
    
    #special case handling
    if a == 0.0 and c == 0.0 and b != 0:
        #This means that were no observed days of rain so can't calc
        #appropriate hss. Set a = 1 to get a more appropriate hss
        a = 1.0
    
    if b == 0.0 and d == 0.0 and c != 0.0:
        #This means that there was observed rain every day so can't calc
        #appropriate hss. Set d = 1 to get a more appropriate hss
        d = 1.0    

    den = ((a + c) * (c + d)) + ((a + b) * (b + d))
    
    if den == 0.0:
        #This is a perfect forecast with all true_true or false_false
        return 1.0
    
    return (2.0 * ((a * d) - (b * c))) / den