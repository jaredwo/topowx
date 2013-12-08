'''
Created on Feb 8, 2012

@author: jared.oyler
'''

'''
    MERRA
    1/2 Lat * 2/3 Lon

56 km latitude
50 km longitude
'''

from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from utils.util_ncdf import ncdf_raster
import sys
import matplotlib.cm as cm  
import matplotlib
import mpl_toolkits.basemap as basem
import mpl_toolkits.basemap.cm as basecm
from scipy.interpolate import spline
import matplotlib.mlab as mlab
from scipy.misc import imresize
from PIL import Image
from scipy import stats

def executePCA(X):
    
    #Calculate covariance matrix with columns representing variables
    S = np.cov(X,rowvar=0)
    
    #Calculate eigenvalues/eigenvectors for S
    w,U = np.linalg.eig(S)
    #Calculate principal component scores for each grid cell
    P = np.dot(X,U)
    
    return S,w,U,P

def pca_analysis3():
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    
    eta_merra = ds_eta_merra.variables['et'][:,68,174]
    eta_tm = ds_eta_tm.variables['et'][:,68,174]
    
    plt.plot(eta_merra)
    plt.plot(eta_tm)
    plt.show()
    

def pca_analysis2():
    
    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_fmf.variables['mask'][:,:]

    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_merra = ds_eta_merra.variables['et'][0,:,:]
    eta_merra = np.reshape(eta_merra,(eta_merra.shape[0]*eta_merra.shape[1]))
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    
    mask_ndata = np.logical_not(eta_merra.mask)
    
    fmask = np.logical_and(np.ravel(cce_mask),mask_ndata)
    
    pcs_topo = np.load("/projects/daymet2/modis/pca/pcs_topo.npy")
    wgts_topo = np.load("/projects/daymet2/modis/pca/wgts_topo.npy")
    vari_topo = np.load("/projects/daymet2/modis/pca/var_topo.npy")
    print 100*vari_topo[0:15]
    pcs_merra = np.load("/projects/daymet2/modis/pca/pcs_merra.npy")
    wgts_merra = np.load("/projects/daymet2/modis/pca/wgts_merra.npy")
    vari_merra = np.load("/projects/daymet2/modis/pca/var_merra.npy")
    print 100*vari_merra[0:15]
    
    apc_topo = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    apc_merra = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    
    apc_topo[fmask] = wgts_topo[1,:]
    apc_merra[fmask] = wgts_merra[1,:]*-1
    
    apc_topo = np.reshape(apc_topo,cce_mask.shape)
    apc_merra = np.reshape(apc_merra,cce_mask.shape)
    
    apc_topo = np.ma.masked_array(apc_topo,np.isnan(apc_topo))
    apc_merra = np.ma.masked_array(apc_merra,np.isnan(apc_merra))
    
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    eta_topo = ds_eta_tm.variables['et'][:,:,:]
    
    mask_pc_tm_max = apc_topo >= np.percentile(apc_topo[np.logical_not(apc_topo.mask)],90)
    mask_pc_tm_min = apc_topo <= np.percentile(apc_topo[np.logical_not(apc_topo.mask)],10)
    
    plt.imshow(mask_pc_tm_max)
    plt.show()
    
    mask_pc_merra_max = apc_merra >= np.percentile(apc_merra[np.logical_not(apc_merra.mask)],90)
    mask_pc_merra_min = apc_merra <= np.percentile(apc_merra[np.logical_not(apc_merra.mask)],10)
    
    yrs,days = get_days_data(ds_eta_tm.variables['time'][:])
    udays = np.unique(days)
    
    pc_tm_min = []
    pc_merra_min = []
    
    pc_tm_max = []
    pc_merra_max = []  
    
    for day in udays:
        
        day_mask = days == day
        
        day_mean = np.mean(eta_topo[day_mask,:,:],axis=0)
        
        pc_tm_min.append(np.mean(day_mean[mask_pc_tm_min]))
        pc_tm_max.append(np.mean(day_mean[mask_pc_tm_max]))
        
        day_mean = np.mean(eta_merra[day_mask,:,:],axis=0)
        
        pc_merra_min.append(np.mean(day_mean[mask_pc_merra_min]))
        pc_merra_max.append(np.mean(day_mean[mask_pc_merra_max]))
        print day
    
    plt.subplot(211)  
    plt.plot(pc_tm_min)
    plt.plot(pc_tm_max)
    plt.subplot(212)  
    plt.plot(pc_merra_min)
    plt.plot(pc_merra_max)
    plt.show()
    
    
    
    
    

def pca_analysis():
    
    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_fmf.variables['mask'][:,:]

    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_merra = ds_eta_merra.variables['et'][0,:,:]
    eta_merra = np.reshape(eta_merra,(eta_merra.shape[0]*eta_merra.shape[1]))
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    
    mask_ndata = np.logical_not(eta_merra.mask)
    
    fmask = np.logical_and(np.ravel(cce_mask),mask_ndata)
    
    pcs_topo = np.load("/projects/daymet2/modis/pca/pcs_topo.npy")
    wgts_topo = np.load("/projects/daymet2/modis/pca/wgts_topo.npy")
    vari_topo = np.load("/projects/daymet2/modis/pca/var_topo.npy")
    print 100*vari_topo[0:15]
    pcs_merra = np.load("/projects/daymet2/modis/pca/pcs_merra.npy")
    wgts_merra = np.load("/projects/daymet2/modis/pca/wgts_merra.npy")
    vari_merra = np.load("/projects/daymet2/modis/pca/var_merra.npy")
    print 100*vari_merra[0:15]
    
    apc_topo = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    apc_merra = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    
    apc_topo[fmask] = wgts_topo[0,:]
    apc_merra[fmask] = wgts_merra[0,:]
    
    apc_topo = np.reshape(apc_topo,cce_mask.shape)
    apc_merra = np.reshape(apc_merra,cce_mask.shape)
    
    apc_topo = np.ma.masked_array(apc_topo,np.isnan(apc_topo))
    apc_merra = np.ma.masked_array(apc_merra,np.isnan(apc_merra))
    
    
    slope, intercept, r_value1, p_value, std_err = stats.linregress(pcs_topo[0,:],pcs_merra[0,:])
    slope, intercept, r_value2, p_value, std_err = stats.linregress(pcs_topo[1,:],pcs_merra[1,:]*-1)
    print r_value1,r_value2
    plt.subplot(211)
    plt.plot(pcs_topo[0,:],label="MOD16-Topo")
    plt.plot(pcs_merra[0,:],label="MOD16-Merra")
    plt.ylabel("PC Score",fontsize=20)
    plt.gca().set_xticklabels([])
    for label in plt.gca().get_yticklabels() + plt.gca().get_xticklabels():
        label.set_fontsize(15)
    plt.text(425,-580,"r=%.2f"%r_value1,fontsize=15)
    plt.title("PC1 Time Series",fontsize=20,weight="bold")
    
    plt.subplot(212)
    plt.plot(pcs_topo[1,:],label="MOD16-Topo")
    plt.plot(pcs_merra[1,:]*-1,label="MOD16-Merra")
    plt.ylabel("PC Score",fontsize=20)
    plt.xlabel('8-Day Composite #',fontsize=20)
    
    for label in plt.gca().get_yticklabels() + plt.gca().get_xticklabels():
        label.set_fontsize(15)
    plt.text(425,-180,"r=%.2f"%r_value2,fontsize=15)
    plt.title("PC2 Time Series",fontsize=20,weight="bold")
    plt.legend()
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/legend2.png',dpi=300)
    plt.show()
    sys.exit()
    
    
    
    
    r,c =  np.nonzero(apc_topo==np.max(apc_topo))
    r = r[0]
    c = c[0]
    
    plt.plot(ds_eta_tm.variables['et'][:,r,c])
    
    r,c =  np.nonzero(apc_topo==np.min(apc_topo))
    r = r[0]
    c = c[0]
    
    plt.plot(ds_eta_tm.variables['et'][:,r,c])
#    plt.show()
    ######################################
    plt.subplot(212)
    r,c =  np.nonzero(apc_merra==np.max(apc_merra))
    r = r[0]
    c = c[0]
    
    plt.plot(ds_eta_merra.variables['et'][:,r,c])
    
    r,c =  np.nonzero(apc_merra==np.min(apc_merra))
    r = r[0]
    c = c[0]
    
    plt.plot(ds_eta_merra.variables['et'][:,r,c])
    plt.show()
    

def pca_figs():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    
    
    
    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_fmf.variables['mask'][:,:]

    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))

    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_merra = ds_eta_merra.variables['et'][0,:,:]
    eta_merra = np.reshape(eta_merra,(eta_merra.shape[0]*eta_merra.shape[1]))
    
    mask_ndata = np.logical_not(eta_merra.mask)
    
    fmask = np.logical_and(np.ravel(cce_mask),mask_ndata)
    
    pcs_topo = np.load("/projects/daymet2/modis/pca/pcs_topo.npy")
    wgts_topo = np.load("/projects/daymet2/modis/pca/wgts_topo.npy")
    vari_topo = np.load("/projects/daymet2/modis/pca/var_topo.npy")
    print 100*vari_topo[0:15]
    pcs_merra = np.load("/projects/daymet2/modis/pca/pcs_merra.npy")
    wgts_merra = np.load("/projects/daymet2/modis/pca/wgts_merra.npy")
    vari_merra = np.load("/projects/daymet2/modis/pca/var_merra.npy")
    print 100*vari_merra[0:15]
    apc_topo = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    apc_merra = np.zeros((cce_mask.shape[0]*cce_mask.shape[1]))*np.nan
    
    apc_topo[fmask] = wgts_topo[1,:]
    apc_merra[fmask] = wgts_merra[1,:]*-1
    
    apc_topo = np.reshape(apc_topo,cce_mask.shape)
    apc_merra = np.reshape(apc_merra,cce_mask.shape)
    
    
    np.percentile(dem.vals,25)
    
    
    plt.figure(1,(13,6))
    #plt.subplot(121)
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    lines = m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')[-1]
    
    m.imshow(apc_merra,origin='upper',cmap=cm.RdBu_r)
    cb = m.colorbar()
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=15)
    plt.axes(imaxes)
    plt.title('MOD16-Merra PC2 Loadings (10% Var.)',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_pca2_merra.png',dpi=300)
    plt.show()
    sys.exit()
    
    #plt.subplot(122)
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    lines = m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')[-1]
    
    m.imshow(apc_topo,origin='upper',cmap=cm.RdBu_r)
    cb = m.colorbar()
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=15)
    plt.axes(imaxes)
    plt.title('MOD16-Topo PC2 Loadings (7% Var.)',fontsize=20,weight='bold')
    #plt.subplots_adjust(hspace=0.10, wspace=0.1)
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_pca2_merra.png',dpi=300)
    plt.show()

def pca():
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    
    
    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    fmf_mask = ds_fmf.variables['mask'][:,:]
    
    eta_tm_rs = np.reshape(eta_tm,(eta_tm.shape[0],eta_tm.shape[1]*eta_tm.shape[2]))
    eta_merra_rs = np.reshape(eta_merra,(eta_merra.shape[0],eta_merra.shape[1]*eta_merra.shape[2]))
    
    mask_ndata = np.logical_not(eta_tm_rs.mask[0,:])
    
    fmask = np.logical_and(np.ravel(fmf_mask),mask_ndata)
    
    eta_tm_rs =  np.ma.asarray(eta_tm_rs)
    eta_merra_rs =  np.ma.asarray(eta_merra_rs)
     
    eta_tm_rs_c = mlab.center_matrix(eta_tm_rs[:,fmask], dim=1)
    eta_merra_rs_c = mlab.center_matrix(eta_merra_rs[:,fmask], dim=1)

    eta_tm_rs_c = np.transpose(eta_tm_rs_c)
    eta_merra_rs_c = np.transpose(eta_merra_rs_c)

    print "start topo"
#    pcs,wgts,variance = mlab.prepca(eta_tm_rs_c)
#    np.save("/projects/daymet2/modis/pca/pcs_topo.npy", pcs)
#    np.save("/projects/daymet2/modis/pca/wgts_topo.npy", wgts)
#    np.save("/projects/daymet2/modis/pca/var_topo.npy", variance)
    print "start merra"
    pcs,wgts,variance = mlab.prepca(eta_merra_rs_c)
    np.save("/projects/daymet2/modis/pca/pcs_merra.npy", pcs)
    np.save("/projects/daymet2/modis/pca/wgts_merra.npy", wgts)
    np.save("/projects/daymet2/modis/pca/var_merra.npy", variance)
    print "done merra"
#    pc1_a = np.zeros((eta_merra.shape[1]*eta_merra.shape[2]))*np.nan
#    
#    pc1_a[fmask] = wgts[2,:]
#    print pc1_a.size
#    print eta_tm_rs.shape
#    
#    pc1 = np.reshape(pc1_a,(eta_tm.shape[1],eta_tm.shape[2]))
#    
#    plt.imshow(pc1)
#    plt.show()
#    
#    plt.plot(pcs[1,:],)
#    plt.show()
#    
#    print "done pca"


def met_data():
    
    ds_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    
    met_tm = ds_tm.variables['srad'][:,:,:]
    met_merra = ds_merra.variables['srad'][:,:,:]
    
    plt.imshow((np.mean(met_tm,axis=0)-np.mean(met_merra,axis=0))/np.mean(met_merra,axis=0)*100)
    plt.colorbar()
    plt.show()
    
    

def get_days_data(modis_8day):
    
    yrs = []
    days = []
    
    for x in np.arange(modis_8day.size):
        
        date_str = str(int(modis_8day[x]))
        
        yrs.append(int(date_str[0:4]))
        days.append(int(date_str[4:]))
    
    return np.array(yrs),np.array(days)

def map_pct_difs_seasonal():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)

    yrs,days = get_days_data(ds_eta_tm.variables['time'][:])
    
    janfeb_mask = np.logical_and(days >= 1,days <= 57)
    novdec_mask = days >= 305
    days_mask = np.logical_or(janfeb_mask,novdec_mask)
    
    eta_tm_mean = np.mean(eta_tm[days_mask,:,:],axis=0)
    eta_merra_mean = np.mean(eta_merra[days_mask,:,:],axis=0)
    
    #pct_difs = (eta_tm_mean-eta_merra_mean)/eta_merra_mean*100.
    pct_difs = eta_tm_mean-eta_merra_mean
    
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, dem.vals,3,colors='k')
    #plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    cmap = cm.RdBu_r
    
    
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    cmap.set_bad('grey',1.0)
    print type(pct_difs)
    norm = matplotlib.colors.Normalize(vmin=0,vmax=200)
    levels = np.arange(0,220,20)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)
    #m.imshow(pct_difs,origin='upper',cmap=cmap,norm=norm)
    m.imshow(pct_difs,origin='upper')
    cb = m.colorbar(extend="both")
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.show()


def elev_ann_cycle():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    elev = dem.vals
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    no_lc_mask = np.logical_and(lc != 16,np.logical_and(lc != 0, lc != 13))
    lc_mask = np.logical_and(lc ==7,no_lc_mask)
    
    elev1_mask = elev <= 1200
    elev2_mask = np.logical_and(elev>1200,elev<=1400)
    elev3_mask = np.logical_and(elev>1400,elev<=1800)
    elev4_mask = elev > 1800 
    
    yrs,days = get_days_data(ds_eta_tm.variables['time'][:])
    udays = np.unique(days)
    
    et_tm_elev1 = []
    et_merra_elev1 = []
    et_tm_elev2 = []
    et_merra_elev2 = []
    et_tm_elev3 = []
    et_merra_elev3 = []
    et_tm_elev4 = []
    et_merra_elev4 = []    
    
    for day in udays:
        
        day_mask = days == day
        
        day_mean = np.mean(eta_tm[day_mask,:,:],axis=0)
        
        et_tm_elev1.append(np.mean(day_mean[np.logical_and(lc_mask,elev1_mask)]))
        et_tm_elev2.append(np.mean(day_mean[np.logical_and(lc_mask,elev2_mask)]))
        et_tm_elev3.append(np.mean(day_mean[np.logical_and(lc_mask,elev3_mask)]))
        et_tm_elev4.append(np.mean(day_mean[np.logical_and(lc_mask,elev4_mask)]))
        
        day_mean = np.mean(eta_merra[day_mask,:,:],axis=0)
        et_merra_elev1.append(np.mean(day_mean[np.logical_and(lc_mask,elev1_mask)]))
        et_merra_elev2.append(np.mean(day_mean[np.logical_and(lc_mask,elev2_mask)]))
        et_merra_elev3.append(np.mean(day_mean[np.logical_and(lc_mask,elev3_mask)]))
        et_merra_elev4.append(np.mean(day_mean[np.logical_and(lc_mask,elev4_mask)]))
        
        print day

    xticks = [1,33,65,97,129,161,193,225,257,289,321,353]
    xlabs = ["1-JAN","2-FEB","6-MAR","7-APR","9-MAY","10-JUN","12-JUL","13-AUG","14-SEP","16-OCT","12-NOV","19-DEC"]
    plt.subplot(2,2,1)
    plt.plot(udays,et_tm_elev1)
    plt.plot(udays,et_merra_elev1)
    plt.xticks(xticks,xlabs)
    plt.legend(["TopoMet","Merra"])
    plt.title("Elev 1")
    
    plt.subplot(2,2,2)
    plt.plot(udays,et_tm_elev2)
    plt.plot(udays,et_merra_elev2)
    plt.xticks(xticks,xlabs)
    plt.legend(["TopoMet","Merra"])
    plt.title("Elev 2")
    
    plt.subplot(2,2,3)
    plt.plot(udays,et_tm_elev3)
    plt.plot(udays,et_merra_elev3)
    plt.xticks(xticks,xlabs)
    plt.legend(["TopoMet","Merra"])
    plt.title("Elev 3")
    
    plt.subplot(2,2,4)
    plt.plot(udays,et_tm_elev4)
    plt.plot(udays,et_merra_elev4)
    plt.xticks(xticks,xlabs)
    plt.legend(["TopoMet","Merra"])
    plt.title("Elev 4")
    
    plt.show()
        
def overall_ann_cycle():
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    no_lc_mask = np.logical_and(lc != 16,np.logical_and(lc != 0, lc != 13))
    lc_mask = np.logical_and(lc >=0,no_lc_mask)
    
    yrs,days = get_days_data(ds_eta_tm.variables['time'][:])
    udays = np.unique(days)
    
    et_tm_vals = []
    et_merra_vals = []
    
    for day in udays:
        
        day_mask = days == day
        
        day_mean = np.mean(eta_tm[day_mask,:,:],axis=0)
        et_tm_vals.append(np.mean(day_mean[lc_mask]))
        
        day_mean = np.mean(eta_merra[day_mask,:,:],axis=0)
        et_merra_vals.append(np.mean(day_mean[lc_mask]))
        
        print day

    xticks = [1,33,65,97,129,161,193,225,257,289,321,353]
    xlabs = ["1-JAN","2-FEB","6-MAR","7-APR","9-MAY","10-JUN","12-JUL","13-AUG","14-SEP","16-OCT","12-NOV","19-DEC"]
    plt.plot(udays,et_tm_vals)
    plt.plot(udays,et_merra_vals)
    plt.xticks(xticks,xlabs)
    plt.legend(["TopoMet","Merra"])
    plt.show()

def normRGB(rgb):
    return np.array(rgb,dtype=np.float64)/255.

def elev_ann_difs():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    lc = np.ravel(lc)
    m_eta_tm = np.ravel(m_eta_tm)
    m_eta_merra = np.ravel(m_eta_merra)
    elev = np.ravel(dem.vals)
    
    #1 = enf
    #10 = grass
    #7 = open shrub
    
    no_lc_mask = np.logical_and(lc != 16,np.logical_and(lc != 0, lc != 13))
    
    lc_mask = np.logical_and(lc == 7,no_lc_mask)
        
    m_eta_tm = m_eta_tm[lc_mask]
    m_eta_merra = m_eta_merra[lc_mask]
    elev = elev[lc_mask]
    
    elev1_mask = elev <= 1200
    elev2_mask = np.logical_and(elev>1200,elev<=1400)
    elev3_mask = np.logical_and(elev>1400,elev<=1800)
    elev4_mask = elev > 1800 
    
    pct_difs = ((m_eta_tm-m_eta_merra)/m_eta_merra)*100.
    
    print pct_difs[elev1_mask].size,pct_difs[elev2_mask].size,pct_difs[elev3_mask].size,pct_difs[elev4_mask].size
    
    plt.boxplot((pct_difs[elev1_mask],pct_difs[elev2_mask],pct_difs[elev3_mask],pct_difs[elev4_mask]))
    plt.ylim((-50,50))
    plt.hlines(0,*plt.xlim())
    plt.show()


def overall_ann_boxplots():
    #plt.figure(1,(13,6))
    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_fmf.nc')
    fmf_mask = ds_fmf.variables['mask'][:,:]
    
    ds_glac = Dataset('/projects/daymet2/modis/crown_mask_glac.nc')
    glac_mask = ds_glac.variables['mask'][:,:]
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
#    plt.subplot(121)
#    plt.imshow(fmf_mask)
#    plt.subplot(122)
#    plt.imshow(glac_mask)
#    plt.show()
    
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    no_lc_mask = np.logical_and(lc != 16,np.logical_and(lc != 0, lc != 13))
    lc_mask = np.logical_and(lc >=0,no_lc_mask)
    lc_mask = np.logical_and(cce_mask,lc_mask)
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    all_mask = np.logical_and(lc_mask,fmf_mask)
    #plt.grid(zorder=-1)
    plt.gca().yaxis.grid(zorder=-1)
    bp = plt.boxplot((m_eta_merra[lc_mask],m_eta_tm[lc_mask],m_eta_merra[all_mask],m_eta_tm[all_mask]), patch_artist=True)
    plt.setp(bp['medians'], lw=2,color="blue")
    plt.setp(bp['boxes'], lw=2,color="grey")
    plt.setp(bp['whiskers'], lw=2,color="black")
    plt.setp(bp['caps'],lw=2,color="black")
    plt.setp(bp['fliers'],markeredgewidth=2,color="black")
    plt.xticks([1,2,3,4],["MOD16-Merra\nEntire CCE","MOD16-Topo\nEntire CCE","MOD16-Merra\nMFF Basin","MOD16-Topo\nMFF Basin"],fontsize=15)
    plt.ylabel('mm/yr',fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Overall ET Distributions',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/boxplot_ann_et.png',dpi=300)
    plt.show()
    #all_mask = fmf_mask
    
    print (np.sum(m_eta_tm[all_mask]) - np.sum(m_eta_merra[all_mask]))/np.sum(m_eta_merra[all_mask])
    print "Mean TopoMet: "+str(np.sum(m_eta_tm[all_mask]))
    print "Mean Merra: "+str(np.sum(m_eta_merra[all_mask]))
    
    print "STD TopoMet: "+str(np.std(m_eta_tm[all_mask],ddof=1))
    print "STD Merra: "+str(np.std(m_eta_merra[all_mask],ddof=1))
    
    

def overall_ann_difs():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    lc = np.ravel(lc)
    m_eta_tm = np.ravel(m_eta_tm)
    m_eta_merra = np.ravel(m_eta_merra)
    elev = np.ravel(dem.vals)
    #1 = enf
    #10 = grass
    #7 = open shrub
    lc_mask = lc ==1
    
    m_eta_tm = m_eta_tm[lc_mask]
    m_eta_merra = m_eta_merra[lc_mask]
    elev = elev[lc_mask]
    
    #plt.subplot(2,1,1)
    #plt.plot(elev,m_eta_tm,'.')
    #plt.title('TopoMet')
    
    #plt.subplot(2,1,2)
    
    print "Mean TopoMet: "+str(np.mean(m_eta_tm))
    print "Mean Merra: "+str(np.mean(m_eta_merra))
    
    print "STD TopoMet: "+str(np.std(m_eta_tm,ddof=1))
    print "STD Merra: "+str(np.std(m_eta_merra,ddof=1))
    
    #plt.boxplot((m_eta_tm,m_eta_merra))
    plt.boxplot((m_eta_tm-m_eta_merra)/m_eta_merra)
    #plt.plot(elev,(m_eta_tm-m_eta_merra)/m_eta_merra,'.')
#    plt.title('Merra')
    
    plt.show()
    

def map_elev():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))
    
    plt.figure(1,(13,6))
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y,elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    lines = m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')[-1]

    #[1200,1500,1800]
    print np.percentile(dem.vals,25)
    print np.percentile(dem.vals,50)
    print np.percentile(dem.vals,75)
    
    m.imshow(elev,origin='upper',cmap=cm.spectral)
    cb = m.colorbar()

    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    
    cb.set_label('meters',fontsize=20)
    plt.title('CCE Elevation',fontsize=20,weight='bold')
    plt.legend([lines],["Middle Fork\nFlathead Basin"],loc=3)
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_elev.png',dpi=300)
    
    plt.show()


def map_lc():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    lc = np.ma.masked_array(lc,np.logical_not(cce_mask))
    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))
    
    plt.figure(1,(13,6))
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    lines = m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')[-1]
    
    '''
    "water = 0\n \
evergreen needleleaf forest = 1\n \
evergreen broadleaf forest = 2\n \
deciduous needleleaf forest = 3\n \
deciduous broadleaf forest = 4\n \
mixed forests = 5\n \
closed shrublands = 6\n \
open shrublands = 7\n \
woody savannas = 8\n \
savannas = 9\n \
grasslands = 10\n \
croplands = 12\n \
urban and built-up = 13\n \
barren or sparsely vegetated = 16\n \
unclassfied = 254"
    '''  
    colors = [normRGB((68,79,137)), #water
              normRGB((1,100,0)), #enf
              normRGB((1,130,0)), #ebf
              normRGB((151,191,71)), #dnf
              normRGB((2,220,0)), #dbf
              normRGB((0,255,0)), #mf
              normRGB((255,173,0)), #closed shrub
              normRGB((255,251,195)), #open shrub
              normRGB((220,206,0)), #woody savannah
              normRGB((255,255,255)), #savanna
              normRGB((140,72,9)), #grasslands
              normRGB((247,165,255)), #cropland
              'red', #urban
              'grey'] #bare ground
             
    
    cmap = matplotlib.colors.ListedColormap(colors)
    
    levels = [0,1,2,3,4,5,6,7,8,9,10,12,13,16,17]
    tick_locs = [.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,11,12.5,14.5,16.5]
    tick_labs = ['water',
                 'evergreen needleleaf forest',
                 'evergreen broadleaf forest',
                 'deciduous needleleaf forest',
                 'deciduous broadleaf forest',
                 'mixed forests',
                 'closed shrublands',
                 'open shrublands',
                 'woody savannas',
                 'savannas',
                 'grasslands',
                 'croplands',
                 'urban and built-up',
                 'barren']
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,ncolors=len(levels))
    
    m.imshow(lc,origin='upper',cmap=cmap,norm=norm)
    cb = m.colorbar()
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=15)
    plt.axes(imaxes)
    cb.set_ticks(tick_locs)
    cb.set_ticklabels(tick_labs)
    plt.title('CCE Land Cover',fontsize=20,weight='bold')
    plt.text(-112.5,46.55,'Contours = Elevation (m)',fontsize=15)
    plt.legend([lines],["Middle Fork\nFlathead Basin"],loc=3)
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_lc.png',dpi=300)
    plt.show()

def map_ann_et():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))
    
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    mask_nodata = m_eta_tm.mask
    
    m_eta_tm = np.ma.masked_array(np.asarray(m_eta_tm),np.logical_or(mask_nodata,np.logical_not(cce_mask)))
    m_eta_merra = np.ma.masked_array(np.asarray(m_eta_merra),np.logical_or(mask_nodata,np.logical_not(cce_mask)))
    
    plt.figure(1,(13,6))
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y,elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='blue')
    
    cmap = cm.RdBu_r
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    #cmap.set_bad('grey',1.0)
    norm = matplotlib.colors.Normalize(vmin=100,vmax=600)
    levels = np.arange(100,625,50)
    #levels = np.arange(-50,60,10)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)

#    norm = matplotlib.colors.BoundaryNorm(levels,
#                        ncolors=256, clip = False)
    m.imshow(m_eta_tm,origin='upper',norm=norm,cmap=cmap)
    cb = m.colorbar(extend="both")
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    plt.text(-112.5,46.55,'Contours = Elevation (m)',fontsize=15)
    
    cb.set_label('ET (mm/yr)',fontsize=20)
    plt.title('MOD16-Topo',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_annet_topo.png',dpi=300)
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.show()


def elev_vs_et():
    
    #fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8,8))
    #print axes.flat
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    ds_srad_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_srad_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    srad_tm = ds_srad_tm.variables['srad'][:,:,:]
    srad_merra = ds_srad_merra.variables['srad'][:,:,:]
    vpd_tm = ds_srad_tm.variables['vpd'][:,:,:]
    vpd_merra = ds_srad_merra.variables['vpd'][:,:,:]
    tmin_tm = ds_srad_tm.variables['tmin'][:,:,:]
    tmin_merra = ds_srad_merra.variables['tmin'][:,:,:]
    
    m_srad_tm = np.mean(srad_tm,axis=0)
    m_srad_merra = np.mean(srad_merra,axis=0)
        
    m_vpd_tm = np.mean(vpd_tm,axis=0)
    m_vpd_merra = np.mean(vpd_merra,axis=0)
    
    m_tmin_tm = np.mean(tmin_tm,axis=0)
    m_tmin_merra = np.mean(tmin_merra,axis=0)
    
    difs_srad = m_srad_tm-m_srad_merra#)/m_srad_merra*100.
    difs_tmin = m_tmin_tm-m_tmin_merra
    difs_vpd = m_vpd_tm-m_vpd_merra

#    ds_fmf = Dataset('/projects/daymet2/modis/crown_mask_fmf.nc')
#    fmf_mask = ds_fmf.variables['mask'][:,:]
    
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    no_lc_mask = np.logical_and(lc != 16,np.logical_and(lc != 0, lc != 13))
    no_lc_mask = np.logical_and(no_lc_mask,cce_mask)
    
    lc_mask_shrub = np.logical_and(lc ==7,no_lc_mask)
    lc_mask_enf = np.logical_and(lc ==1,no_lc_mask)
    lc_mask_grass = np.logical_and(lc ==10,no_lc_mask)

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    difs_et = m_eta_tm-m_eta_merra#)/m_eta_merra*100.
    
    #all_mask = np.logical_and(lc_mask,fmf_mask)
    
    #plt.rcParams['xtick.labelsize'] = 15
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,10))
    print axes.shape
    fig.subplots_adjust(hspace=0.10, wspace=0.1)
    
#    for ax in axes.flat:
#        # Hide all ticks and labels
#        ax.xaxis.set_visible(False)
#        ax.yaxis.set_visible(False)
    
    col = normRGB((23,55,94))
    
    axes[0,0].plot(difs_et[lc_mask_enf],difs_tmin[lc_mask_enf],'.',color=col)
    axes[0,1].plot(difs_et[lc_mask_grass],difs_tmin[lc_mask_grass],'.',color=col)
    axes[0,2].plot(difs_et[lc_mask_shrub],difs_tmin[lc_mask_shrub],'.',color=col)
    
    axes[0,0].set_ylim((-6,3))
    axes[0,1].set_ylim((-6,3))
    axes[0,2].set_ylim((-6,3))
    axes[0,0].set_ylabel(u'Tmin(\u00b0C)',fontsize=20)
    axes[0,1].set_ylabel((-6,3))
    axes[0,2].set_ylabel((-6,3))
    axes[0,0].set_xlim((-100,150))
    axes[0,1].set_xlim((-150,250))
    axes[0,2].set_xlim((-150,200))
    axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(),fontsize=20)
    
    axes[0,0].xaxis.set_visible(False)
    axes[0,1].xaxis.set_visible(False)
    axes[0,2].xaxis.set_visible(False)
    axes[0,1].yaxis.set_visible(False)
    axes[0,2].yaxis.set_visible(False)
    
    axes[0,0].set_title("Evergreen\nNeedleleaf",fontsize=20)
    axes[0,1].set_title("Grassland",fontsize=20)
    axes[0,2].set_title("Open\nShrubland",fontsize=20)
    
    axes[1,0].plot(difs_et[lc_mask_enf],difs_vpd[lc_mask_enf],'.',color=col)
    axes[1,1].plot(difs_et[lc_mask_grass],difs_vpd[lc_mask_grass],'.',color=col)
    axes[1,2].plot(difs_et[lc_mask_shrub],difs_vpd[lc_mask_shrub],'.',color=col)
    axes[1,0].set_ylim((-400,500))
    axes[1,1].set_ylim((-400,500))
    axes[1,2].set_ylim((-400,500))
    axes[1,0].set_ylabel('VPD(Pa)',fontsize=20)
    axes[1,0].set_xlim((-100,150))
    axes[1,1].set_xlim((-150,250))
    axes[1,2].set_xlim((-150,200))
    axes[1,0].xaxis.set_visible(False)
    axes[1,1].xaxis.set_visible(False)
    axes[1,2].xaxis.set_visible(False)
    axes[1,1].yaxis.set_visible(False)
    axes[1,2].yaxis.set_visible(False)
    
    axes[2,0].plot(difs_et[lc_mask_enf],difs_srad[lc_mask_enf],'.',color=col)
    axes[2,1].plot(difs_et[lc_mask_grass],difs_srad[lc_mask_grass],'.',color=col)
    axes[2,2].plot(difs_et[lc_mask_shrub],difs_srad[lc_mask_shrub],'.',color=col)
    axes[2,0].set_ylim((-60,120))
    axes[2,1].set_ylim((-60,120))
    axes[2,2].set_ylim((-60,120))
    axes[2,0].set_ylabel('SRAD(w/m-2)',fontsize=20)
    axes[2,0].set_xlim((-100,150))
    axes[2,1].set_xlim((-150,250))
    axes[2,2].set_xlim((-150,200))
    axes[2,1].yaxis.set_visible(False)
    axes[2,2].yaxis.set_visible(False)
    axes[2,0].set_xlabel('ET (mm/yr)',fontsize=20)
    axes[2,1].set_xlabel('ET (mm/yr)',fontsize=20)
    axes[2,2].set_xlabel('ET (mm/yr)',fontsize=20)
    
    #plt.suptitle('MOD16-Topo Minus MOD16-Merra\nMeterological vs. ET Avg. Annual Differences',fontsize=20,weight='bold')
    
#    axes[0,0].yaxis.set_ticks_position('left')
#    axes[0,0].yaxis.set_visible(True)
#    axes[0,0].set_ylim((-150,250))
    
#    for row in [0,1,2]:
#        
#        for col in [0,1,2]:
#            
#            axes[x,y].plot(pct_difs_srad[lc_mask],pct_difs_et[lc_mask],'.')
    
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/scatter_difs.png',dpi=300)
    
    plt.show()
    #axes[0,1].plot(pct_difs_srad[lc_mask],pct_difs_et[lc_mask],'.')
    #axes[0,2].plot(pct_difs_srad[lc_mask],pct_difs_et[lc_mask],'.')
    #plt.subplot(121)
    #plt.plot(pct_difs_srad[lc_mask],pct_difs_et[lc_mask],'.')
    #plt.subplot(122)
    #plt.plot(m_srad_merra[all_mask],m_eta_merra[all_mask],'.')
    

    
    #plt.show()
    

def transect_map():
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    lc = np.ma.masked_array(lc,np.logical_not(cce_mask))
    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))
    
    plt.figure(1,(13,6))
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    lines = m.readshapefile('/projects/crown_ws/crownboundary_GIS/crownofthecontinentWGS84/coc_site2_wgs84_usa','coc_site2_wgs84_usa', drawbounds=True,linewidth=2,color='k')[-1]
    
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    m.plot((-113.676,-112.856),(48.4505,48.4505),lw=3)
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/transect_map.png',dpi=300)
    plt.show()
    #lines = m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')[-1]

def transect():
    
    
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

#m.plot((-114.113,-113.657),(47.3639,47.3639))
    #missions
#    x1,y1 = dem.getGridCellOffset(-114.439, 47.3351)
#    x2,y2 = dem.getGridCellOffset(-113.657, 47.3351)
    
    #FMF to Front
    x1,y1 = dem.getGridCellOffset(-113.676, 48.4505)
    x2,y2 = dem.getGridCellOffset(-112.856, 48.4505)
    
    
    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    fig = plt.figure(1,(15,6))
    ax1 = fig.add_subplot(111)
    lons =  np.around(dem.lons[x1:x2],3)
    line1 = ax1.plot(lons,m_eta_tm[y1,x1:x2],label='MOD16-Topo')
    line2 = ax1.plot(lons,m_eta_merra[y1,x1:x2],label='MOD16-Merra')
    ticks = ax1.get_xticks()
    ticks = [str(x) for x in ticks]
    
    ax1.set_xlabel("Longitude",fontsize=15)
    ax1.set_ylabel("Avg. Ann. ET (mm/yr)",fontsize=15)
    ax1.set_xticklabels(ticks,rotation=-10)
    ax1.grid()
    ax2 = ax1.twinx()
    line3 = ax2.plot(lons,dem.vals[y1,x1:x2],color="red")
    ax2.set_xticklabels(ticks,rotation=-10)
    ax2.set_ylabel("Elevation (m)",fontsize=15)
    ax1.legend([line1,line2,line3],['MOD16-Topo','MOD16-Merra','Elevation'])
    plt.title("ET Transect",fontsize=20,weight="bold")
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/transect.png',dpi=300)
    plt.show()

def map_srad():

    plt.figure(1,(13,6))
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    eta_tm = ds_eta_tm.variables['srad'][:,:,:]
    eta_merra = ds_eta_merra.variables['srad'][:,:,:]
    
    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))
    eta_tm = np.ma.masked_array(np.mean(eta_tm,axis=0),np.logical_not(cce_mask))
    eta_merra = np.ma.masked_array(np.mean(eta_merra,axis=0),np.logical_not(cce_mask))

    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
                llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
    
    #m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='red')
    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
#    x, y = m(*np.meshgrid(dem.lons,dem.lats))
#    cs = m.contour(x,y,elev,3,colors='k')
#    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    m.imshow(eta_tm,origin='upper')
    cb = m.colorbar()
    cb.set_label('W m-2',fontsize=20)
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    plt.title("Avg. SRAD: Topo-Informed",fontsize=20,weight="bold")
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_srad_topo.png',dpi=300)
    plt.show()

def map_difs_srad():
    
    plt.figure(1,(9.5,6.25))
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    eta_tm = ds_eta_tm.variables['srad'][:,:,:]
    eta_merra = ds_eta_merra.variables['srad'][:,:,:]

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)

    pct_difs = (m_eta_tm-m_eta_merra)/m_eta_merra*100.
    
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='blue')
    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, dem.vals,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    cmap = cm.RdBu_r
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    cmap.set_bad('grey',1.0)
    print type(pct_difs)
    norm = matplotlib.colors.Normalize(vmin=-20,vmax=20)
    levels = np.arange(-20,25,5)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)
    m.imshow(pct_difs,origin='upper',cmap=cmap,norm=norm)
    cb = m.colorbar(extend="both")
    
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    cb.set_label('% Difference',fontsize=20)
    plt.text(-112.7,46.55,'Contours = Elevation (m)',fontsize=15)
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.title('2000-2009 Avg. SRAD\nTopoMet % Difference From Merra',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_srad_dif.png',dpi=300)
    plt.show()

def map_difs_vpd():
    
    plt.figure(1,(9.5,6.25))
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    eta_tm = ds_eta_tm.variables['vpd'][:,:,:]
    eta_merra = ds_eta_merra.variables['vpd'][:,:,:]

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)

    pct_difs = (m_eta_tm-m_eta_merra)/m_eta_merra*100.
    
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='blue')
    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, dem.vals,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    cmap = cm.RdBu_r
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    cmap.set_bad('grey',1.0)
    print type(pct_difs)
    norm = matplotlib.colors.Normalize(vmin=-50,vmax=50)
    levels = np.arange(-50,60,10)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)
    m.imshow(pct_difs,origin='upper',cmap=cmap,norm=norm)
    cb = m.colorbar(extend="both")
    
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    cb.set_label('% Difference',fontsize=20)
    plt.text(-112.7,46.55,'Contours = Elevation (m)',fontsize=15)
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.title('2000-2009 Avg. VPD\nTopoMet % Difference From Merra',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_vpd_dif.png',dpi=300)
    plt.show()

def map_difs_tmin():
    
    plt.figure(1,(9.5,6.25))
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/crown_topomet_8day.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/crown_merra.nc')
    eta_tm = ds_eta_tm.variables['tmin'][:,:,:]
    eta_merra = ds_eta_merra.variables['tmin'][:,:,:]

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)

    pct_difs = m_eta_tm-m_eta_merra#)#/m_eta_merra*100.
    
    plt.imshow(m_eta_tm)
    plt.show()
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='blue')
    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y, dem.vals,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    cmap = cm.RdBu_r
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    cmap.set_bad('grey',1.0)
    print type(pct_difs)
    norm = matplotlib.colors.Normalize(vmin=-3,vmax=3)
    levels = np.arange(-3,3.5,.5)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)
    m.imshow(pct_difs,origin='upper',cmap=cmap,norm=norm)
    cb = m.colorbar(extend="both")
    
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    cb.set_label(u'\u00b0C Difference',fontsize=20)
    plt.text(-112.7,46.55,'Contours = Elevation (m)',fontsize=15)
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.title('2000-2009 Avg. Tmin\nTopoMet Minus Merra',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_tmin_dif.png',dpi=300)
    plt.show()

def map_pct_difs():
    
    plt.figure(1,(13,6))
    
    ds_cce = Dataset('/projects/daymet2/modis/crown_mask_crown.nc')
    cce_mask = ds_cce.variables['mask'][:,:]
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

    elev = np.ma.masked_array(dem.vals,np.logical_not(cce_mask))

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    mask_nodata = m_eta_tm.mask
    
    m_eta_tm = np.ma.masked_array(np.asarray(m_eta_tm),np.logical_or(mask_nodata,np.logical_not(cce_mask)))
    m_eta_merra = np.ma.masked_array(np.asarray(m_eta_merra),np.logical_or(mask_nodata,np.logical_not(cce_mask)))

    pct_difs = (m_eta_tm-m_eta_merra)/m_eta_merra*100.
    
    plt.imshow(pct_difs)
    plt.show()
    sys.exit()
    
    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')

    m.readshapefile('/projects/daymet2/modis/fmf','fmf', drawbounds=True,linewidth=2,color='blue')
    parallels = np.arange(0,90,1)
    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=15)
    
    meridians = np.arange(-120.,-100,1)
    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=15)
    x, y = m(*np.meshgrid(dem.lons,dem.lats))
    cs = m.contour(x,y,elev,3,colors='k')
    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
    
    cmap = cm.RdBu_r
    
    #cmap.set_under("blue",1.)
    cmap.set_over('red',1.0)
    cmap.set_under('purple',1.0)
    #cmap.set_bad('grey',1.0)
    print type(pct_difs)
    norm = matplotlib.colors.Normalize(vmin=-40,vmax=40)
    levels = np.arange(-40,50,10)
    #cs = m.contourf(x,y,pct_difs,levels,cmap=cmap,norm=norm,extend="both")
    
#    cs.cmap.set_over('red')
#    cs.cmap.set_under('purple')
#    cs.cmap.set_bad('grey')
    
#    #cmap = cm.GMT_wysiwyg
#    
#    
#    

    norm = matplotlib.colors.BoundaryNorm(levels,
                        ncolors=256, clip = False)
    m.imshow(pct_difs,origin='upper',cmap=cmap,norm=norm)
    cb = m.colorbar(extend="both")
    
    #m.plot((-114.113,-113.657),(47.3639,47.3639))
    
    imaxes = plt.gca()
    plt.axes(cb.ax)
    plt.yticks(fontsize=20)
    plt.axes(imaxes)
    cb.set_label('% Difference',fontsize=20)
    plt.text(-112.5,46.55,'Contours = Elevation (m)',fontsize=15)
#    cb.set_label('%')
    #cb = m.colorbar(cs)
    plt.title('ET % Dif. MOD16-Topo vs. MOD16-Merra',fontsize=20,weight='bold')
    plt.savefig('/projects/daymet2/docs/agu_chapman2012_poster/map_ann_et_pct.png',dpi=300)
    plt.show()
    

def single_pt():
    
    #Shrubland RMF at GNP (+ dif)
    #lon=-113.2
    #lat=48.8

    #Grassland RMF S. Bob (-dif)
#    lon=-112.536
#    lat=47.61

    lon=-113.738
    lat=48.5939
    
    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    x,y =  dem.getGridCellOffset(lon,lat)
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,y,x]
    eta_merra = ds_eta_merra.variables['et'][:,y,x]

    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    print ds_lc.variables['lc'][y,x]
    
    print (np.sum(eta_tm)-np.sum(eta_merra))/np.sum(eta_merra)*100
    #plt.plot(eta_tm-eta_merra)
    plt.plot(eta_tm)
    plt.plot(eta_merra)
    plt.legend(["TopoMet","Merra"])
    plt.show()
    

def one_one_plots():
    
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    eta_tm = ds_eta_tm.variables['et'][:,:,:]
    eta_merra = ds_eta_merra.variables['et'][:,:,:]

    m_eta_tm = np.mean(eta_tm,axis=0)
    m_eta_merra = np.mean(eta_merra,axis=0)
    
    et_mask = np.logical_not(m_eta_tm.mask)
    
    lc_mask = np.logical_and(lc>=0,et_mask)
    
    print np.mean(np.ravel(m_eta_tm[lc_mask])-np.ravel(m_eta_merra[lc_mask]))
    print np.mean(np.abs(np.ravel(m_eta_tm[lc_mask])-np.ravel(m_eta_merra[lc_mask])))
    
    print np.mean((np.ravel(m_eta_tm[lc_mask])-np.ravel(m_eta_merra[lc_mask]))/np.ravel(m_eta_merra[lc_mask])*100.)
    print np.mean(np.abs(np.ravel(m_eta_tm[lc_mask])-np.ravel(m_eta_merra[lc_mask]))/np.ravel(m_eta_merra[lc_mask])*100.)
    
    pct_difs = (m_eta_tm-m_eta_merra)/m_eta_merra*100.
    
    plt.imshow(pct_difs)
    plt.colorbar()
    plt.show()
    
    
    plt.plot(np.ravel(m_eta_merra[lc_mask]),np.ravel(m_eta_tm[lc_mask]),'.')
    
    plt.show()
    
def land_cover_cnts():
    
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    lc = ds_lc.variables['lc'][:,:]
    
    ttl = float(lc.size)
    
    u_lc = np.unique(lc)
    
    for alc in u_lc:
        
        print "%d = %d (%.2f)"%(alc,lc[lc==alc].size,(lc[lc==alc].size/ttl)*100.)
    
if __name__ == '__main__':
    
    #map_srad()
    pca_analysis()
    #map_ann_et()
    #pca_analysis3()
    #transect_map()
    #overall_ann_boxplots()
    #pca_figs()
    #pca()
    #elev_vs_et()
    #transect()
    #map_difs_srad()
    #map_difs_vpd()
    #map_difs_tmin()
    #map_pct_difs()
    #overall_ann_boxplots()
    #one_one_plots()
    #overall_ann_difs()
    #map_lc()
    #map_elev()
    #met_data()
    #single_pt()
    #elev_ann_cycle()
    #map_lc()
    #map_pct_difs_seasonal()
    #overall_ann_cycle()
    #elev_ann_difs()
    #elev_grad()
    #map_elev()
    #map_lc()
    #map_ann_et()
    #land_cover_cnts()
    #map_pct_difs()
    #one_one_plots()
    sys.exit()

    dem = ncdf_raster('/projects/daymet2/modis/crown_dem.nc','elev')
    ds_et_tm = Dataset('/projects/daymet2/modis/mod16_topomet.nc')
    ds_et_merra = Dataset('/projects/daymet2/modis/mod16_merragmao.nc')
    ds_lc = Dataset('/projects/daymet2/modis/crown_landcover.nc')
    ds_eta_tm = Dataset('/projects/daymet2/modis/mod16ann_topomet.nc')
    ds_eta_merra = Dataset('/projects/daymet2/modis/mod16ann_merragmao.nc')
    
    a_et_tm = ds_eta_tm.variables['et'][:,:,:]
    plt.imshow(np.mean(a_et_tm,axis=0))
    plt.show()
    sys.exit()
    
#    
#    m = Basemap(projection='cyl',llcrnrlat=dem.min_y,urcrnrlat=dem.max_y,\
#            llcrnrlon=dem.min_x,urcrnrlon=dem.max_x,resolution='c')
#    
#    #m.drawstates()
#
#    parallels = np.arange(0,90,1)
#    m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#    
#    meridians = np.arange(-120.,-100,1)
#    m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#    x, y = m(*np.meshgrid(dem.lons,dem.lats))
#    print x.shape
#    cs = m.contour(x,y, dem.vals,3,colors='k')
#    plt.clabel(cs,inline=1,fontsize=10,fmt='%d')
#    
    a_et_tm = ds_et_tm.variables['et'][:,:,:]
    a_et_merra = ds_et_merra.variables['et'][:,:,:]
    
    mtm = np.mean(a_et_tm, axis=0)
    mmerra = np.mean(a_et_merra, axis=0)
    
    plt.plot(np.ravel(mtm),np.ravel(mmerra),'.')
    plt.show()
    
    #a2 = np.reshape(a_et,(a_et.shape[0],a_et.shape[1]*a_et.shape[2]))
    
    #plt.plot(a2[:,1])
    #plt.plot(a_et[:,1,0])
    #plt.show()
    
#    m.imshow(np.mean(a_et_tm, axis=0)-np.mean(a_et_merra, axis=0),origin='upper')
#    #m.imshow(dem.vals,origin='upper')
#    cb = m.colorbar()
#    cb.set_label('kg/m2/8day')
#    #plt.contour(dem.vals,origin='upper')
#    #m.drawcountries()
#    
#    plt.show()
    
    
    