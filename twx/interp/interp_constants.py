'''
Created on Oct 16, 2012

@author: jared.oyler
'''
import numpy as np

RM_STN_IDS_TAIR = np.array(['RAWS_NLIM','RAWS_OKEE','RAWS_NCHA','GHCN_CA002503650','RAWS_NFX1'])
AREA_MONTANA_BUFFER = 1519900.0 #km-2 lon: -119,-101,lat: 42,52
NNGH_RANGE = 10 #the number of nghs to use for interpolation should be between min_ngh and min_ngh+NNGH_RANGE
TILES_MONTANA = np.concatenate([np.arange(4,11),np.arange(24,31),np.arange(49,55)]) #the tile numbers encompassing montana
SCALE_FACTOR = np.float32(0.01) #factor by which interp outputs are scaled. everything is stored as int16