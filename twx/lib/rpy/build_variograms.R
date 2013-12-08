# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################
library(rgdal)
library(gstat)
library(rgeos)


LCC_AREAS = c(	592917.1,#1	Appalachian
				211991.9,#2	California
				572719.6,#3	Desert
				535719.4,#4	Eastern Tallgrass Prairie and Big Rivers
				562694.8,#5	Great Basin
				743553.5,#6	Great Northern
				782015,#7	Great Plains
				392083.1,#8	Gulf Coast Prairie
				727747.3,#9	Gulf Coastal Plains and Ozarks
				294396.2,#10	North Atlantic
				183982.7,#11	North Pacific
				95676.18,#12	Peninsular Florida
				999025.4,#13	Plains and Prairie Potholes
				358421,#14	South Atlantic
				516745.7,#15	Southern Rockies
				660880.5)#16	Upper Midwest and Great Lakes
		
source('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/vario_fit.R')

load("/projects/daymet2/station_data/infill/infill_fnl/stns_tmin.RData")
STNS$col <- STNS@coords[,1] #lon
STNS$row <- STNS@coords[,2] #lat

ALBERS_PROJ4 <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
MAX_NGH_STNS <- 150

WIDTH <- 10
FITMETHOD <-7
LCC_NUM<-7
#fitmethod1: weight by # of points in lag
#fitmethod7: weight by # of points in lag and give smaller lags more weight


stn_mask <- STNS$neon == LCC_NUM
stns_lcc <- STNS[stn_mask,]
plot(stns_lcc)
avg_radius <- sqrt(MAX_NGH_STNS*(LCC_AREAS[LCC_NUM]/sum(stn_mask))*(1.0/pi))
CUTOFF = round((avg_radius*1.4)/WIDTH)*WIDTH

avar <- variogram(tair~lon+lat+elev+tdi+lst,stns_lcc,cutoff=CUTOFF,width=WIDTH)#avar$gamma[1]
avar.model <- my.autofit.gwvario(avar,model=c("Exp"),fix.values = c(avar$gamma[1],NA,0.6780442),start_vals = c(NA, NA,NA),fit.method=FITMETHOD)$var_model
plot(avar,model=avar.model)
print(avar.model)

#Fit variogram model to GLS residuals
g <- gstat(NULL,"val",tair~lon+lat+elev+tdi+lst,data=stns_lcc,model=avar.model)
gls_df <- predict(g,stns_lcc,BLUE=TRUE)
#residual = obs - predicted
stns_lcc$resid <- stns_lcc$tair - gls_df$val.pred
avar.gls <- variogram(resid~1,stns_lcc,cutoff=CUTOFF,width=WIDTH)#avar.gls$gamma[1]
avar.model.gls <- my.autofit.gwvario(avar.gls,model=c("Exp"),fix.values = c(avar.gls$gamma[1],NA,NA),fit.method=FITMETHOD)$var_model
plot(avar.gls,model=avar.model.gls)
print(avar.model.gls)

