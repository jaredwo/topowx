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

ALBERS_PROJ4 <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"
MAX_NGH_STNS <- 150
WIDTH <- 2
FITMETHOD <-7
LCC_NUM<-6#15#6#13#7#5
#fitmethod1: weight by # of points in lag
#fitmethod7: weight by # of points in lag and give smaller lags more weight
source('/home/jared.oyler/ecl_juno_workspace/wxtopo/wxTopo_R/vario_fit.R')

load("/projects/daymet2/station_data/infill/infill_fnl/stns_tmax.RData")
#arcsine, sqrt transform vcf percentages
#STNS$vcf <- asin(sqrt(STNS$vcf/100))
STNS$col <- STNS@coords[,1] #lon
STNS$row <- STNS@coords[,2] #lat

stn_mask <- stns$neon == LCC_NUM
avg_radius <- sqrt(MAX_NGH_STNS*(LCC_AREAS[LCC_NUM]/sum(stn_mask))*(1.0/pi))
#CUTOFF = round((avg_radius*1.4)/WIDTH)*WIDTH

#polys<-readOGR("/projects/daymet2/dem/interp_grids/montana_interp_bbox","AOI_Polygon",verbose=FALSE)
#polys<-readOGR("/projects/daymet2/dem/NEON_DOMAINS","neon_mask3",verbose=FALSE)
#polys<-readOGR("/projects/daymet2/dem/interp_grids/conus/tifs","lcc",verbose=FALSE)
#polys<-readOGR("/projects/daymet2/dem/interp_grids/conus/tifs","fwpusgs_lcc_masked",verbose=FALSE)

#polys<-spTransform(polys,CRS(ALBERS_PROJ4))
#stns_aea <- spTransform(STNS,CRS(ALBERS_PROJ4))

#If needed, subset to polygon of interest
#polys <- polys[polys$grid_code==LCC_NUM,]
#polys <- polys[polys$GRIDCODE==LCC_NUM,]
#plot(polys)
#
#polys.area <- 0
#for (x in seq(length(polys@polygons)))
#{
#	polys.area <- polys.area + polys@polygons[[x]]@area * 1.0e-6 #km2
#}
#print(polys.area)


#imask <- as.vector(gIntersects(stns_aea,polys,byid=TRUE))
#avg_radius <- sqrt(MAX_NGH_STNS*(polys.area/sum(imask))*(1.0/pi))
#CUTOFF = round((avg_radius*1.4)/WIDTH)*WIDTH
#
#polys.buf <- polys
##polys.buf <- gBuffer(polys,width=CUTOFF*1000,byid=FALSE)
#polys.buf <- gBuffer(polys,width=1,byid=FALSE)
#imask <- as.vector(gIntersects(stns_aea,polys.buf,byid=TRUE))
#stns_buf <- STNS[imask,]
#plot(stns_buf)


#Analyze residuals of OLS fit
#a.lm <- lm(tair~lon+lat+elev+tdi+lst*vcf,data=stns_buf)
#ols.resid <- residuals(a.lm)

#Fit variogram model to OLS residuals
avar <- variogram(tair~lon+lat+elev+tdi+lst,stns_buf,cutoff=CUTOFF,width=WIDTH)#avar$gamma[1]
avar.model <- my.autofit.gwvario(avar,model=c("Exp"),fix.values = c(NA,NA,NA),start_vals = c(NA, NA,NA),fit.method=FITMETHOD)$var_model
plot(avar,model=avar.model)
print(avar.model)

#Fit variogram model to GLS residuals
g <- gstat(NULL,"val",tair~lon+lat+elev+tdi+lst,data=stns_buf,model=avar.model)
gls_df <- predict(g,stns_buf,BLUE=TRUE)
#residual = obs - predicted
stns_buf$resid <- stns_buf$tair - gls_df$val.pred
avar.gls <- variogram(resid~1,stns_buf,cutoff=CUTOFF,width=WIDTH)#avar.gls$gamma[1]
avar.model.gls <- my.autofit.gwvario(avar.gls,model=c("Exp"),fix.values = c(NA,NA,NA),fit.method=FITMETHOD)$var_model
plot(avar.gls,model=avar.model.gls)
print(avar.model.gls)

