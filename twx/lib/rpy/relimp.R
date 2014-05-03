# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

library(relaimpo)


calc_rel_impo <- function(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair)
{
	stns <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,norm=ngh_tair)
	
	mthLm <- lm(norm~lon+lat+elev+lst,data=stns)
	rel.lm <- calc.relimp(mthLm,rela=TRUE)
	
	lmgLon <- rel.lm@lmg['lon']
	lmgLat <- rel.lm@lmg['lat']
	lmgElev <- rel.lm@lmg['elev']
	lmgLst <- rel.lm@lmg['lst']
	r2 <- rel.lm@R2
	
	elevLm <- lm(norm~elev,data=stns)
	elevR2 <- summary(elevLm)$r.squared
	
	lstLm <- lm(norm~lst,data=stns)
	lstR2 <- summary(lstLm)$r.squared
	
	elevlstLm <- lm(elev~lst,data=stns)
	elevlstR2 <- summary(elevlstLm)$r.squared
	
	return(c(lmgLon,lmgLat,lmgElev,lmgLst,r2,elevR2,lstR2,elevlstR2))
	
}


#stnsTmax <- read.csv('/projects/daymet2/docs/final_writeup/cce_stns_tmax.csv')
#stnsTmin <- read.csv('/projects/daymet2/docs/final_writeup/cce_stns_tmin.csv')
#
#stns <- stnsTmax
#
#lmgLon <- c()
#lmgLat <- c()
#lmgElev <- c()
#lmgLst <- c()
#r2 <- c()
#r2.lm <- c()
#
#for (mth in 1:12)
#{
#	normVar <- paste("norm",formatC(mth,width=2,format="d",flag="0"),sep="")
#	lstVar <- paste("lst",formatC(mth,width=2,format="d",flag="0"),sep="")
#	stnsMth <- data.frame(elev=stns$ELEV,lon=stns$LON,lat=stns$LAT,norm=stns[,c(normVar)],lst=stns[,c(lstVar)])
#	mthLm <- lm(norm~lon+lat+elev+lst,data=stnsMth)
#	rel.lm <- calc.relimp(mthLm,rela=TRUE)
#	
#	lmgLon[mth] <- rel.lm@lmg['lon']
#	lmgLat[mth] <- rel.lm@lmg['lat']
#	lmgElev[mth] <- rel.lm@lmg['elev']
#	lmgLst[mth] <- rel.lm@lmg['lst']
#	r2[mth] <- rel.lm@R2
#	r2.lm[mth] <- summary(mthLm)$r.squared#adj.r.squared
#}
#
#lmg <- data.frame(lon=lmgLon,lat=lmgLat,elev=lmgElev,lst=lmgLst,r2=r2)
#write.table(lmg,'/projects/daymet2/docs/final_writeup/cce_relimpr2_tmax.csv',row.names=FALSE,sep=",",quote=FALSE)

