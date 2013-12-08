# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

#library(mgcv)
library(MASS)
library(boot)


MAX_VAR_SUM <- 0.95

#
#tair_gam <- function(stn_obs,stn_pts,lon,lat,elev)
#{
#	ndays <- nrow(stn_obs)
#	PT = data.frame(LON=lon,LAT=lat,ELEV=elev)
#	tair.vals<-seq(0,ndays)
#	
#	for (x in seq(ndays))
#	{
#		tair.gam <- gam(stn_obs[x,]~s(LON)+s(LAT)+s(ELEV),data=stn_pts)
#		tair.vals[x] <- predict(tair.gam,PT)
#		print(x)
#	}
#	return(tair.vals)
#}


boot_lm_tair <- function(stn_data,indices)
{
	stn_data <- stn_data[indices,]
	lm.mod <-lm(TAIR~LON+LAT+ELEV,data=stn_data,weights=WGT/sum(WGT))
	return(predict(lm.mod,PT))
}

#apt = data.frame(LON=lon,LAT=lat,ELEV=elev)

boot_tair <- function(stn_pts,apt)
{
	PT <<- apt
	my.boot <- boot(stn_pts,boot_lm_tair,1000)
	se <- sqrt(var(my.boot$t))
	ci <- boot.ci(boot.out = my.boot, type = "perc")$percent
	return(list(value=my.boot$t0,se=se,ci_l=ci[4],ci_u=ci[5]))
}

pca_tair <- function(stn_pts,stn_obs,pt)
{		
	#data already centered
	a.pca <- prcomp(stn_obs,center=F)
	vars <- a.pca$sdev^2
	vars <- vars/sum(vars)
	vars.cum <- cumsum(vars)
	npc <- which(vars.cum >= MAX_VAR_SUM)[1]
	#print(npc)
	loads <- c()
	for (x in seq(npc))
	{
		lm_ld <- lm(a.pca$rotation[,x]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
		loads[x] <- predict(lm_ld,pt)
	}
	
	if(npc > 1)
	{
		tair <- a.pca$x[,1:npc]%*%loads
	}
	else
	{
		tair <- a.pca$x[,1:npc]*loads
	}
	
	if (! is.null(dim(tair)))
	{
		tair <- tair[,1]
	}
	
	
	#Add back mean center and return
	return(tair+pt$TAIR)
}

pca_po <- function(stn_pts,stn_obs,pt)
{		
	#do not center
	a.pca <- prcomp(stn_obs,center=F)
	vars <- a.pca$sdev^2
	vars <- vars/sum(vars)
	vars.cum <- cumsum(vars)
	npc <- which(vars.cum >= MAX_VAR_SUM)[1]
	loads <- c()
	for (x in seq(npc))
	{
		lm_ld <- lm(a.pca$rotation[,x]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
		loads[x] <- predict(lm_ld,pt)
	}
	
	if(npc > 1)
	{
		po <- a.pca$x[,1:npc]%*%loads
	}
	else
	{
		po <- a.pca$x[,1:npc]*loads
	}
	
	if (! is.null(dim(po)))
	{
		po <- po[,1]
	}
	
	return(po)
}

interp_po <- function(stn_pts,stn_obs,lon,lat,elev)
{
	pt <- data.frame(LON=lon,LAT=lat,ELEV=elev)
	po <- pca_po(stn_pts,stn_obs,pt)
	
	return(po)
}

#interp_tair <- function(stn_pts,stn_obs,lon,lat,elev,boot)
#{
#	pt <- data.frame(LON=lon,LAT=lat,ELEV=elev)
#	
#	save(stn_pts,stn_obs,pt,file='/projects/daymet2/rdata/kalispell.Rdata')
#	
#	if (boot)
#	{
#		result <- boot_tair(stn_pts,pt)
#	}
#	else
#	{
#		result = list(se=0,ci_l=0,ci_u=0)
#		lm.mod <-lm(TAIR~LON+LAT+ELEV,data=stn_pts,weights=WGT/sum(WGT))
#		result$value <- predict(lm.mod,pt)
#	}
#	
#	pt$TAIR <- result$value
#	result$interp_vals <- pca_tair(stn_pts,stn_obs,pt)
#	
#	return(result)
#}

interp_tair <- function(stn_pts,stn_obs,lon,lat,elev,boot)
{
	pt <- data.frame(LON=lon,LAT=lat,ELEV=elev)
	
	save(stn_pts,stn_obs,pt,file='/projects/daymet2/rdata/bob.Rdata')
	sign_init<-1
	stns_dif<-NULL
	for (x in seq(length(stn_pts$LON)-1))
	{
		sign<-sign_init
		i_start = x+1
		for (i in seq(i_start,length(stn_pts$LON)))
		{
			dif_x<-(sign * (stn_pts$LON[x] - stn_pts$LON[i]))
			dif_y<-(sign * (stn_pts$LAT[x] - stn_pts$LAT[i]))
			dif_z<-(sign * (stn_pts$ELEV[x] - stn_pts$ELEV[i]))
			dif_val<-(sign * (stn_pts$TAIR[x] - stn_pts$TAIR[i]))
			wght<-(stn_pts$WGT[x]*stn_pts$WGT[i])
			
			row = data.frame(dif_x=dif_x,dif_y=dif_y,dif_z=dif_z,dif_val=dif_val,wght=wght)
			
			if (is.null(stns_dif))
			{
				stns_dif<-row
			}
			else
			{
				stns_dif<-rbind(stns_dif,row)	
			}
			
			sign<--sign
		}
	}
	
	lm.model<-lm(dif_val~dif_x+dif_y+dif_z,data=stns_dif,weights=wght/sum(wght))
	stn_pt_difs <- data.frame(dif_x=pt$LON - stn_pts$LON,dif_y=pt$LAT - stn_pts$LAT, dif_z=pt$ELEV - stn_pts$ELEV)
	lm.pred<-predict(lm.model,stn_pt_difs,interval=c("prediction"))
	result <- list(se=0,ci_l=weighted.mean(stn_pts$TAIR+lm.pred[,2],w=stn_pts$WGT),ci_u=weighted.mean(stn_pts$TAIR+lm.pred[,3],w=stn_pts$WGT))
	result$value <-  weighted.mean(stn_pts$TAIR+lm.pred[,1],w=stn_pts$WGT)
	
	pt$TAIR <- result$value
	result$interp_vals <- pca_tair(stn_pts,stn_obs,pt)
	
	return(result)
}

#load("/projects/daymet2/rdata/flattop_tmin.Rdata")
#stn_pts$TAIR <- colMeans(stn_obs)
#ainterp <- interp_tair(stn_pts,sweep(stn_obs,2,stn_pts$TAIR),pt$LON,pt$LAT,pt$ELEV)

#prcp_pca <- function(stn_obs,stn_pts,lon,lat,elev,years,months,uniq_years,uniq_months,pop_crit)
#{
#	
#	##########################
#	#START: PREDICT MTHLY PRCP
#	##########################
#	prcp_mthly<-aggregate(stn_obs,by=list(years=years,months=months),FUN=sum)
#	prcp_mthly<-prcp_mthly[order(prcp_mthly$years,prcp_mthly$months),]
#	prcp_mthly$years<-NULL
#	prcp_mthly$months<-NULL
#	prcp_mthly<-as.matrix(prcp_mthly)
#	
#	pca<-prcomp(prcp_mthly,center=F)
#	
#	lm.pc1 <- lm(pca$rotation[,1]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc2 <- lm(pca$rotation[,2]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc3 <- lm(pca$rotation[,3]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc4 <- lm(pca$rotation[,4]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc5 <- lm(pca$rotation[,5]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	
#	loads <- c(predict(lm.pc1,pt),predict(lm.pc2,pt),predict(lm.pc3,pt),predict(lm.pc4,pt),predict(lm.pc5,pt))
#	prcp_mthly <- pca$x[,1:5]%*%loads
#	##########################
#	#END: PREDICT MTHLY PRCP
#	##########################
#	
#	##########################
#	#START: PREDICT DAILY PRCP OCCURANCE
#	##########################
#	po <- stn_obs
#	pca_po<-prcomp(po,center=F)
#	
#	lm.pc1 <- lm(pca_po$rotation[,1]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc2 <- lm(pca_po$rotation[,2]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc3 <- lm(pca_po$rotation[,3]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc4 <- lm(pca_po$rotation[,4]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc5 <- lm(pca_po$rotation[,5]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	loads <- c(predict(lm.pc1,pt),predict(lm.pc2,pt),predict(lm.pc3,pt),predict(lm.pc4,pt),predict(lm.pc5,pt))
#	prcp_occ <- pca_po$x[,1:5]%*%loads
#	prcp_occ[prcp_occ > pop_crit] <- 1
#	prcp_occ[prcp_occ <= pop_crit] <- 0
#	dim(prcp_occ)<-c(length(prcp_occ))
#	##########################
#	#END: PREDICT DAILY PRCP OCCURANCE
#	##########################
#	
#	##########################
#	#START: PREDICT DAILY PRCP AMT
#	##########################
#	pca_daily<-prcomp(stn_obs,center=F)
#	
#	lm.pc1 <- lm(pca_daily$rotation[,1]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc2 <- lm(pca_daily$rotation[,2]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc3 <- lm(pca_daily$rotation[,3]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc4 <- lm(pca_daily$rotation[,4]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	lm.pc5 <- lm(pca_daily$rotation[,5]~LON+LAT+ELEV,data=stn_pts,weights=WGT)
#	loads <- c(predict(lm.pc1,pt),predict(lm.pc2,pt),predict(lm.pc3,pt),predict(lm.pc4,pt),predict(lm.pc5,pt))
#	prcp_amt <- pca_daily$x[,1:5]%*%loads
#	##########################
#	#END: PREDICT DAILY PRCP AMT
#	##########################
#	
#	##########################
#	#START: MODIFY DAILY
#	##########################
#	prcp_amt[prcp_occ==0] <- 0 
#	
#	x <- 1
#	for (year in uniq_years)
#	{
#		for (mth in uniq_months)
#		{
#			mask <- years==year & months == mth
#			ratio <- prcp_mthly[x]/sum(prcp_amt[mask])
#			if (is.infinite(ratio))
#			{
#				ratio <- 0				
#			}
#			prcp_amt[mask] <- prcp_amt[mask]*ratio
#			x <- x+1
#		}
#	}
#	
#	##########################
#	#END: MODIFY DAILY
#	##########################
#	return(prcp_amt)
#}
#
#tair_pca_difs <- function(stn_obs,stn_pts,lon,lat,elev,td,sign_init)
#{
#	pca <- prcomp(stn_obs,center=F)
#	n_pc <- 5
#	
#	stns_dif<-NULL
#	for (x in seq(length(stn_pts$WGT)-1))
#	{
#		sign<-sign_init
#		i_start = x+1
#		for (i in seq(i_start,length(stn_pts$WGT)))
#		{
#			dif_x<-(sign * (stn_pts$LON[x] - stn_pts$LON[i]))
#			dif_y<-(sign * (stn_pts$LAT[x] - stn_pts$LAT[i]))
#			dif_z<-(sign * (stn_pts$ELEV[x] - stn_pts$ELEV[i]))
#			dif_td<-(sign * (stn_pts$TD[x] - stn_pts$TD[i]))
#			
#			dif_pc1<-(sign * (pca$rotation[,1][x] - pca$rotation[,1][i]))
#			dif_pc2<-(sign * (pca$rotation[,2][x] - pca$rotation[,2][i]))
#			dif_pc3<-(sign * (pca$rotation[,3][x] - pca$rotation[,3][i]))
#			dif_pc4<-(sign * (pca$rotation[,4][x] - pca$rotation[,4][i]))
#			dif_pc5<-(sign * (pca$rotation[,5][x] - pca$rotation[,5][i]))
#			
#			wght<-(stn_pts$WGT[x]*stn_pts$WGT[i])
#			
#			row = data.frame(	dif_x=dif_x,dif_y=dif_y,dif_z=dif_z,dif_td=dif_td,
#								dif_pc1=dif_pc1,dif_pc2=dif_pc2,dif_pc3=dif_pc3,
#								dif_pc4=dif_pc4,dif_pc5=dif_pc5,wght=wght)
#			
#			if (is.null(stns_dif))
#			{
#				stns_dif<-row
#			}
#			else
#			{
#				stns_dif<-rbind(stns_dif,row)	
#			}
#			
#			sign<--sign
#		}
#	}
#	
#	lm_pc1 <- lm(dif_pc1~dif_x+dif_y+dif_z+dif_td,data=stns_dif,weights=wght)
#	lm_pc2 <- lm(dif_pc2~dif_x+dif_y+dif_z+dif_td,data=stns_dif,weights=wght)
#	lm_pc3 <- lm(dif_pc3~dif_x+dif_y+dif_z+dif_td,data=stns_dif,weights=wght)
#	lm_pc4 <- lm(dif_pc4~dif_x+dif_y+dif_z+dif_td,data=stns_dif,weights=wght)
#	lm_pc5 <- lm(dif_pc5~dif_x+dif_y+dif_z+dif_td,data=stns_dif,weights=wght)
#	
#	pc1_sum = 0
#	pc2_sum = 0
#	pc3_sum = 0
#	pc4_sum = 0
#	pc5_sum = 0
#	
#	pc1_wgt_sum = 0
#	pc2_wgt_sum = 0
#	pc3_wgt_sum = 0
#	pc4_wgt_sum = 0
#	pc5_wgt_sum = 0
#	
#	for (x in seq(length(stn_pts$WGT)))
#	{	
#		dif_df <- data.frame(dif_x=lon-stn_pts$LON[x],dif_y=lat-stn_pts$LAT[x],
#				dif_z=elev-stn_pts$ELEV[x],dif_td=td-stn_pts$TD[x])
#		
#		pc1_sum = pc1_sum + (stn_pts$WGT[x]*(pca$rotation[,1][x] + predict(lm_pc1,dif_df)))
#		pc2_sum = pc2_sum + (stn_pts$WGT[x]*(pca$rotation[,2][x] + predict(lm_pc2,dif_df)))
#		pc3_sum = pc3_sum + (stn_pts$WGT[x]*(pca$rotation[,3][x] + predict(lm_pc3,dif_df)))
#		pc4_sum = pc4_sum + (stn_pts$WGT[x]*(pca$rotation[,4][x] + predict(lm_pc4,dif_df)))
#		pc5_sum = pc5_sum + (stn_pts$WGT[x]*(pca$rotation[,5][x] + predict(lm_pc5,dif_df)))
#		
#		pc1_wgt_sum = pc1_wgt_sum + stn_pts$WGT[x]
#		pc2_wgt_sum = pc2_wgt_sum + stn_pts$WGT[x]
#		pc3_wgt_sum = pc3_wgt_sum + stn_pts$WGT[x]
#		pc4_wgt_sum = pc4_wgt_sum + stn_pts$WGT[x]
#		pc5_wgt_sum = pc5_wgt_sum + stn_pts$WGT[x]
#	}
#	
#	loads <- c(pc1_sum/pc1_wgt_sum,
#			pc2_sum/pc2_wgt_sum,
#			pc3_sum/pc3_wgt_sum,
#			pc4_sum/pc4_wgt_sum,
#			pc5_sum/pc5_wgt_sum)
#	
#	tair <- pca$x[,1:n_pc]%*%loads
#	return(tair)
#}



