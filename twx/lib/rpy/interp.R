# Functions for kriging mean Tair
# 
# Author: jared.oyler
###############################################################################

options(warn = -1)
library(gstat)
#library(mgcv)

gwr_meantair <- function(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt)
{
	#Build dataframes
	stns_ngh <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,tair=ngh_tair,ngh_wgt=ngh_wgt)
	pt <- data.frame(lon=pt[1],lat=pt[2],elev=pt[3],tdi=pt[4],lst=pt[5],neon=pt[6])
	
	#GWR
	#a.lm <- lm(FORMULA,stns_ngh,weights=ngh_wgt)
	#Changed to perform a regular linear regression
	a.lm <- lm(FORMULA,stns_ngh)
	tair_mean <- as.numeric(predict.lm(a.lm,pt))
	
	#tair.gam <- gam(FORMULA,data=stns_ngh)
	#tair.gam <- gam(tair~lst,data=stns_ngh)
	#tair_mean <- as.numeric(predict(tair.gam,pt))
	
	return(c(tair_mean,0))
}

gwr_anomaly <- function(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_wgt,obs_matrix,pt)
{
	#Build dataframes
	stns_ngh <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,ngh_wgt=ngh_wgt)
	pt <- data.frame(lon=pt[1],lat=pt[2],elev=pt[3],tdi=pt[4],lst=pt[5],neon=pt[6])
	
	pt_anom <- rep(0,nrow(obs_matrix))
	pt_r2 <- rep(0,nrow(obs_matrix))
	
	for (x in seq(nrow(obs_matrix)))
	{
		stns_ngh$anom <- obs_matrix[x,]
		
		#pt_anom[x] <- weighted.mean(stns_ngh$anom,stns_ngh$ngh_wgt)
		
		a.lm <- lm(anom~lon+lat+elev+tdi+lst,stns_ngh,weights=ngh_wgt)
		pt_anom[x] <- predict.lm(a.lm,pt)
		pt_r2[x] <- summary(a.lm)$r.squared
	}

	print(mean(pt_r2))
	
	return(pt_anom)
}

get_vario_params <- function(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,ngh_dist)
{	
	#Build dataframes
	stns_ngh <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,tair=ngh_tair)
	
	#Turn dataframes into spatial dataframes
	coordinates(stns_ngh) <- ~lon+lat
	proj4string(stns_ngh)=CRS("+proj=longlat +datum=WGS84")
	
	cutoff <- max(ngh_dist)*1.4
	avar <- variogram(FORMULA,stns_ngh,cutoff=cutoff,width=5)

	sill <- var(residuals(lm(FORMULA,data=stns_ngh)))
#	print(plot(avar))
#	Sys.sleep(15)
	#print(c(min(avar$gamma),sill))
	#avar.model <- tryCatch(my.autofit.gwvario(avar,model=c("Exp"),fix.values = c(min(avar$gamma),NA,sill),fit.method=7)$var_model,error=function(e) e)
	avar.model <- tryCatch(my.autofit.gwvario(avar,model=c("Exp"),fix.values = c(NA,NA,NA),fit.method=7)$var_model,error=function(e) e)
	
	if (is(avar.model,"simpleError"))
	{
		avar.model <- vgm(sill,"Nug")
	}
	else if (avar.model$psill[1] == 0 & avar.model$psill[2] == 0)
	{
		avar.model <- vgm(sill,"Nug")
	}
		
	g <- gstat(NULL,"val",FORMULA,data=stns_ngh,model=avar.model)
	gls_df <- predict(g,stns_ngh,BLUE=TRUE,debug.level = 0)
	stns_ngh$resid <- stns_ngh$tair - gls_df$val.pred
	sill <- var(stns_ngh$resid)
	avar.gls <- variogram(resid~1,stns_ngh,cutoff=cutoff,width=5)
	#avar.model.gls <- tryCatch(my.autofit.gwvario(avar.gls,model=c("Exp"),fix.values = c(min(avar.gls$gamma),NA,sill),fit.method=7)$var_model,error=function(e) e)
	avar.model.gls <- tryCatch(my.autofit.gwvario(avar.gls,model=c("Exp"),fix.values = c(NA,NA,NA),fit.method=7)$var_model,error=function(e) e)
	if (is(avar.model.gls,"simpleError"))
	{
		avar.model.gls <- vgm(sill,"Nug")
	}
	else if (avar.model.gls$psill[1] == 0 & avar.model.gls$psill[2] == 0)
	{
		avar.model.gls <- vgm(sill,"Nug")
	}
	
#	print(avar.model.gls)
#	print(plot(avar.gls,model=avar.model.gls))
#	Sys.sleep(3)

	if (length(avar.model.gls$model) == 1)
	{
		#This is a pure nugget
		#nug,psill,range
		return(c(avar.model.gls$psill[1],0,0))
	}
	else
	{
		#nug,psill,range
		return(c(avar.model.gls$psill[1],avar.model.gls$psill[2],avar.model.gls$range[2]))
	}
}

#Builds variogram params and performs actual kriging all in one step
krig_all <- function(pt,ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,ngh_dist)
{
	vario_params = get_vario_params(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,ngh_dist)
	nug <- vario_params[1]
	psill <- vario_params[2]
	range <- vario_params[3]
	
	#rslts = c(tair_mean,tair_var,0)
	rslts <- krig_meantair(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt,nug,psill,range)
	
	return(rslts)
}

krig_meantair <- function(ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt,nug,psill,range)
{	
	#Build dataframes
	stns_ngh <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,tair=ngh_tair,ngh_wgt=ngh_wgt)
	pt <- data.frame(lon=pt[1],lat=pt[2],elev=pt[3],tdi=pt[4],lst=pt[5])
	
	#a.lm <- lm(FORMULA,stns_ngh)#,weights=ngh_wgt)
	#print(a.lm)
#	print(as.numeric(predict.lm(a.lm,pt)))
	
	#tair_mean <- as.numeric(predict.lm(a.lm,pt))
#	tair_var <- 1.0
#	return(c(tair_mean,tair_var,0))
	#
	
	#arcsine, sqrt transform vcf percentages
#	stns_ngh$vcf <- asin(sqrt(stns_ngh$vcf/100))
#	pt$vcf <- asin(sqrt(pt$vcf/100))
	
	#Turn dataframes into spatial dataframes
	coordinates(stns_ngh) <- ~lon+lat
	proj4string(stns_ngh)=CRS("+proj=longlat +datum=WGS84")
	coordinates(pt) <- ~lon+lat
	proj4string(pt)=CRS("+proj=longlat +datum=WGS84")
	
	if (range == 0)
	{
		#no range, so make pure nugget
		var_model <- vgm(psill+nug,"Nug")
	}
	else
	{
		var_model <- vgm(model="Exp",nugget=nug,psill=psill,range=range)
	}
	
	
	
	#save(list=c("stns_ngh","pt","var_model","FORMULA"),file='/projects/daymet2/rdata/RKtest.Rdata')
	
	#Regression kriging
	####################################################
	#Model GLS trend
#	g <- gstat(NULL,"val",FORMULA,data=stns_ngh,model=var_model)
#	gls_stns <- predict(g,stns_ngh,BLUE=TRUE,debug.level = 0)
#	gls_pt <- predict(g,pt,BLUE=TRUE,debug.level = 0)
#	
#	#residual = obs - predicted
#	stns_ngh$resid <- stns_ngh$tair - gls_stns$val.pred
#	
#	#OK of residual
#	krig.resid <- krige(resid~1,stns_ngh,newdata=pt,model=var_model,debug.level = 0)
#	
#	#Combine trend and residual
#	tair_mean <- gls_pt$val.pred + krig.resid$var1.pred
	####################################################
	
	#Kriging with External Drift to get prediction variance
	####################################################
	krig.rslts <- krige(FORMULA,stns_ngh,newdata=pt,model=var_model,debug.level = 0)
	tair_mean <- krig.rslts$var1.pred
	tair_var <- krig.rslts$var1.var
#	krig.rslts <- krige0(FORMULA,stns_ngh,newdata=pt,model=var_model,debug.level = 0,computeVar = TRUE)
#	tair_mean <- krig.rslts$pred
#	tair_var <- krig.rslts$var
	
	#print(c(tair_mean,tair_mean2,tair_var,tair_var2))
	####################################################

	#print(c(tair_mean,tair_mean2))
	
	return(c(tair_mean,tair_var,0))
	
}

build_formula <- function(y_var,x_vars)
{
	return(as.formula(paste(paste(y_var,"~"),paste(x_vars,collapse="+"))))
}

save_stn_spatial_df <- function(stn_id,lon,lat,elev,tdi,lst,neon,tair,imp_flag,fpath_out)
{
	STNS <- data.frame(stn_id=stn_id,lon=lon,lat=lat,elev=elev,tdi=tdi,lst=lst,neon=neon,tair=tair,imp_flag=imp_flag)
	coordinates(STNS) <- ~lon+lat
	proj4string(STNS)=CRS("+proj=longlat +datum=WGS84")
	save(list=c("STNS"),file=fpath_out)
}

my.autofit.gwvario <- function(experimental_variogram, model = c("Sph", "Exp", "Gau", "Ste"),
		kappa = c(0.05, seq(0.2, 2, 0.1), 5, 10), fix.values = c(NA,NA,NA),fit.method=7,
		verbose = FALSE, GLS.model = NA, start_vals = c(NA,NA,NA),boundaries = NA, 
		miscFitOptions = list(),...)
# This function automatically fits a variogram to input_data
{
	# Check for anisotropy parameters
	if('alpha' %in% names(list(...))) warning('Anisotropic variogram model fitting not supported, see the documentation of autofitVariogram for more details.')
	
	# Take the misc fit options and overwrite the defaults by the user specified ones
	miscFitOptionsDefaults = list(merge.small.bins = TRUE, min.np.bin = 5)
	miscFitOptions = modifyList(miscFitOptionsDefaults, miscFitOptions)
	
	# set initial values
	if(is.na(start_vals[1])) {  # Nugget
		initial_nugget = min(experimental_variogram$gamma)
	} else {
		initial_nugget = start_vals[1]
	}
	if(is.na(start_vals[2])) { # Range
		initial_range = 0.1 * max(experimental_variogram$dist)   # 0.10 times the length of the central axis through the area
	} else {
		initial_range = start_vals[2]
	}
	if(is.na(start_vals[3])) { # Sill
		initial_sill = mean(c(max(experimental_variogram$gamma), median(experimental_variogram$gamma)))
	} else {
		initial_sill = start_vals[3]
	}
	
	# Determine what should be automatically fitted and what should be fixed
	# Nugget
	if(!is.na(fix.values[1]))
	{
		fit_nugget = FALSE
		initial_nugget = fix.values[1]
	} else
		fit_nugget = TRUE
	
	# Range
	if(!is.na(fix.values[2]))
	{
		fit_range = FALSE
		initial_range = fix.values[2]
	} else
		fit_range = TRUE
	
	# Partial sill
	if(!is.na(fix.values[3]))
	{
		fit_sill = FALSE
		initial_sill = fix.values[3]
	} else
		fit_sill = TRUE
	
	getModel = function(psill, model, range, kappa, nugget, fit_range, fit_sill, fit_nugget, verbose,fit.method)
	{
		if(verbose) debug.level = 1 else debug.level = 0
		if(model == "Pow") {
			warning("Using the power model is at your own risk, read the docs of autofitVariogram for more details.")
			if(is.na(start_vals[1])) nugget = 0
			if(is.na(start_vals[2])) range = 1    # If a power mode, range == 1 is a better start value
			if(is.na(start_vals[3])) sill = 1
		}
		
		obj = try(fit.variogram(experimental_variogram,
						model = vgm(psill=psill, model=model, range=range,
								nugget=nugget,kappa = kappa),
						fit.ranges = c(fit_range), fit.sills = c(fit_nugget, fit_sill),
						debug.level = debug.level,fit.method=fit.method), 
				TRUE)
		if("try-error" %in% class(obj)) {
			#print(traceback())
			warning("An error has occured during variogram fitting. Used:\n", 
					"\tnugget:\t", nugget, 
					"\n\tmodel:\t", model, 
					"\n\tpsill:\t", psill,
					"\n\trange:\t", range,
					"\n\tkappa:\t",ifelse(kappa == 0, NA, kappa),
					"\n  as initial guess. This particular variogram fit is not taken into account. \nGstat error:\n", obj)
			return(NULL)
		} else return(obj)
	}
	
	
	# Automatically testing different models, the one with the smallest sums-of-squares is chosen
	test_models = model
	SSerr_list = c()
	vgm_list = list()
	counter = 1
	
	for(m in test_models) {
		if(m != "Mat" && m != "Ste") {        # If not Matern and not Stein
			model_fit = getModel(initial_sill - initial_nugget, m, initial_range, kappa = 0, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose,fit.method=fit.method)
			if(!is.null(model_fit)) {	# skip models that failed
				vgm_list[[counter]] = model_fit
				SSerr_list = c(SSerr_list, attr(model_fit, "SSErr"))}
			counter = counter + 1
		} else {                 # Else loop also over kappa values
			for(k in kappa) {
				model_fit = getModel(initial_sill - initial_nugget, m, initial_range, k, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose,fit.method=fit.method)
				#print(plot(experimental_variogram,model=model_fit))
				if(!is.null(model_fit)) {
					vgm_list[[counter]] = model_fit
					SSerr_list = c(SSerr_list, attr(model_fit, "SSErr"))}
				counter = counter + 1
			}
		}
	}
	
	# Check for negative values in sill or range coming from fit.variogram
	# and NULL values in vgm_list, and remove those with a warning
	strange_entries = sapply(vgm_list, function(v) any(c(v$psill, v$range) < 0) | is.null(v))
	if(any(strange_entries)) {
		if(verbose) {
			print(vgm_list[strange_entries])
			cat("^^^ ABOVE MODELS WERE REMOVED ^^^\n\n")
		}
		warning("Some models where removed for being either NULL or having a negative sill/range/nugget, \n\tset verbose == TRUE for more information")
		SSerr_list = SSerr_list[!strange_entries]
		vgm_list = vgm_list[!strange_entries]
	}
	
	if(verbose) {
		cat("Selected:\n")
		print(vgm_list[[which.min(SSerr_list)]])
		cat("\nTested models, best first:\n")
		tested = data.frame("Tested models" = sapply(vgm_list, function(x) as.character(x[2,1])), 
				kappa = sapply(vgm_list, function(x) as.character(x[2,4])), 
				"SSerror" = SSerr_list)
		tested = tested[order(tested$SSerror),]
		print(tested)
	}
	
	result = list(exp_var = experimental_variogram, var_model = vgm_list[[which.min(SSerr_list)]], sserr = min(SSerr_list))
	class(result) = c("autofitVariogram","list")    
	
	return(result)
}
