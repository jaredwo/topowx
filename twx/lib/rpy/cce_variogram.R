# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

AREA_CCE_BUFFER <- 69786.9 #km2
WGS84 = CRS("+proj=longlat +datum=WGS84")
ALBERS_PROJ4 <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

library(gstat)
library(rgdal)
library(rgeos)
library(mgcv)

load('/projects/daymet2/rdata/CCE_STNS_TMAX.R')
FORMULA <- build_formula("tair",c("lon","lat","elev"))#,"tdi"))#"lst"))
FORMULA <- build_formula("tair",c("lon","lat","elev","tdi"))#"lst"))
FORMULA <- build_formula("tair",c("lon","lat","elev","tdi","lst"))#"lst"))

cce_buffer<-readOGR("/projects/daymet2/dem/cce/CCE_StudyArea","cce_buffer",verbose=FALSE)
cce_buffer<-spTransform(cce_buffer,CRS(ALBERS_PROJ4))
stns_aea <- spTransform(STNS,CRS(ALBERS_PROJ4))
#STNS@proj4string@projargs <- cce_buffer@proj4string@projargs
imask <- as.vector(gIntersects(stns_aea,cce_buffer,byid=TRUE))
stns_cce = STNS[imask,]

tair.gam <- gam(tair~s(lon)+s(lat)+s(elev)+s(tdi)+s(lst),data=stns_cce)
tair.lm <- lm(FORMULA,data=stns_cce)
plot(tair.gam,residuals=TRUE,se=TRUE,ask=FALSE,cex.lab=1.5)
resid.gam <- residuals(tair.gam)
stns_cce$resid <- resid.gam
avar <- variogram(resid~1,stns_cce,cutoff=CUTOFF,width=WIDTH)

WIDTH <- 5
FITMETHOD <-7
#fitmethod2: weight by # of points in lag
#fitmethod7: weight by # of points in lag and give smaller lags more weight

avg_radius = sqrt(110*(AREA_CCE_BUFFER/nrow(stns_cce))*(1.0/pi))
CUTOFF = round((avg_radius*1.4)/WIDTH)*WIDTH

cce_db <- gBuffer(cce_buffer,width=CUTOFF*1000,byid=FALSE)
imask <- as.vector(gIntersects(stns_aea,cce_db,byid=TRUE))
stns_buf <- STNS[imask,]

avar <- variogram(FORMULA,stns_cce,cutoff=CUTOFF,width=WIDTH)
avar.model <- my.autofit.gwvario(avar,model=c("Exp"),fit.method=FITMETHOD)$var_model
plot(avar,model=avar.model)
print(avar.model)


build_formula <- function(y_var,x_vars)
{
	return(as.formula(paste(paste(y_var,"~"),paste(x_vars,collapse="+"))))
}

my.autofit.gwvario <- function(experimental_variogram, model = c("Sph", "Exp", "Gau", "Ste"),
		kappa = c(0.05, seq(0.2, 2, 0.1), 5, 10), fix.values = c(NA,NA,NA),fit.method=7,
		verbose = FALSE, GLS.model = NA, start_vals = c(NA,NA,NA),boundaries = NA, 
		miscFitOptions = list(),...)
# This function automatica10.2838lly fits a variogram to input_data
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


