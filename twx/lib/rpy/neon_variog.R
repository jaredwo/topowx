# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

library(gstat)
library(automap)
library(geosphere)

MAX_DIST_SCALER <- 1.4
TREND_VARS <- c("lon","lat","elev","tdi","lst")

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

NEON_AREAS <- list(	"0"=AREA_NEON_0,
					"2"=AREA_NEON_2,
					"3"=AREA_NEON_3,
					"5"=AREA_NEON_5,
					"6"=AREA_NEON_6,
					"7"=AREA_NEON_7,
					"8"=AREA_NEON_8,
					"9"=AREA_NEON_9,
					"10"=AREA_NEON_10,
					"11"=AREA_NEON_11,
					"12"=AREA_NEON_12,
					"13"=AREA_NEON_13,
					"14"=AREA_NEON_14,
					"15"=AREA_NEON_15,
					"16"=AREA_NEON_16,
					"17"=AREA_NEON_17)

load('/projects/daymet2/rdata/all_stns_tmin.Rdata')
stns_tmin <- stns_tmin[!is.na(stns_tmin$tdi) & !is.na(stns_tmin$neon),]
coordinates(stns_tmin) <- ~lon+lat
proj4string(stns_tmin)=CRS("+proj=longlat +datum=WGS84")

co.vars <- c("lon","lat","elev","tdi","lst")
mean_tair_form_gstat <- as.formula(paste(paste("tair","~",sep=""),paste(co.vars,collapse="+")))

optim.varios <- list()
min_nstns <- 50
max_nstns <- 60
avg_nstns <- round(mean(c(min_nstns,max_nstns)),0)

for (neon in names(neon_areas))
{
	neon.area <- as.numeric(neon_areas[neon])
	neon <- as.numeric(neon) 
	stns_neon <- stns_tmin[stns_tmin$neon==neon,]
	avg_radius = sqrt(avg_nstns*(neon.area/nrow(stns_neon))*(1.0/pi))
	maxdist <- avg_radius*MAX_DIST_SCALER
	nbins <- round(maxdist/2)
	
	auto.var<-my.autofit(mean_tair_form_gstat,stns_neon,maxdist=maxdist,nbins=nbins,model=c("Ste"))
	#print(sum(auto.var$var_model$psill))
	optim.varios[[as.character(neon)]] <- auto.var
}

df.varios <- as.df.autofits(optim.varios)
write.csv(df.varios,"/projects/daymet2/station_data/infill/impute_tair/neon_varios.csv",row.names=FALSE,quote=FALSE):q
a<-read.csv("/projects/daymet2/station_data/infill/impute_tair/neon_varios.csv",header=TRUE)


neon.polys<-readOGR("/projects/daymet2/dem/NEON_DOMAINS","neon_mask3")
neon.polys<-spTransform(neon.polys,CRS(ALBERS_PROJ4))
neon.shp <- neon.poly[neon.poly$GRIDCODE==12,]
neon.shp <- spTransform(neon.shp,CRS(ALBERS_PROJ4))
neon.buf <- gBuffer(neon.shp,width=100*1000,byid=FALSE)

build.neon.varios <- function(nngh,stns,trend.vars=TREND_VARS,max.dist.scale=MAX_DIST_SCALER)
{
	mean_tair_form_gstat <- as.formula(paste("tair~",paste(trend.vars,collapse="+")))
	
	for (neon in names(NEON_AREAS))
	{
		neon.area <- as.numeric(NEON_AREAS[neon])
		neon <- as.numeric(neon) 
		stns_neon <- stns[stns$neon==neon,]
		avg_radius = sqrt(nngh*(neon.area/nrow(stns_neon))*(1.0/pi))
		
		neon.shp <- neon.polys[neon.polys$GRIDCODE==neon,]
		neon.buf <- gBuffer(neon.shp,width=avg_radius*1000,byid=FALSE)
		imask <- as.vector(gIntersects(stns,neon.buf,byid=TRUE))
		
		stns_buf <- stns[imask,]
		stns_buf@coords <- stns_buf@coords/1000 #rescale from m to km
		
#		avg_lon <- mean(stns_neon@coords[,1])
#		avg_lat <- mean(stns_neon@coords[,2])
#		
#		min_bb_lon <- destPoint(c(min(stns_neon@coords[,1]),avg_lat),270,avg_radius*1000)[1]
#		max_bb_lon <- destPoint(c(max(stns_neon@coords[,1]),avg_lat),90,avg_radius*1000)[1]
#		
#		min_bb_lat <- destPoint(c(avg_lon,min(stns_neon@coords[,2])),180,avg_radius*1000)[2]
#		max_bb_lat <- destPoint(c(avg_lon,max(stns_neon@coords[,2])),0,avg_radius*1000)[2]
#		
#		stn_box_mask = stns@coords[,1] >= min_bb_lon & stns@coords[,1] <= max_bb_lon & stns@coords[,2] >= min_bb_lat & stns@coords[,2] <= max_bb_lat
#		stns_neon_box <- stns[stn_box_mask,]
		
		maxdist <- avg_radius*max.dist.scale
		nbins <- round(maxdist/2)
		#
		auto.var<-my.autofit(mean_tair_form_gstat,stns_buf,maxdist=maxdist,nbins=nbins,model=c("Ste"),fit.method=7)
		optim.varios[[as.character(neon)]] <- auto.var
	}
	
	df.varios <- as.df.autofits(optim.varios)
	return(df.varios)
}

as.df.autofits <- function(optim.varios)
{
	
	neon_nums <- as.numeric(names(optim.varios))
	nugs <- c()
	psills <- c()
	rngs <- c()
	kappa <- c()
	
	for (x in seq(length(optim.varios)))
	{
		a.model <- optim.varios[[x]]$var_model
		nugs[x] <- a.model$psill[1]
		psills[x] <- a.model$psill[2]
		rngs[x] <- a.model$range[2]
		kappa[x] <- a.model$kappa[2]
	}
	
	return(data.frame(neon=neon_nums,nug=nugs,psill=psills,range=rngs,kappa=kappa))
}

my.autofit <- function(formula, input_data, model = c("Sph", "Exp", "Gau", "Ste"),
		kappa = c(0.05, seq(0.2, 2, 0.1), 5, 10), fix.values = c(NA,NA,NA),
		verbose = FALSE, GLS.model = NA, start_vals = c(NA,NA,NA),maxdist,nbins,fit.method, 
		miscFitOptions = list(),...)
# This function automatically fits a variogram to input_data
{
	# Check for anisotropy parameters
	if('alpha' %in% names(list(...))) warning('Anisotropic variogram model fitting not supported, see the documentation of autofitVariogram for more details.')
	
	# Take the misc fit options and overwrite the defaults by the user specified ones
	miscFitOptionsDefaults = list(merge.small.bins = TRUE, min.np.bin = 5)
	miscFitOptions = modifyList(miscFitOptionsDefaults, miscFitOptions)
	
	# Create boundaries
	longlat = !is.projected(input_data)
	if(is.na(longlat)) longlat = FALSE
	#diagonal = spDists(t(bbox(input_data)), longlat = longlat)[1,2]                # 0.35 times the length of the central axis through the area
	#boundaries = c(2,4,6,9,12,15,25,35,50,65,80,100) * maxdist * 0.35/100         # Boundaries for the bins in km
	boundaries <- seq(0,maxdist,maxdist/nbins)
	#print(boundaries)
	
	# If you specifiy a variogram model in GLS.model the Generelised least squares sample variogram is constructed
	if(!is(GLS.model, "variogramModel")) {
		experimental_variogram = variogram(formula, input_data,boundaries = boundaries, ...)
	} else {
		if(verbose) cat("Calculating GLS sample variogram\n")
		g = gstat(NULL, "bla", formula, input_data, model = GLS.model, set = list(gls=1))
		experimental_variogram = variogram(g, boundaries = boundaries, ...)
	}
	
	# request by Jon Skoien
	if(miscFitOptions[["merge.small.bins"]]) {
		if(verbose) cat("Checking if any bins have less than 5 points, merging bins when necessary...\n\n")
		while(TRUE) {
			if(length(experimental_variogram$np[experimental_variogram$np < miscFitOptions[["min.np.bin"]]]) == 0 | length(boundaries) == 1) break
			boundaries = boundaries[2:length(boundaries)]			
			if(!is(GLS.model, "variogramModel")) {
				experimental_variogram = variogram(formula, input_data,boundaries = boundaries, ...)
			} else {
				experimental_variogram = variogram(g, boundaries = boundaries, ...)
			}
		}	
	}
	
	# set initial values
	if(is.na(start_vals[1])) {  # Nugget
		initial_nugget = min(experimental_variogram$gamma)
	} else {
		initial_nugget = start_vals[1]
	}
	if(is.na(start_vals[2])) { # Range
		initial_range = 0.1 * maxdist   # 0.10 times the length of the central axis through the area
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
	
	getModel = function(psill, model, range, kappa, nugget, fit_range, fit_sill, fit_nugget, verbose)
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
						debug.level = 0,fit.method=fit.method), 
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
			model_fit = getModel(initial_sill - initial_nugget, m, initial_range, kappa = 0, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose)
			if(!is.null(model_fit)) {	# skip models that failed
				vgm_list[[counter]] = model_fit
				SSerr_list = c(SSerr_list, attr(model_fit, "SSErr"))}
			counter = counter + 1
		} else {                 # Else loop also over kappa values
			for(k in kappa) {
				model_fit = getModel(initial_sill - initial_nugget, m, initial_range, k, initial_nugget, fit_range, fit_sill, fit_nugget, verbose = verbose)
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