# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

distance.weight <- function(x, xy, tau) {
	# x is a vector location
	# xy is an array of locations, one per row
	# tau is the bandwidth
	# Returns a vector of weights
	apply(xy, 1, function(z) exp(-(z-x) %*% (z-x) / (2 * tau^2)))
}

covariance <- function(y, weights) {
	# y is an m by n matrix
	# weights is length m
	# Returns the weighted covariance matrix of y (by columns).
	if (missing(weights)) return (cov(y))
	w <- zapsmall(weights / sum(weights)) # Standardize the weights
	y.bar <- apply(y * w, 2, sum)         # Compute column means
	z <- t(y) - y.bar                     # Remove the means
	z %*% (w * t(z))  
}

correlation <- function(y, weights) {
	z <- covariance(y, weights)
	sigma <- sqrt(diag(z))       # Standard deviations
	z / (sigma %o% sigma)
}

#gw.pca <- function(x, xy, y, tau) {
#	# x is a vector denoting a location
#	# xy is a set of locations as row vectors
#	# y is an array of attributes, also as rows
#	# tau is a bandwidth
#	# Returns a `princomp` object for the geographically weighted PCA
#	# ..of y relative to the point x.
#	w <- distance.weight(x, xy, tau)
#	princomp(covmat=correlation(y, w))
#}

gw.pca <- function(y,w) 
{
	# x is a vector denoting a location
	# y is an array of attributes, also as rows
	# tau is a bandwidth
	# Returns a `princomp` object for the geographically weighted PCA
	# ..of y relative to the point x.
	w <- distance.weight(x, xy, tau)
	princomp(covmat=correlation(y, w))
}

pca_tair <- function(ngh_obs,ndays,ngh_lon,ngh_lat,ngh_elev,ngh_tdi,ngh_lst,ngh_tair,ngh_wgt,pt)
{	
	ngh_obs <- matrix(ngh_obs,ndays,length(ngh_lon),byrow=TRUE)
	
	#Build dataframes
	stns_ngh <- data.frame(lon=ngh_lon,lat=ngh_lat,elev=ngh_elev,tdi=ngh_tdi,lst=ngh_lst,tair=ngh_tair,ngh_wgt=ngh_wgt)
	pt <- data.frame(lon=pt[1],lat=pt[2],elev=pt[3],tdi=pt[4],lst=pt[5],tair=pt[6])
	
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
		lm_ld <- lm(a.pca$rotation[,x]~lon+lat+elev+tdi+lst,data=stns_ngh,weights=ngh_wgt)
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
	return(tair+pt$tair)
}

local.center <- function(x,wt)
	sweep(x,2,colSums(sweep(x,1,wt,'*'))/sum(wt))

set.seed(17)
n.data <- 550
n.vars <- 30
xy <- matrix(rnorm(n.data * 2), ncol=2)
y <- matrix(rnorm(n.data * n.vars), ncol=n.vars)

stns.mat <- as.matrix(stns[,c("ELEV","TDI","LST","TAIR")])
wgts <- as.vector(stns$WGT)

##############################################
covmat <- covariance(X,wgt)
sigma.wgt <- sqrt(diag(z))
w <- zapsmall(wgt / sum(wgt)) # Standardize the weights
mean.wgt <- apply(X * w, 2, sum)         # Compute column means
X.mc <- sweep(X,2,mean.wgt,"-")
X.sc <- sweep(X.mc,2,sigma,"/")
prcomp(X.sc,center=FALSE,scale=FALSE)
###########################################

wgt <- as.vector(wgt)
covmat <- correlation(X,wgt)
a<-princomp(covmat=covmat)


w <- zapsmall(wgt / sum(wgt)) # Standardize the weights
X.bar <- apply(X * w, 2, sum)         # Compute column means
X.mc <- sweep(X,2,X.bar,"-")
X.sc <- sweep(X.mc,2,sigma,"/")

b<-princomp(stns.mat)
z <- covariance(X, wgt)
sigma <- sqrt(diag(z))

covmat <- covariance(stns.mat,wgts)
b<-svd(sweep(sweep(local.center(X,wgt),2,sigma,"/"),1,sqrt(wgt),'*'),nu=0)
c<-prcomp(sweep(sweep(local.center(X,wgt),2,sigma,"/"),1,sqrt(wgt),'*'),center=FALSE)
