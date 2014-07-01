# Functions for estimating the mean and variance of incomplete station
# time series using the "norm" R package:

# Schafer JL. 1997. Analysis of Incomplete Multivariate Data. 
# Chapman and Hall/CRC: Boca Raton, FL.
###############################################################################

library(norm)
SEED<-4324

infill_mu_sigma <- function(mat)
{	

	rngseed(SEED)
	mat[is.nan(mat)]<-NA
	pre <- prelim.norm(mat)
	mle <- em.norm(pre,showits=FALSE)
	
	impute.params <- getparam.norm(pre,mle)
	mu <- impute.params$mu[1]
	sigma <- impute.params$sigma[1,1]

	return(c(mu,sigma))
}

impute_norm_all <- function(mat)
{	
	#save(list=c("mat"),file="/projects/daymet2/rdata/impute_norm.Rdata")
	rngseed(SEED)
	mat[is.nan(mat)]<-NA
	pre <- prelim.norm(mat)
	mle <- em.norm(pre,showits=FALSE)
	
	#impute.params <- getparam.norm(pre,mle)
	#mu <- impute.params$mu[1]
	#sigma <- impute.params$sigma[1,1]
	mle.imputed <- imp.norm(pre, mle, mat)

	return(mle.imputed[,1])
}