# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################

#load("/projects/daymet2/rdata/stignatius_prcp.Rdata")
#load("/projects/daymet2/rdata/stjohns_prcp.Rdata")
#load("/projects/daymet2/rdata/redlands_prcp_all.Rdata")
#
#prcp.mat[is.nan(prcp.mat)]<-NA
#
#
#mids.prcp <- mice(prcp.mat,method="norm",m=5)
#
#f <- c()
#
#for (x in seq(mids.prcp$m))
#{
#	f<-rbind(f,complete(mids.prcp,x)[,1])
#}
#
#f[f < 0] = 0
#f <- f^2
#
#f <- colMeans(f)
#
#
#fit <- array(0,dim=c(length(prcp.obs)))
#fit[po_mask] <- f
#
#(sum(f[xval_mask])-sum(prcp.obs[xval_mask]))/sum(prcp.obs[xval_mask])*100

#library(mice)
#library(Amelia)
library(norm)
#library(mvnmle)
SEED<-4324

#impute_mean <-function(mat)
#{
#	mat[is.nan(mat)]<-NA
#	
#	cat("Matrix shape: ",dim(mat),"\n")
#	
#	mids.prcp <- mice(mat,method="norm",m=5,printFlag=FALSE,seed=SEED)
#	
#	f <- c()
#	
#	for (x in seq(mids.prcp$m))
#	{
#		f<-rbind(f,complete(mids.prcp,x)[,1])
#	}
#	
#	f[f < 0] = 0
#	f <- f^2
#	
#	f <- colMeans(f)
#	
#	#p<-pool(with(mids.prcp, lm(V1 ~ 1)))
#	#return(as.numeric(p$qbar))
#	
#	return(mean(f))
#}
#
#impute_prcp <-function(mat)
#{
#	mat[is.nan(mat)]<-NA
#	
#	cat("Matrix shape: ",dim(mat),"\n")
#	
#	mids.prcp <- mice(mat,method="norm.predict",m=5,printFlag=TRUE)
#
#	f <- c()
#	
#	for (x in seq(mids.prcp$m))
#	{
#		f<-rbind(f,complete(mids.prcp,x)[,1])
#	}
#	
#	f <- colMeans(f)
#	f[f < 0] = 0
#	#f <- f^2
#	
#	return(f)
#}

impute_norm <- function(mat)
{	
	#save(list=c("mat"),file="/projects/daymet2/rdata/impute_norm.Rdata")
	rngseed(SEED)
	mat[is.nan(mat)]<-NA
	pre <- prelim.norm(mat)
	mle <- em.norm(pre,showits=FALSE)
	
	impute.params <- getparam.norm(pre,mle)
	mu <- impute.params$mu[1]
	sigma <- impute.params$sigma[1,1]
	#mle.imputed <- imp.norm(pre, mle, mat)
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

#impute_mvnmle <- function(mat)
#{
#	mat[is.nan(mat)]<-NA
#	rslt <- mlest(mat)
#	
#	return(c(rslt$muhat[1],rslt$sigmahat[1,1]))
#}


#impute_amelia <-function(mat,norms=NULL,stds=NULL)
#{
#	
##	if (!is.null(norms) & !is.null(stds))
##	{
##		priors = matrix(rep(0,ncol(mat)*4),nrow=ncol(mat),ncol=4)
##		priors[,2] <- seq(ncol(mat))
##		priors[,3] <- norms
##		priors[,4] <- stds
##	}
##	else
##	{
##		priors <- NULL
##	}
#	
#	priors <- NULL#matrix(priors[1,],ncol=4)
#	
#	mat[is.nan(mat)]<-NA
#	mat <<- mat
#	
#	#save.image('/projects/daymet2/rdata/impute_test.RData')
#	
#	cat("Matrix shape: ",dim(mat),"\n")
#	
#	#print("PRIORS")
#	#print(priors)
#	a.out<-amelia(mat,m=5,p2s=0,priors=priors)
#	
#	obs.mask <- ! a.out$missMatrix[,1]
#	
#	a.over <- overimpute(a.out,var=1)
#	
#	imp <- c()
#	
#	for (x in seq(a.out$m))
#	{
#		imp<-rbind(imp,a.out$imputations[[x]][,1])
#	}
#	
#	imp <- colMeans(imp)
#	
#	idxs <- order(a.over[,1])
#	imp[obs.mask] <- a.over[idxs,3]
#	
#	return(imp)
#}


