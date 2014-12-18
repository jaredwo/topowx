# Functions for estimating the mean and variance of incomplete station
# time series using the "norm" R package:

# Schafer JL. 1997. Analysis of Incomplete Multivariate Data. 
# Chapman and Hall/CRC: Boca Raton, FL.

#Copyright 2014, Jared Oyler.
#
#This file is part of TopoWx.
#
#TopoWx is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#TopoWx is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with TopoWx.  If not, see <http://www.gnu.org/licenses/>.

###############################################################################

library(norm)
SEED<-4324

#Estimate the mean and variance of an incomplete station time series

#Parameters
#----------
#mat : matrix
#	N*P 2-D matrix where N is the number days in the time series and P
#	is the number of station and/or other time series to be used for
# 	estimation. The first column is the station time series of focus.
#	Use NaN to represent missing observations.

#Returns
#----------
#vector
#	A vector containing the estimated mean and variance for the first
#	column of the input matrix.
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