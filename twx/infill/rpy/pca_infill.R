# Functions for infilling missing values for an incomplete time series
# using probabilistic principal component analysis (PPCA).

# Stacklies W, Redestig H, Scholz M, Walther D, Selbig J. 2007. 
# pcaMethods–a bioconductor package providing PCA methods for incomplete data.
# Bioinformatics 23: 1164–1167. DOI: 10.1093/bioinformatics/btm069.

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
options(warn=-1)
library(changepoint)
library(pcaMethods)

FRAC_OBS<-0.6#.5
SEED<-4324
THRESHOLD<-1e-5
MAX_R2CUM<-0.99#0.995
CENTER<-FALSE
SCALE<-'none'

#Infill missing values for an incomplete station time series
#using probabilistic principal component analysis (PPCA), 
#a form of PCA robust to missing values implemented in the 
#R Bioconductor pcaMethods package:

#Stacklies W, Redestig H, Scholz M, Walther D, Selbig J. 2007. 
#pcaMethods–a bioconductor package providing PCA methods for incomplete data.
#Bioinformatics 23: 1164–1167. DOI: 10.1093/bioinformatics/btm069.

#Parameters
#----------
#pca_matrix : matrix
#	N*P 2-D matrix where N is the number days in the time series and P
#	is the number of station and/or other time series to be used for
# 	estimation. The first column is the station time series of focus.
#	Use NaN to represent missing observations.
#means : vector or boolean, optional
# 	A vector of size P containing previously estimated means of each column.
#	Each column will be mean centered based on these means. If single boolean and True,
#	each column will be mean centered with means calculated with the incomplete data.
#	If NULL or False, no mean centering is performed.
#scales : vector or boolean, optional
# 	A vector of size P containing previously estimated standard deviations  of each column.
#	Each column will be scaled based on these standard deviations. If single boolean and True,
#	each column will be scaled with standard deviations calculated with the incomplete data.
#	If NULL or False, no scaling is performed.
#frac_obs : double, optional
#	The fraction of the total number of columns that should be used as the
#	initial number of PCs. Example: if frac_obs is 0.5 and the number of columns
#	is 10, the initial number of PCs will be 5.
#max_r2cum : double, optional
#	The required variance to be explained by the PCs. Example: if 0.99, the first
#	n PCs that account for 99% of the variance in pca_matrix will be used
#npcs : int, optional
#	Use a specific set number of PCs. If npcs = 0, the number of PCs is determined
#	dynamically based on max_r2cum.
#convThres : float, optional
#	The convergence threshold for the PPCA algorithm.
#verbose : boolean, optional
#	If true, PPCA progress information is printed
	

#Returns
#----------
#list
#ppca_fit
#	The infilled time series. Every day contains an estimate from the PPCA algorithm
#	even if it was not initially missing.
#npcs
#	The final number of PCs used in PPCA
ppca_tair <-function(pca_matrix,means=NULL,scales=NULL,frac_obs=FRAC_OBS,max_r2cum=MAX_R2CUM,npcs=0,convThres=THRESHOLD,verbose=FALSE)
{
	pca_matrix[is.nan(pca_matrix)]<-NA
	
	max_pc<-run_ppca(pca_matrix,means,scales,frac_obs,max_r2cum,npcs,convThres,verbose)
	eval_pcs<-max_pc$npcs
	ppca_mod<-max_pc$ppca_rslt
	
	ppca_fit <- fitted(ppca_mod,nPcs = eval_pcs)
	
	if(is.vector(scales) & ! is.logical(scales))
	{
		ppca_fit <- sweep(ppca_fit,2,scales,"*")
	}
	
	if(is.vector(means) & ! is.logical(means))
	{
		ppca_fit <- sweep(ppca_fit,2,means,"+")
	}
	
	return(list(ppca_fit=ppca_fit[,1],npcs=eval_pcs))
}

#Run through the actual PPCA algorithm until the max_r2cum is met
#or using npcs if npcs is not 0.
run_ppca <- function(pca_tair,means=NULL,scales=NULL,frac_obs=FRAC_OBS,max_r2cum=MAX_R2CUM,npcs=0,convThres=THRESHOLD,verbose=FALSE)
{
	
	if(is.logical(means))
	{
		center<-means
		pca_matrix<-pca_tair
	}
	else if(is.vector(means))
	{
		center<-FALSE
		pca_matrix<-sweep(pca_tair,2,means,"-")
	}
	else
	{
		center<-CENTER
		pca_matrix<-pca_tair
	}
	
	
	if(is.logical(scales))
	{
		if (scales)
		{
			scale<-'uv'
		}
		else
		{
			scale<-'none'
		}
	}
	else if(is.vector(scales))
	{
		scale<-'none'
		pca_matrix<-sweep(pca_matrix,2,scales,"/")
	}
	else
	{
		scale<-SCALE
	}
	
	if(verbose) cat("THE CONVERGENCE THRES IS ",convThres,"\n")
	
	if (npcs != 0)
	{
		if(verbose) cat("running with set num_pcs=",npcs,"\n")
		ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres)
		if(verbose) print(ppca_rslt)
		return(list(npcs=npcs,ppca_rslt=ppca_rslt))
	}
	
	npcs <- round((ncol(pca_tair)-1)*frac_obs,0)
	if (npcs < 2)
	{
		npcs <- 2	
	}
	
	pcs_fnd <- FALSE
	warn_fnd <- FALSE
	
	
	
	rslts <- list()
	best_r2cum <- c(0,0) #r2cum,npcs
	
	while(!pcs_fnd)
	{
		stopifnot(npcs > 0)
		
		if (is.null(rslts[[as.character(npcs)]]))
		{
			if (warn_fnd & best_r2cum[2] > npcs)
			{
				ppca_rslt<-rslts[[as.character(best_r2cum[2])]]
				npcs <- best_r2cum[2]
			}
			else
			{	
				if(verbose) cat("testing num_pcs=",npcs,"\n")
				ppca_rslt<-tryCatch(pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres),error=function(e) e, warning=function(w) w)
			}
		}
		else
		{
			if (warn_fnd & best_r2cum[2] > npcs)
			{
				ppca_rslt<-rslts[[as.character(best_r2cum[2])]]
				npcs <- best_r2cum[2]
			}
			else
			{
				ppca_rslt<-rslts[[as.character(npcs)]]
			}
		}
		
		if(verbose) print(ppca_rslt)
		if (is(ppca_rslt,"warning") || is(ppca_rslt,"simpleError"))
		{
			npcs_new <- as.numeric(strsplit(ppca_rslt$message," ")[[1]][4]) - 1
			if(verbose) print(npcs_new)
			if (is.na(npcs_new))
			{
				npcs <- npcs - 1
			}
			else if (npcs >= ncol(pca_matrix))
			{
				npcs <- npcs - 1
			}
			else
			{
				npcs <- npcs_new	
			}
			
			warn_fnd <- TRUE
		}
		else if (max(ppca_rslt@R2cum) >= max_r2cum)
		{
#			npcs_maxr2 <- length(ppca_rslt@R2cum)
#			npcs <- npcs_maxr2
#			pcs_fnd <- TRUE
#			warn_fnd <- FALSE
#			fnl_ppca_rslt <- ppca_rslt
			
			npcs_maxr2 <- which(ppca_rslt@R2cum >= max_r2cum)[1]
			pcs_fnd <- TRUE
			warn_fnd <- FALSE
			if (npcs_maxr2 == npcs)
			{
				fnl_ppca_rslt <- ppca_rslt
			}
			else
			{
				npcs <- npcs_maxr2
				
				if (is.null(rslts[[as.character(npcs)]]))
				{
					if(verbose) cat("testing num_pcs for final=",npcs,"\n")
					fnl_ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres)
					#ncalls = ncalls + 1
					if(verbose) print(fnl_ppca_rslt)
				}
				else
				{
					fnl_ppca_rslt<-rslts[[as.character(npcs)]]	
				}
			}
			
			
			
		}
		else if (warn_fnd)
		{
			pcs_fnd <- TRUE
			fnl_ppca_rslt<-ppca_rslt
		}
		else
		{
			if (max(ppca_rslt@R2cum)>best_r2cum[1])
			{
				best_r2cum <- c(max(ppca_rslt@R2cum),npcs)		
			}
			
			rslts[[as.character(npcs)]] <- ppca_rslt
			
			add_pcs <- min(c(10,round((max_r2cum - ppca_rslt@R2cum[length(ppca_rslt@R2cum)])/ppca_rslt@R2[length(ppca_rslt@R2)])))
			add_pcs <- max(c(1,add_pcs))
			
			
			if(verbose) cat("adding num_pcs=",add_pcs,"\n")
			npcs <- npcs + add_pcs
			warn_fnd <- FALSE
		}
	}
	
	if (npcs == 1)
	{
		if (length(rslts) > 0)
		{
			max_npcs = max(as.numeric(names(rslts)))
			if (max_npcs > npcs)
			{
				npcs <- max_npcs
				fnl_ppca_rslt<-rslts[[as.character(npcs)]]
				if(verbose) cat("Removed bogus PC1 100% variance\n")
			}
		}
	}
	
	if(verbose) cat("PPCA DONE. FINAL # of PCs: ",npcs,"\n")
	if(verbose) print(fnl_ppca_rslt)
	return(list(npcs=npcs,ppca_rslt=fnl_ppca_rslt))
}


hasVarChgPt <- function(vals,sig=0.0000000001,verbose=FALSE)
{
	if (verbose) cat("VARIANCE CHANGE POINT AT: ",cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig)),"\n")
	return(length(cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig))) > 0)
}

getVarChgPt <- function(vals,sig=0.0000000001,verbose=FALSE)
{	
	aCpts <- cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig))
	#aCpts <- cpts(cpt.var(vals))
	if (verbose) cat("VARIANCE CHANGE POINT AT: ",aCpts,"\n")
	return(aCpts)
}

		


