# TODO: Add comment
# 
# Author: jared.oyler
###############################################################################


library(pcaMethods)
library(changepoint)

LAG<-2
TOL<--1e-3
FRAC_OBS<-0.6#.5
CV_NSEGS<-10
CV_NRUNS<-1
SEED<-4324
THRESHOLD<-1e-5
options(warn=-1)
MAX_R2CUM<-0.99#0.995
CENTER<-FALSE
SCALE<-'none'  #'none'
VERBOSE<-FALSE

kEstimate_po <- function(Matrix, method="ppca", evalPcs=1:3, segs=3, nruncv=5,
		em="q2", allVariables=FALSE,po_crit=0.5,
		verbose=interactive(), ...) {
	
	fastKE <- FALSE
	if (method == "ppca" | method == "bpca" | method == "nipals" |
			method == "nlpca")
		fastKE <- TRUE
	
	method <- match.arg(method, listPcaMethods())
	em <- match.arg(em, c("nrmsep", "q2"))
	maxPcs <- max(evalPcs)
	lengthPcs <- length(evalPcs)
	
	## If the data is a data frame, convert it into a matrix
	Matrix <- as.matrix(Matrix, rownames.force=TRUE)
	if(maxPcs > (ncol(Matrix) - 1))
		stop("maxPcs exceeds matrix size, choose a lower value!")
	
	## And now check if everything is right...
	if( !checkData(Matrix, verbose=interactive()) )
		stop("Invalid data format! Use checkData(Matrix, verbose = TRUE) for details.\n")
	
	if( (sum(is.na(Matrix)) == 0) && (allVariables == FALSE) )
		stop("No missing values. Maybe you want to set allVariables = TRUE. Exiting\n")
	
	
	missing <- apply(is.na(Matrix), 2, sum) > 0
	missIx     <- which(missing == TRUE)
	if (allVariables)
		missIx <- 1:ncol(Matrix)
	missIx <- 1:1
	
	complete <- !missing
	compIx    <- which(complete == TRUE)
	
	error <- matrix(0, length(missIx), length(evalPcs))
	iteration <- 0
	for(nPcs in evalPcs) {
		
		## If the estimated observations are just scores %*% t(loadings)
		## we can calculate all we need at once, this saves many
		## iterations...
		if (fastKE) nPcs = maxPcs
		
		iteration = iteration + 1
		if (verbose && !fastKE) { cat("Doing CV for ", nPcs, " component(s) \n") }
		else if (verbose && fastKE) {cat("Doing CV ... \n")}
		for(cviter in 1:nruncv) {
			pos <- 0
			if (verbose) cat("Incomplete variable index: ")
			for (index in missIx) {
				pos <- pos + 1
				#cat(pos, ":", sep="")
				target <- Matrix[, index, drop = FALSE]
				compObs <- !is.na(target)
				missObs <- is.na(target)
				nObs <- sum(compObs)
				
				## Remove all observations that are missing in the target genes,
				## as additional missing values may tamper the results
				set <- Matrix[compObs,]
				
				if (nObs >= (2 * segs)) {
					segments <- segs
				} else
					segments <- ceiling(nObs / 2)
				
				## We assume uniformly distributed missing values when
				## choosing the segments
				tt <- gl(segments, ceiling(nObs / segments))[1:nObs]
				cvsegs <- split(sample(nObs), tt)
				#print(cvsegs)
				set <- Matrix[compObs,]
				if (fastKE) {
					hss_sum <- array(0, length(evalPcs))
					hss_cnt <- array(0, length(evalPcs))
					nrmsep <- array(0, length(evalPcs))
					q2 <- array(0, length(evalPcs))
				} else {
					nrmsep <- 0; q2 <- 0
				}
				
				for (i in 1:length(cvsegs)) {
					n <- length(cvsegs[[i]]) # n is the number of created
					# missing values
					## Impute values using the given regression method
					testSet <- set
					testSet[cvsegs[[i]], index] <- NA
					if (method == "llsImpute") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = FALSE, 
								center = FALSE, xval = index)
					} else if (method == "llsImputeAll") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = TRUE, 
								center = FALSE, xval = index)
					} else {
						estimate <- pca(testSet, nPcs = nPcs, verbose = FALSE,
								method = method, center = CENTER,...)
					}
					
					if (fastKE) {
						for (np in evalPcs) {
							estiFitted <- fitted(estimate, data = NULL, nPcs = np)
							estimateVec <- estiFitted[, index]
							original <- target[compObs, ]
							estimateVec[-cvsegs[[i]]] <- testSet[-cvsegs[[i]], index]
							## Error of prediction, error is calculated for removed
							## elements only
							nIx <- which(evalPcs == np) 
							if (em == "nrmsep") {
								
								modeled_po <- estimateVec[cvsegs[[i]]] >= po_crit
								hss<-calc_hss(original[cvsegs[[i]]],modeled_po)
								#print(hss)
								hss_sum[nIx] <- hss_sum[nIx] + hss
								hss_cnt[nIx] <- hss_cnt[nIx] + 1
								
								nrmsep[nIx] <- nrmsep[nIx] + sum( (original - estimateVec)^2) 
							} else {
								q2[nIx] <- q2[nIx] + sum( (original - estimateVec)^2 )
							}
						}    
					} else {
						estimate <- estimate@completeObs[, index]
						original <- target[compObs, ]
						## Error of prediction, error is calculated for removed
						## elements only
						if (em == "nrmsep") {
							nrmsep <- nrmsep + sum( (original - estimate)^2)
						} else {
							q2 <- q2 + sum( (original - estimate)^2 )
						}
					}
				} ## iteration over cv segments
				
				if (fastKE) {
					if (em == "nrmsep") {
						error[pos, ] <-
								error[pos, ] + nrmsep / (nrow(set) * var(set[,index]))
					} else
						error[pos, ] <- error[pos, ] + (1 - (q2 / sum(set[, index]^2)))
				} else {
					if (em == "nrmsep") {
						error[pos, iteration] <- error[pos, iteration] + 
								nrmsep / (nrow(set) * var(set[,index]))
					} else
						error[pos, iteration] <-
								error[pos, iteration] + (1 - (q2 / sum(set[, index]^2)))
				}
			} # iteration over variables
			if (verbose) cat("\n")
			
		} #iteration over nruncv
		
		## The error is the sum over the independent cross validation runs
		error <- error / nruncv
		
		if (verbose && !fastKE)
			cat("The average", em, "for k =", iteration, "is", 
					sum(error[,iteration]) / nrow(error), "\n")
		
		## if nlpca, ppca, bpca, nipals we do not need to iterate over the
		## number of components...
		if (fastKE) break
	} # iteration over number components
	
	if (em == "nrmsep")
		avgError <- sqrt(apply(error, 2, sum) / nrow(error))
	else
		avgError <- apply(error, 2, sum) / nrow(error)
	
	ret <- list()
	if (em == "nrmsep")
		ret$bestNPcs <- evalPcs[which(avgError == min(avgError))]
	else ret$bestNPcs <- evalPcs[which(avgError == max(avgError))]
	ret$eError <- avgError
	if(em == "nrmsep") ret$variableWiseError <- sqrt(error)
	else ret$variableWiseError <- error
	ret$evalPcs <- evalPcs
	ret$variableIx <- missIx
	#print(mae_sum/mae_cnt)
	return(list(ret=ret,hss=hss_sum/hss_cnt))
}


#kEstimate_my <- function(Matrix, method="ppca", evalPcs=1:3, segs=3, nruncv=5,
#		em="q2", allVariables=FALSE,
#		verbose=interactive(), ...) {
#	
#	fastKE <- FALSE
#	if (method == "ppca" | method == "bpca" | method == "nipals" |
#			method == "nlpca")
#		fastKE <- TRUE
#	
#	method <- match.arg(method, listPcaMethods())
#	em <- match.arg(em, c("nrmsep", "q2"))
#	maxPcs <- max(evalPcs)
#	lengthPcs <- length(evalPcs)
#	
#	## If the data is a data frame, convert it into a matrix
#	Matrix <- as.matrix(Matrix, rownames.force=TRUE)
#	#if(maxPcs > (ncol(Matrix) - 1))
#	if(maxPcs > ncol(Matrix))
#		stop("maxPcs exceeds matrix size, choose a lower value!")
#	
#	## And now check if everything is right...
#	if( !checkData(Matrix, verbose=interactive()) )
#		stop("Invalid data format! Use checkData(Matrix, verbose = TRUE) for details.\n")
#	
#	if( (sum(is.na(Matrix)) == 0) && (allVariables == FALSE) )
#		stop("No missing values. Maybe you want to set allVariables = TRUE. Exiting\n")
#	
#	
#	missing <- apply(is.na(Matrix), 2, sum) > 0
#	missIx     <- which(missing == TRUE)
#	if (allVariables)
#		missIx <- 1:ncol(Matrix)
#	missIx <- 1:1
#	
#	complete <- !missing
#	compIx    <- which(complete == TRUE)
#	
#	error <- matrix(0, length(missIx), length(evalPcs))
#	iteration <- 0
#	for(nPcs in evalPcs) {
#		
#		## If the estimated observations are just scores %*% t(loadings)
#		## we can calculate all we need at once, this saves many
#		## iterations...
#		if (fastKE) nPcs = maxPcs
#		
#		iteration = iteration + 1
#		if (verbose && !fastKE) { cat("Doing CV for ", nPcs, " component(s) \n") }
#		else if (verbose && fastKE) {cat("Doing CV ... \n")}
#		for(cviter in 1:nruncv) {
#			pos <- 0
#			if (verbose) cat("Incomplete variable index: ")
#			for (index in missIx) {
#				pos <- pos + 1
#				#cat(pos, ":", sep="")
#				target <- Matrix[, index, drop = FALSE]
#				compObs <- !is.na(target)
#				missObs <- is.na(target)
#				nObs <- sum(compObs)
#				
#				## Remove all observations that are missing in the target genes,
#				## as additional missing values may tamper the results
#				set <- Matrix[compObs,]
#				
#				if (nObs >= (2 * segs)) {
#					segments <- segs
#				} else
#					segments <- ceiling(nObs / 2)
#				
#				## We assume uniformly distributed missing values when
#				## choosing the segments
#				tt <- gl(segments, ceiling(nObs / segments))[1:nObs]
#				#cvsegs <- split(sample(nObs), tt)
#				cvsegs <- split(1:nObs, tt)
#				#print(cvsegs)
#				set <- Matrix[compObs,]
#				if (fastKE) {
#					mae_sum <- array(0, length(evalPcs))
#					mae_cnt <- array(0, length(evalPcs))
#					bias_sum <- array(0, length(evalPcs))
#					bias_cnt <- array(0, length(evalPcs))
#					mse_sum <- array(0, length(evalPcs))
#					mse_cnt <- array(0, length(evalPcs))
#					press <- array(0, length(evalPcs))
#					
#					biasm <- matrix(0,segments, length(evalPcs))
#					maem <- matrix(0,segments, length(evalPcs))
#					
#					nrmsep <- array(0, length(evalPcs))
#					q2 <- array(0, length(evalPcs))
#				} else {
#					nrmsep <- 0; q2 <- 0
#				}
#				
#				for (i in 1:length(cvsegs)) {
#					n <- length(cvsegs[[i]]) # n is the number of created
#					# missing values
#					## Impute values using the given regression method
#					testSet <- set
#					testSet[cvsegs[[i]], index] <- NA
#					if (method == "llsImpute") {
#						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
#								allVariables = FALSE, 
#								center = FALSE, xval = index)
#					} else if (method == "llsImputeAll") {
#						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
#								allVariables = TRUE, 
#								center = FALSE, xval = index)
#					} else {
#						estimate <- pca(testSet, nPcs = nPcs, verbose = FALSE,
#								method = method, center = TRUE,...)
#					}
#					
#					if (fastKE) {
#						for (np in evalPcs) {
#							estiFitted <- fitted(estimate, data = NULL, nPcs = np)
#							estimateVec <- estiFitted[, index]
#							original <- target[compObs, ]
#							estimateVec[-cvsegs[[i]]] <- testSet[-cvsegs[[i]], index]
#							## Error of prediction, error is calculated for removed
#							## elements only
#							nIx <- which(evalPcs == np) 
#							
#							err <- estimateVec[cvsegs[[i]]] - original[cvsegs[[i]]]
#							err_abs<-abs(estimateVec[cvsegs[[i]]] - original[cvsegs[[i]]])
#							
#							mae_sum[nIx] <- mae_sum[nIx] + sum(err_abs)
#							mae_cnt[nIx] <- mae_cnt[nIx] + length(err_abs)
#							
#							bias_sum[nIx] <- bias_sum[nIx] + sum(err)
#							bias_cnt[nIx] <- bias_cnt[nIx] + length(err)
#							
#							biasm[i,nIx] <- mean(err)
#							maem[i,nIx] <- mean(err_abs)
#							
#							mse_sum[nIx] <- mse_sum[nIx] + sum(err^2)
#							mse_cnt[nIx] <- mse_cnt[nIx] + length(err)
#							
#							press[nIx] <- press[nIx] + sum(err^2)
#							
#							if (em == "nrmsep") {
#								
#								nrmsep[nIx] <- nrmsep[nIx] + sum( (original - estimateVec)^2) 
#							} else {
#								q2[nIx] <- q2[nIx] + sum( (original - estimateVec)^2 )
#							}
#						}    
#					} else {
#						estimate <- estimate@completeObs[, index]
#						original <- target[compObs, ]
#						## Error of prediction, error is calculated for removed
#						## elements only
#						if (em == "nrmsep") {
#							nrmsep <- nrmsep + sum( (original - estimate)^2)
#						} else {
#							q2 <- q2 + sum( (original - estimate)^2 )
#						}
#					}
#				} ## iteration over cv segments
#				
#				if (fastKE) {
#					if (em == "nrmsep") {
#						error[pos, ] <-
#								error[pos, ] + nrmsep / (nrow(set) * var(set[,index]))
#					} else
#						error[pos, ] <- error[pos, ] + (1 - (q2 / sum(set[, index]^2)))
#				} else {
#					if (em == "nrmsep") {
#						error[pos, iteration] <- error[pos, iteration] + 
#								nrmsep / (nrow(set) * var(set[,index]))
#					} else
#						error[pos, iteration] <-
#								error[pos, iteration] + (1 - (q2 / sum(set[, index]^2)))
#				}
#			} # iteration over variables
#			if (verbose) cat("\n")
#			
#		} #iteration over nruncv
#		
#		## The error is the sum over the independent cross validation runs
#		error <- error / nruncv
#		
#		if (verbose && !fastKE)
#			cat("The average", em, "for k =", iteration, "is", 
#					sum(error[,iteration]) / nrow(error), "\n")
#		
#		## if nlpca, ppca, bpca, nipals we do not need to iterate over the
#		## number of components...
#		if (fastKE) break
#	} # iteration over number components
#	
#	if (em == "nrmsep")
#		avgError <- sqrt(apply(error, 2, sum) / nrow(error))
#	else
#		avgError <- apply(error, 2, sum) / nrow(error)
#	
#	ret <- list()
#	if (em == "nrmsep")
#		ret$bestNPcs <- evalPcs[which(avgError == min(avgError))]
#	else ret$bestNPcs <- evalPcs[which(avgError == max(avgError))]
#	ret$eError <- avgError
#	if(em == "nrmsep") ret$variableWiseError <- sqrt(error)
#	else ret$variableWiseError <- error
#	ret$evalPcs <- evalPcs
#	ret$variableIx <- missIx
#	
#	print(biasm)
#	print(apply(biasm,2,mean))
#	print("###################################")
#	print(maem)
#	print(apply(maem,2,mean))
#	
#	
#	return(list(ret=ret,mae=mae_sum/mae_cnt,bias=bias_sum/bias_cnt,mse=mse_sum/mse_cnt,press=press))
#}


kEstimate_my <- function(Matrix, method="ppca", evalPcs=1:3, segs=3, nruncv=5,
		em="q2", allVariables=FALSE,
		verbose=interactive(), ...) {
	
	fastKE <- FALSE
	if (method == "ppca" | method == "bpca" | method == "nipals" |
			method == "nlpca")
		fastKE <- TRUE
	
	method <- match.arg(method, listPcaMethods())
	maxPcs <- max(evalPcs)
	lengthPcs <- length(evalPcs)
	
	## If the data is a data frame, convert it into a matrix
	Matrix <- as.matrix(Matrix, rownames.force=TRUE)
	#if(maxPcs > (ncol(Matrix) - 1))
	if(maxPcs > ncol(Matrix))
		stop("maxPcs exceeds matrix size, choose a lower value!")
	
	## And now check if everything is right...
	if( !checkData(Matrix, verbose=interactive()) )
		stop("Invalid data format! Use checkData(Matrix, verbose = TRUE) for details.\n")
	
	if( (sum(is.na(Matrix)) == 0) && (allVariables == FALSE) )
		stop("No missing values. Maybe you want to set allVariables = TRUE. Exiting\n")
	
	missIx <- 1:1
	
	iteration <- 0
	for(nPcs in evalPcs) {
		
		## If the estimated observations are just scores %*% t(loadings)
		## we can calculate all we need at once, this saves many
		## iterations...
		if (fastKE) nPcs = maxPcs
		
		iteration = iteration + 1
		if (verbose && !fastKE) { cat("Doing CV for ", nPcs, " component(s) \n") }
		else if (verbose && fastKE) {cat("Doing CV ... \n")}
		for(cviter in 1:nruncv) {
			pos <- 0
			if (verbose) cat("Incomplete variable index: ")
			for (index in missIx) {
				pos <- pos + 1
				#cat(pos, ":", sep="")
				target <- Matrix[, index, drop = FALSE]
				compObs <- !is.na(target)
				missObs <- is.na(target)
				nObs <- sum(compObs)
				
				which.compObs <- which(compObs)
				
				## Remove all observations that are missing in the target genes,
				## as additional missing values may tamper the results
				#set <- Matrix[compObs,]
				set <- Matrix
				
				if (nObs >= (2 * segs)) {
					segments <- segs
				} else
					segments <- ceiling(nObs / 2)
				
				## We assume uniformly distributed missing values when
				## choosing the segments
				tt <- gl(segments, ceiling(nObs / segments))[1:nObs]
				#cvsegs <- split(sample(nObs), tt)
				cvsegs <- split(1:nObs, tt)
				
				for (i in 1:length(cvsegs)) 
				{
					cvsegs[[i]] <- which.compObs[cvsegs[[i]]]
				}
				
				#print(cvsegs)
				#set <- Matrix[compObs,]
				if (fastKE) {
					
					biasm <- matrix(0,segments, length(evalPcs))
					maem <- matrix(0,segments, length(evalPcs))
					msem <- matrix(0,segments, length(evalPcs))
					pressm <- matrix(0,segments, length(evalPcs))
					
				} 
				
				for (i in 1:length(cvsegs)) {
					n <- length(cvsegs[[i]]) # n is the number of created
					# missing values
					## Impute values using the given regression method
					testSet <- set
					testSet[cvsegs[[i]], index] <- NA
					if (method == "llsImpute") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = FALSE, 
								center = FALSE, xval = index)
					} else if (method == "llsImputeAll") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = TRUE, 
								center = FALSE, xval = index)
					} else {
						estimate <- pca(testSet, nPcs = nPcs, verbose = FALSE,
								method = method, center = CENTER,...)
					}
					
					if (fastKE) {
						for (np in evalPcs) {
							estiFitted <- fitted(estimate, data = NULL, nPcs = np)
							estimateVec <- estiFitted[, index]
							original <- target[compObs, ]
							original <- target
							estimateVec[-cvsegs[[i]]] <- testSet[-cvsegs[[i]], index]
							## Error of prediction, error is calculated for removed
							## elements only
							nIx <- which(evalPcs == np) 
							
							err <- estimateVec[cvsegs[[i]]] - original[cvsegs[[i]]]
							err_abs<-abs(err)
								
							biasm[i,nIx] <- mean(err)
							maem[i,nIx] <- mean(err_abs)
							msem[i,nIx] <- mean(err^2)
							pressm[i,nIx] <- sum(err^2)
							
						}    
					} 
				} ## iteration over cv segments
				
			} # iteration over variables
			if (verbose) cat("\n")
			
		} #iteration over nruncv
		
		## if nlpca, ppca, bpca, nipals we do not need to iterate over the
		## number of components...
		if (fastKE) break
	} # iteration over number components
	
	ret <- list()

	return(list(ret=ret,mae=apply(maem,2,mean),bias=apply(biasm,2,mean),mse=apply(msem,2,mean),press=apply(pressm,2,sum)))
}


kEstimate_prcp <- function(Matrix, po_bool, method="ppca", evalPcs=1:3, segs=3, nruncv=5,
		em="q2", allVariables=FALSE,
		verbose=interactive(), ...) {
	
	fastKE <- FALSE
	if (method == "ppca" | method == "bpca" | method == "nipals" |
			method == "nlpca")
		fastKE <- TRUE
	
	method <- match.arg(method, listPcaMethods())
	em <- match.arg(em, c("nrmsep", "q2"))
	maxPcs <- max(evalPcs)
	lengthPcs <- length(evalPcs)
	
	## If the data is a data frame, convert it into a matrix
	Matrix <- as.matrix(Matrix, rownames.force=TRUE)
	if(maxPcs > (ncol(Matrix) - 1))
		stop("maxPcs exceeds matrix size, choose a lower value!")
	
	## And now check if everything is right...
	if( !checkData(Matrix, verbose=interactive()) )
		stop("Invalid data format! Use checkData(Matrix, verbose = TRUE) for details.\n")
	
	if( (sum(is.na(Matrix)) == 0) && (allVariables == FALSE) )
		stop("No missing values. Maybe you want to set allVariables = TRUE. Exiting\n")
	
	
	missing <- apply(is.na(Matrix), 2, sum) > 0
	missIx     <- which(missing == TRUE)
	if (allVariables)
		missIx <- 1:ncol(Matrix)
	missIx <- 1:1
	
	complete <- !missing
	compIx    <- which(complete == TRUE)
	
	error <- matrix(0, length(missIx), length(evalPcs))
	iteration <- 0
	for(nPcs in evalPcs) {
		
		## If the estimated observations are just scores %*% t(loadings)
		## we can calculate all we need at once, this saves many
		## iterations...
		if (fastKE) nPcs = maxPcs
		
		iteration = iteration + 1
		if (verbose && !fastKE) { cat("Doing CV for ", nPcs, " component(s) \n") }
		else if (verbose && fastKE) {cat("Doing CV ... \n")}
		for(cviter in 1:nruncv) {
			pos <- 0
			if (verbose) cat("Incomplete variable index: ")
			for (index in missIx) {
				pos <- pos + 1
				#cat(pos, ":", sep="")
				target <- Matrix[, index, drop = FALSE]
				compObs <- !is.na(target)
				missObs <- is.na(target)
				nObs <- sum(compObs)
				
				## Remove all observations that are missing in the target genes,
				## as additional missing values may tamper the results
				set <- Matrix[compObs,]
				po_bool_set <- po_bool[compObs]
				
				if (nObs >= (2 * segs)) {
					segments <- segs
				} else
					segments <- ceiling(nObs / 2)
				
				## We assume uniformly distributed missing values when
				## choosing the segments
				tt <- gl(segments, ceiling(nObs / segments))[1:nObs]
				cvsegs <- split(sample(nObs), tt)
				#print(cvsegs)
				set <- Matrix[compObs,]
				if (fastKE) {
					mae_sum <- array(0, length(evalPcs))
					mae_cnt <- array(0, length(evalPcs))
					
					pcter_sum <- array(0, length(evalPcs))
					pcter_cnt <- array(0, length(evalPcs))
					
					nrmsep <- array(0, length(evalPcs))
					q2 <- array(0, length(evalPcs))
				} else {
					nrmsep <- 0; q2 <- 0
				}
				
				for (i in 1:length(cvsegs)) {
					n <- length(cvsegs[[i]]) # n is the number of created
					# missing values
					## Impute values using the given regression method
					testSet <- set
					testSet[cvsegs[[i]], index] <- NA
					if (method == "llsImpute") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = FALSE, 
								center = FALSE, xval = index)
					} else if (method == "llsImputeAll") {
						estimate <- llsImpute(testSet, k = nPcs, verbose = FALSE,
								allVariables = TRUE, 
								center = FALSE, xval = index)
					} else {
						estimate <- pca(testSet, nPcs = nPcs, verbose = FALSE,
								method = method, center = CENTER,...)
					}
					
					if (fastKE) {
						for (np in evalPcs) {
							estiFitted <- fitted(estimate, data = NULL, nPcs = np)
							estimateVec <- estiFitted[, index]
							original <- target[compObs, ]
							estimateVec[-cvsegs[[i]]] <- testSet[-cvsegs[[i]], index]
							## Error of prediction, error is calculated for removed
							## elements only
							nIx <- which(evalPcs == np) 
							if (em == "nrmsep") {
								
								mask_po <- po_bool_set[cvsegs[[i]]]
								
								p_mod <- estimateVec[cvsegs[[i]]][mask_po]
								p_orig <- original[cvsegs[[i]]][mask_po]
								
								err_abs<-abs(p_mod - p_orig)
								
								ttlp_mod <- sum(p_mod)
								ttlp_obs <- sum(p_orig)
								
								pcterr <- abs((ttlp_mod-ttlp_obs)/ttlp_obs)*100
								
								pcter_sum[nIx] <- pcter_sum[nIx] + pcterr
								pcter_cnt[nIx] <- pcter_cnt[nIx] + 1
								
								mae_sum[nIx] <- mae_sum[nIx] + sum(err_abs)
								mae_cnt[nIx] <- mae_cnt[nIx] + length(err_abs)
								
								nrmsep[nIx] <- nrmsep[nIx] + sum( (original - estimateVec)^2) 
							} else {
								q2[nIx] <- q2[nIx] + sum( (original - estimateVec)^2 )
							}
						}    
					} else {
						estimate <- estimate@completeObs[, index]
						original <- target[compObs, ]
						## Error of prediction, error is calculated for removed
						## elements only
						if (em == "nrmsep") {
							nrmsep <- nrmsep + sum( (original - estimate)^2)
						} else {
							q2 <- q2 + sum( (original - estimate)^2 )
						}
					}
				} ## iteration over cv segments
				
				if (fastKE) {
					if (em == "nrmsep") {
						error[pos, ] <-
								error[pos, ] + nrmsep / (nrow(set) * var(set[,index]))
					} else
						error[pos, ] <- error[pos, ] + (1 - (q2 / sum(set[, index]^2)))
				} else {
					if (em == "nrmsep") {
						error[pos, iteration] <- error[pos, iteration] + 
								nrmsep / (nrow(set) * var(set[,index]))
					} else
						error[pos, iteration] <-
								error[pos, iteration] + (1 - (q2 / sum(set[, index]^2)))
				}
			} # iteration over variables
			if (verbose) cat("\n")
			
		} #iteration over nruncv
		
		## The error is the sum over the independent cross validation runs
		error <- error / nruncv
		
		if (verbose && !fastKE)
			cat("The average", em, "for k =", iteration, "is", 
					sum(error[,iteration]) / nrow(error), "\n")
		
		## if nlpca, ppca, bpca, nipals we do not need to iterate over the
		## number of components...
		if (fastKE) break
	} # iteration over number components
	
	if (em == "nrmsep")
		avgError <- sqrt(apply(error, 2, sum) / nrow(error))
	else
		avgError <- apply(error, 2, sum) / nrow(error)
	
	ret <- list()
	if (em == "nrmsep")
		ret$bestNPcs <- evalPcs[which(avgError == min(avgError))]
	else ret$bestNPcs <- evalPcs[which(avgError == max(avgError))]
	ret$eError <- avgError
	if(em == "nrmsep") ret$variableWiseError <- sqrt(error)
	else ret$variableWiseError <- error
	ret$evalPcs <- evalPcs
	ret$variableIx <- missIx
	#print(mae_sum/mae_cnt)
	return(list(ret=ret,mae=mae_sum/mae_cnt,pcter=pcter_sum/pcter_cnt))
}

ppca_tair_npcs <- function(pca_matrix,eval_npcs,fit_npcs)
{
	pca_matrix[is.nan(pca_matrix)]<-NA
	ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=eval_npcs,scale='none',center=CENTER,cv='none',seed=SEED)	
	ppca_fit <- fitted.pcaRes(ppca_rslt,nPcs = fit_npcs)
	return(list(ppca_fit=ppca_fit[,1]))
}

ppca_tair_1xval <-function(pca_matrix)
{
	pca_matrix[is.nan(pca_matrix)]<-NA
	
	max_pc<-max_pcs(pca_matrix)
	eval_pcs<-max_pc$npcs
	ppca_mod<-max_pc$ppca_rslt
	
	ppca_k<-kEstimate_my(pca_matrix,method='ppca',evalPcs=eval_pcs:eval_pcs,nruncv=CV_NRUNS,em='nrmsep',segs=segs,verbose=FALSE,seed=SEED,threshold=threshold)

	ppca_fit <- fitted.pcaRes(ppca_mod,nPcs = eval_pcs)

	return(list(ppca_fit=ppca_fit[,1],k=ppca_k$mae[1],b=ppca_k$bias[1],mse=ppca_k$mse[1],press=ppca_k$press[1],eval_pcs=eval_pcs,npcs=eval_pcs))
}

ppca_tair_no_xval <-function(pca_matrix,means=NULL,scales=NULL,frac_obs=FRAC_OBS,max_r2cum=MAX_R2CUM,npcs=0,convThres=THRESHOLD)
{
	pca_matrix[is.nan(pca_matrix)]<-NA
	
	max_pc<-max_pcs2(pca_matrix,means,scales,frac_obs,max_r2cum,npcs,convThres)
	eval_pcs<-max_pc$npcs
	ppca_mod<-max_pc$ppca_rslt
	
	#ppca_fit <- fitted.pcaRes(ppca_mod,nPcs = eval_pcs)
	ppca_fit <- fitted(ppca_mod,nPcs = eval_pcs)
	
	if(is.vector(scales) & ! is.logical(scales))
	{
		ppca_fit <- sweep(ppca_fit,2,scales,"*")
		#print("Sweeped in std")
	}
	
	if(is.vector(means) & ! is.logical(means))
	{
		ppca_fit <- sweep(ppca_fit,2,means,"+")
		#print("Sweeped in means")
	}
	
	return(list(ppca_fit=ppca_fit[,1],eval_pcs=eval_pcs,npcs=eval_pcs))
}

ppca_tair <- function(pca_matrix)
{	
	pca_matrix[is.nan(pca_matrix)]<-NA
	threshold<-THRESHOLD
	
	max_pc<-max_pcs(pca_matrix)
	eval_pcs<-max_pc$npcs
	ppca_mod<-max_pc$ppca_rslt
	#cat("eval_pcs=",eval_pcs,"\n")
#	segs<-round(length(pca_matrix[,1])/length(which(is.na(pca_matrix[,1]))))
#	if(is.infinite(segs))
#	{
#		segs <- 3
#	}
	segs <- CV_NSEGS
	
	#cat("cv_segs=",segs,"\n")
	ppca_k<-kEstimate_my(pca_matrix,method='ppca',evalPcs=1:eval_pcs,nruncv=CV_NRUNS,em='nrmsep',segs=segs,verbose=FALSE,seed=SEED,threshold=threshold)
#	
#	#print(ppca_k)
#	
#	if (max(ppca_k$mae) > 10)
#	{
#		threshold<-1e-06
#		print("MAE seems too high.  PPCA likely did not converge well. Rerun with smaller tolerance.")
#		ppca_k<-kEstimate_my(pca_matrix,method='ppca',evalPcs=1:eval_pcs,nruncv=CV_NRUNS,em='nrmsep',segs=segs,verbose=FALSE,seed=SEED,threshold=threshold)
#	}
#	#err_difs <- diff(ppca_k$mae,lag=LAG)
#	#print(ppca_k$mae)
#	#print(err_difs)
#	#x<-min(which(err_difs > TOL))
#	#x<-which.min(ppca_k$mae)
	x<-which.min(ppca_k$press)

#	if (! is.finite(x))
#	{
#		x <- eval_pcs
#	}
	#x<-eval_pcs
	#cat("final num_pcs=",x,"\n")
	
	ppca_fit <- fitted.pcaRes(ppca_mod,nPcs = x)
	
	#return(list(ppca_fit=ppca_fit[,1],k=1,b=1,mse=1,press=1))
	return(list(ppca_fit=ppca_fit[,1],k=ppca_k$mae[x],b=ppca_k$bias[x],mse=ppca_k$mse[x],press=ppca_k$press[x],eval_pcs=eval_pcs,npcs=x))
}

max_pcs2 <- function(pca_tair,means=NULL,scales=NULL,frac_obs=FRAC_OBS,max_r2cum=MAX_R2CUM,npcs=0,convThres=THRESHOLD)
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
	
	if(VERBOSE) cat("THE CONVERGENCE THRES IS ",convThres,"\n")
	
	if (npcs != 0)
	{
		if(VERBOSE) cat("running with set num_pcs=",npcs,"\n")
		ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres)
		if(VERBOSE) print(ppca_rslt)
		return(list(npcs=npcs,ppca_rslt=ppca_rslt))
	}
	
	npcs <- round((ncol(pca_tair)-1)*frac_obs,0)
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
				if(VERBOSE) cat("testing num_pcs=",npcs,"\n")
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
		
		if(VERBOSE) print(ppca_rslt)
		if (is(ppca_rslt,"warning") || is(ppca_rslt,"simpleError"))
		{
			npcs_new <- as.numeric(strsplit(ppca_rslt$message," ")[[1]][4]) - 1
			if(VERBOSE) print(npcs_new)
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
					if(VERBOSE) cat("testing num_pcs for final=",npcs,"\n")
					fnl_ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres)
					#ncalls = ncalls + 1
					if(VERBOSE) print(fnl_ppca_rslt)
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
			
			
			if(VERBOSE) cat("adding num_pcs=",add_pcs,"\n")
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
				if(VERBOSE) cat("Removed bogus PC1 100% variance\n")
			}
		}
	}
	
	if(VERBOSE) cat("PPCA DONE. FINAL # of PCs: ",npcs,"\n")
	if(VERBOSE) print(fnl_ppca_rslt)
	return(list(npcs=npcs,ppca_rslt=fnl_ppca_rslt))
}

max_pcs <- function(pca_tair,means=NULL,scales=NULL,frac_obs=FRAC_OBS,max_r2cum=MAX_R2CUM)
{
	npcs <- round((ncol(pca_tair)-1)*frac_obs,0)
	pcs_fnd <- FALSE
	warn_fnd <- FALSE
	
	
	if(is.logical(means))
	{
		center<-means
		pca_matrix<-pca_tair
	}
	else if(is.vector(means))
	{
		center<-FALSE
		pca_matrix<-sweep(pca_tair,2,means,"-")
		#print("Sweeped out means")
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
		#print("Sweeped out std")
	}
	else
	{
		scale<-SCALE
	}
	
	#print(center)
	#print(scale)
	
	rslts <- list()
	#print(center)
	#ncalls = 0
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
				#print("YOU SAVED A CALL!")
			}
			else
			{	
				if(VERBOSE) cat("testing num_pcs=",npcs,"\n")
				ppca_rslt<-tryCatch(pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres),error=function(e) e, warning=function(w) w)
			
				#ncalls = ncalls + 1
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
		
		if(VERBOSE) print(ppca_rslt)
		if (is(ppca_rslt,"warning") || is(ppca_rslt,"simpleError"))
		{
			npcs_new <- as.numeric(strsplit(ppca_rslt$message," ")[[1]][4]) - 1
			if(VERBOSE) print(npcs_new)
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
					if(VERBOSE) cat("testing num_pcs for final=",npcs,"\n")
					fnl_ppca_rslt<-pca(pca_matrix,method='ppca',nPcs=npcs,scale=scale,center=center,cv='none',seed=SEED,threshold=convThres)
					#ncalls = ncalls + 1
					if(VERBOSE) print(fnl_ppca_rslt)
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
			
			
			if(VERBOSE) cat("adding num_pcs=",add_pcs,"\n")
			npcs <- npcs + add_pcs
			warn_fnd <- FALSE
		}
	}
	if(VERBOSE) cat("PPCA DONE. FINAL # of PCs: ",npcs,"\n")
	#print(fnl_ppca_rslt@R2cum)
	return(list(npcs=npcs,ppca_rslt=fnl_ppca_rslt))
}	


ppca_tair_python <- function(pca_tair,ymd)
{
	ppca_tair<-ppca_tair(pca_tair)
	return (list(fit_tair=ppca_tair$ppca_fit,k_tair=ppca_tair$k,b_tair=ppca_tair$b,mse_tair=ppca_tair$mse,press_tair=ppca_tair$press,eval_pcs=ppca_tair$eval_pcs,npcs=ppca_tair$npcs))
}

calc_hss <- function(obs,mod)
{
	
	
	#model_obs
	true_true <- sum(obs==1 & mod==1)
	false_false <- sum(obs==0 & mod==0)
	true_false <- sum(obs==0 & mod==1)
	false_true <- sum(obs==1 & mod==0)
	
	a <- true_true
	b <- true_false
	c <- false_true
	d <- false_false

	#special case handling
	if (a == 0.0 & c == 0.0 & b != 0)
	{
		#This means that were no observed days of rain so can't calc
		#appropriate hss. Set a = 1 to get a more appropriate hss
		a <- 1.0
	}

	
	if (b == 0.0 & d == 0.0 & c != 0.0)
	{
		#This means that there was observed rain every day so can't calc
		#appropriate hss. Set d = 1 to get a more appropriate hss
		d <- 1.0   
	}
 
	
	den <- ((a+c)*(c+d))+((a+b)*(b+d))
	
	if (den == 0.0)
	{
		#This is a perfect forecast with all true_true or false_false
		return(1.0)
	}			
	
	return ((2.0*((a*d)-(b*c)))/den)
}


hasVarChgPt <- function(vals,sig=0.0000000001)
{
	if (VERBOSE) cat("VARIANCE CHANGE POINT AT: ",cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig)),"\n")
	return(length(cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig))) > 0)
}

getVarChgPt <- function(vals,sig=0.0000000001)
{	
	aCpts <- cpts(cpt.var(vals,penalty="Asymptotic",pen.value=sig))
	if (VERBOSE) cat("VARIANCE CHANGE POINT AT: ",aCpts,"\n")
	return(aCpts)
}

		


