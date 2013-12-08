/* 
 ============================================================================
 Name        : wxTopo.c
 Author      : Jared Oyler
 Version     :
 Copyright   : Your copyright notice
 Description : wxTopo C functions
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include "wxTopo.h"
#include "lapacke.h"
#include <gsl/gsl_cdf.h>

#define min(a,b) ((a)>(b)?(b):(a))
#define PCT_PC_THRESHOLD 0.99
#define PO_THRES_MIN 0.0001
#define PO_THRES_MAX 0.9997

/*Performs a singular value decomposition for principal component analysis. To save processing,
 * it does not calculate the unneeded U (the left singular vectors). Uses dgesvd from LAPACK which
 * is much faster than the GSL svd function.
 *
 * Params:
 * gsl_matrix* A: the PCA matrix
 * gsl_matrix* V: a ncols*ncols matrix to hold the right singular vectors (i.e--principal components)
 * gsl_vector* S: a ncols vector to hold the singular values
 * int nrows: the number of rows in A
 * int ncols: the number of cols in A
 *
 * Returns:
 * double info: if > 0, svd did not converge
 *
 *
 * make all
Building file: ../src/funcs_gsl.c
Invoking: GCC C Compiler
gcc -I/home/jared.oyler/local/include -I/usr/include -O3 -Wall -c -fmessage-length=0 -fPIC -DHAVE_INLINE -MMD -MP -MF"src/funcs_gsl.d" -MT"src/funcs_gsl.d" -o"src/funcs_gsl.o" "../src/funcs_gsl.c"
Finished building: ../src/funcs_gsl.c

Building file: ../src/wxTopo.c
Invoking: GCC C Compiler
gcc -I/home/jared.oyler/local/include -I/usr/include -O3 -Wall -c -fmessage-length=0 -fPIC -DHAVE_INLINE -MMD -MP -MF"src/wxTopo.d" -MT"src/wxTopo.d" -o"src/wxTopo.o" "../src/wxTopo.c"
Finished building: ../src/wxTopo.c

Building target: libwxTopo
Invoking: GCC C Linker
gcc -L/home/jared.oyler/Desktop/lapack-3.4.1 -L/usr/lib64 -L/usr/lib64/atlas -lm -shared -o"libwxTopo"  ./src/funcs_gsl.o ./src/wxTopo.o   -lgsl -llapacke -llapack -lf77blas -lcblas -latlas -lgfortran
Finished building target: libwxTopo
 *
 *
 * */
double svd_for_pca(gsl_matrix* A, gsl_matrix *V, gsl_vector *S, int nrows, int ncols)
{
	int m = nrows; //# of rows
	int n = ncols; //# of cols
	int lda = n;
	int ldu = m;
	int ldvt = n;

	double info;
	double *superb = malloc((min(m,n)-1) * sizeof(double));
	double *s = S->data;
	double *vt = V->data;
	double u[1] = {1}; //u not used malloc(ldu * m * sizeof(double));
	double *a = A->data;

	info = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'N', 'A', m, n, a, lda,
							s, u, ldu, vt, ldvt, superb );

	//dgesvd returns V transposed, so transpose it back
	gsl_matrix_transpose(V);

	free(superb);

	return(info);
}

int pca_gwpca(pca_struct* pca_str)
{
	int nrows = pca_str->nrows;
	int ncols = pca_str->ncols;
	int x;
	double sum;

	gsl_matrix_view A = gsl_matrix_view_array(pca_str->A,nrows,ncols);
	gsl_matrix *A2 = gsl_matrix_alloc(nrows,ncols);
	gsl_matrix_view V = gsl_matrix_view_array(pca_str->pc_loads,ncols,ncols);
	gsl_matrix_view pc_scores = gsl_matrix_view_array(pca_str->pc_scores,nrows,ncols);
	gsl_vector_view S = gsl_vector_view_array(pca_str->var_explain,ncols);

	gsl_matrix_memcpy(A2,&A.matrix);

	//Use LINPACK svd function instead of GSL gsl_linalg_SV_decomp; it's much faster. Send in copied
	//version of A because even though the LINPACK svd functions says A is not overwritten, it appears to be.
	if (svd_for_pca(A2,&V.matrix,&S.vector,nrows,ncols) > 0)
	{
		//SVD did not converge
		gsl_matrix_free(A2);
		return(1);
	}

	//Square singular values and divide by n - 1 to get variance
	gsl_vector_mul(&S.vector,&S.vector);
//	gsl_vector_scale(&S.vector,1.0/(nrows - 1.0));

	//Sum variance and then divide by sum to get % variance explained by each PC
	sum = 0;
	for (x=0 ; x < ncols ; ++x)
	{
		sum = sum + gsl_vector_get(&S.vector,x);
	}
	gsl_vector_scale(&S.vector,1.0/sum);

	//Get PC scores
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A.matrix,&V.matrix,0.0,&pc_scores.matrix);

	gsl_matrix_free(A2);

	return(0);
}

int pca_basic(pca_struct* pca_str)
{
	int nrows = pca_str->nrows;
	int ncols = pca_str->ncols;
	int x;
	double sum;

	gsl_matrix_view A = gsl_matrix_view_array(pca_str->A,nrows,ncols);
	gsl_matrix *A2 = gsl_matrix_alloc(nrows,ncols);
	gsl_matrix_view V = gsl_matrix_view_array(pca_str->pc_loads,ncols,ncols);
	gsl_matrix_view pc_scores = gsl_matrix_view_array(pca_str->pc_scores,nrows,ncols);
	gsl_vector_view S = gsl_vector_view_array(pca_str->var_explain,ncols);

	gsl_matrix_memcpy(A2,&A.matrix);

	//gsl_linalg_SV_decomp(&A.matrix,V,S,work);
	//gsl_linalg_SV_decomp_mod(&A.matrix,work2,V,S,work);
	//Use LINPACK svd function instead of GSL gsl_linalg_SV_decomp; it's much faster. Send in copied
	//version of A because even though the LINPACK svd functions says A is not overwritten, it appears to be.
	if (svd_for_pca(A2,&V.matrix,&S.vector,nrows,ncols) > 0)
	{
		//SVD did not converge
		gsl_matrix_free(A2);
		return(1);
	}

	//Square singular values and divide by n - 1 to get variance
	gsl_vector_mul(&S.vector,&S.vector);
	gsl_vector_scale(&S.vector,1.0/(nrows - 1.0));

	//Sum variance and then divide by sum to get % variance explained by each PC
	sum = 0;
	for (x=0 ; x < ncols ; ++x)
	{
		sum = sum + gsl_vector_get(&S.vector,x);
	}
	gsl_vector_scale(&S.vector,1.0/sum);

	//Get PC scores
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A.matrix,&V.matrix,0.0,&pc_scores.matrix);

	gsl_matrix_free(A2);

	return(0);
}

int repRegress(RepRegressStruct* aRepRegressStr)
{
	int nXRows = aRepRegressStr->nYRows;
	int nXCols = aRepRegressStr->nYCols;
	int nZLen = aRepRegressStr->nZLen;
	int day;
	int ndays = nXRows;

	gsl_matrix_view X = gsl_matrix_view_array(aRepRegressStr->Y,nXRows,nXCols);
	gsl_vector_view z = gsl_vector_view_array(aRepRegressStr->z,nZLen);
	gsl_vector_view y;

	for (day=0 ; day < ndays ; ++day)
	{
		y = gsl_matrix_row(&X.matrix,day);
		gsl_blas_ddot(&z.vector,&y.vector,&aRepRegressStr->returnvals[day]);
	}

	return(0);
}

int prcomp_interp(prcomp_struct* pca_str)
{
	int nrows = pca_str->nrows;
	int ncols = pca_str->ncols;
	int nvars = pca_str->nvars;
	int x,i,npcs;
	double cumsum, sum, ptload_predict, tss, r2;

	gsl_matrix_view X = gsl_matrix_view_array(pca_str->X,ncols,nvars+1); /*add 1 for intercept in design matrix */
	gsl_vector_view w = gsl_vector_view_array(pca_str->wgts,ncols);
	gsl_vector_view pt_vals = gsl_vector_view_array(pca_str->pt_vals,nvars);

	gsl_matrix_view A = gsl_matrix_view_array(pca_str->obs,nrows,ncols);
	gsl_matrix *A2 = gsl_matrix_alloc(nrows,ncols);
	gsl_matrix *V = gsl_matrix_alloc(ncols,ncols);
	gsl_matrix *Vsub; //holds subset of PCs in V
	gsl_vector *S = gsl_vector_alloc(ncols);
	gsl_matrix *PCs;
	gsl_vector_view stnloads;
	gsl_vector *ptloads;
	gsl_vector_view results = gsl_vector_view_array(pca_str->returnvals,pca_str->nrows);

	gsl_matrix_memcpy(A2,&A.matrix);

	//gsl_linalg_SV_decomp(&A.matrix,V,S,work);
	//gsl_linalg_SV_decomp_mod(&A.matrix,work2,V,S,work);
	//Use LINPACK svd function instead of GSL gsl_linalg_SV_decomp; it's much faster. Send in copied
	//version of A because even though the LINPACK svd functions says A is not overwritten, it appears to be.
	if (svd_for_pca(A2,V,S,nrows,ncols) > 0)
	{
		//SVD did not converge
		gsl_matrix_free(V);
		gsl_matrix_free(A2);
		gsl_vector_free(S);

		return(1);

	}

	//Square singular values and divide by n - 1 to get variance
	gsl_vector_mul(S,S);
	gsl_vector_scale(S,1.0/(nrows - 1.0));

	//Sum variance and then divide by sum to get % variance explained by each PC
	sum = 0;
	for (x=0 ; x < ncols ; ++x)
	{
		sum = sum + gsl_vector_get(S,x);
	}
	gsl_vector_scale(S,1.0/sum);


	//gsl_vector_fprintf(stdout,S,"%f");
	//Use the # of PCs that account for at least PCT_PC_THRESHOLD of the variability
	x = 0;
	cumsum = 0;
	while (cumsum < PCT_PC_THRESHOLD)
	{
		cumsum = cumsum + gsl_vector_get(S,x);
		x++;
	}
	npcs = x;

	//Used to hold the predicted loadings at the point
	ptloads = gsl_vector_alloc(npcs);
	//A subset of n PCs
	Vsub = gsl_matrix_alloc(ncols,npcs);
	//PC scores
	PCs = gsl_matrix_alloc(nrows,npcs);

	//Data for linear regression
	gsl_vector *c = gsl_vector_alloc(nvars+1);
	gsl_matrix *cov = gsl_matrix_alloc(nvars+1,nvars+1);
	double chisq;
	gsl_multifit_linear_workspace *lmwork = gsl_multifit_linear_alloc(ncols,nvars+1);


	for (x=0 ; x < npcs ; ++x)
	{
		stnloads = gsl_matrix_column(V,x);

		//Copy column into Vsub
		gsl_matrix_set_col(Vsub,x,&stnloads.vector);

		//Predict loading at pt for this PC
		gsl_multifit_wlinear(&X.matrix,&w.vector,&stnloads.vector,c,cov,&chisq,lmwork);

		//Use coefficients to get predicted loading
		ptload_predict = c->data[0];

		for (i=0 ; i < nvars ; ++i)
		{
			ptload_predict = ptload_predict + (c->data[i+1]*gsl_vector_get(&pt_vals.vector,i));
		}

		//Get R2 for this model
		//tss = gsl_stats_wtss(w.vector.data,w.vector.stride,stnloads.vector.data,stnloads.vector.stride,stnloads.vector.size);
		//r2 = 1.0 - (chisq/tss);
		//printf("A R2: %f\n",r2);
		//fflush(stdout);

		gsl_vector_set(ptloads,x,ptload_predict);
	}

	//Get daily PC scores
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &A.matrix,Vsub,0.0,PCs);

	//Use predicted scores and predicted loadings to get prediction values
	gsl_blas_dgemv(CblasNoTrans,1.0,PCs,ptloads,0.0,&results.vector);

	if(pca_str->apply_std)
	{
		//incorporate back standard deviation
		gsl_vector_scale(&results.vector,pca_str->pt_std);
	}

	if(pca_str->apply_mean)
	{
		//add back mean value
		gsl_vector_add_constant(&results.vector,pca_str->pt_mean);
	}

	gsl_multifit_linear_free(lmwork);
	gsl_vector_free(c);
	gsl_matrix_free(cov);
	gsl_matrix_free(V);
	gsl_matrix_free(Vsub);
	gsl_matrix_free(A2);
	gsl_vector_free(S);
	gsl_matrix_free(PCs);
	gsl_vector_free(ptloads);

	return(0);
}

void bootstrap(boot_struct* boot_str)
{

	gsl_matrix_view X = gsl_matrix_view_array(boot_str->X,boot_str->nobs,4);
	//gsl_matrix *X2 = gsl_matrix_alloc(boot_str->nobs,4);
	//gsl_matrix_memcpy(X2,&X.matrix);
	gsl_matrix *Xt = gsl_matrix_alloc(4,boot_str->nobs);
	gsl_matrix_transpose_memcpy(Xt,&X.matrix);
	gsl_vector_view w = gsl_vector_view_array(boot_str->wgts,boot_str->nobs);
	gsl_vector_view y = gsl_vector_view_array(boot_str->obs,boot_str->nobs);
	gsl_vector *c = gsl_vector_alloc(4);

	gsl_matrix *x1 = gsl_matrix_alloc(1,4);
	gsl_matrix_set(x1,0,0,1.0);
	gsl_matrix_set(x1,0,1,boot_str->pt_x);
	gsl_matrix_set(x1,0,2,boot_str->pt_y);
	gsl_matrix_set(x1,0,3,boot_str->pt_z);
	gsl_matrix *x1t = gsl_matrix_alloc(4,1);
	gsl_matrix_transpose_memcpy(x1t,x1);

	gsl_matrix *cov = gsl_matrix_alloc(4,4);
	gsl_vector *r = gsl_vector_alloc(boot_str->nobs); //residuals
	double chisq;
	double df = boot_str->nobs - 4; //degrees of freedom
	gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(boot_str->nobs,4);
	double rsum = 0;
	double mse;
	int i;

	//Create diagonal weights matrix
    gsl_matrix * W = gsl_matrix_alloc(w.vector.size,w.vector.size);
    gsl_vector_view diag = gsl_matrix_diagonal(W);
    gsl_matrix_set_all(W, 0.0); //or whatever number you like
    gsl_vector_memcpy(&diag.vector, &w.vector);

	gsl_multifit_wlinear(&X.matrix,&w.vector,&y.vector,c,cov,&chisq,work);
	gsl_multifit_linear_residuals(&X.matrix,&y.vector,c,r);
	gsl_vector_mul(r,r);
	gsl_vector_mul(r,&w.vector);

	for (i=0; i < boot_str->nobs; ++i)
	{
		rsum = rsum + gsl_vector_get(r,i);
	}
	mse = rsum/df;

	//spred <- sqrt(mse*(1+(x1%*%solve(t(X)%*%W%*%X)%*%t(x1))))
	gsl_matrix *m1 = gsl_matrix_alloc(4,boot_str->nobs);
	gsl_matrix *m2 = gsl_matrix_alloc(4,4);
	gsl_matrix *m3 = gsl_matrix_alloc(1,4);
	gsl_matrix *m4 = gsl_matrix_alloc(1,1);
	gsl_matrix *inv = gsl_matrix_alloc(4,4);
	gsl_vector *b = gsl_vector_alloc(4);
	gsl_vector *x = gsl_vector_alloc(4);
	gsl_permutation *p = gsl_permutation_alloc(4);

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,Xt,W,0.0,m1);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,m1,&X.matrix,0.0,m2);

	int signum;
	gsl_linalg_LU_decomp(m2,p,&signum);

	for (i=0; i < 4; ++i)
	{
		gsl_vector_set_zero(b);
		gsl_vector_set(b,i,1.0);
		gsl_linalg_LU_solve(m2,p,b,x);
		gsl_matrix_set_col(inv,i,x);
	}

	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,x1,inv,0.0,m3);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0,m3,x1t,0.0,m4);

	double spred = sqrt(mse * (1.0 + gsl_matrix_get(m4,0,0)));

	//printf("spred=%f\n",spred);

	/*Mean Value*/
	boot_str->returnvals[0] = c->data[0] + c->data[1]*boot_str->pt_x + c->data[2]*boot_str->pt_y + c->data[3]*boot_str->pt_z;

	//95% confidence intervals
	//printf("mean value=%f\n",boot_str->returnvals[0]);
//	printf("upper-t=%f\n",gsl_cdf_tdist_Q(0.975,df));

	double lwr = boot_str->returnvals[0] - gsl_cdf_tdist_Qinv(0.025,df)*spred;
	double upr = boot_str->returnvals[0] + -gsl_cdf_tdist_Qinv(0.975,df)*spred;

	/* Standard Error */
	boot_str->returnvals[1] = spred;

	/* 95% Confidence Interval */
	boot_str->returnvals[2] = lwr;
	boot_str->returnvals[3] = upr;

	gsl_multifit_linear_free(work);
	gsl_vector_free(c);
	gsl_matrix_free(cov);
	gsl_vector_free(r);
	//gsl_matrix_free(X2);
	gsl_matrix_free(Xt);
	gsl_matrix_free(x1);
	gsl_matrix_free(x1t);
	gsl_matrix_free(W);
	gsl_matrix_free(m1);
	gsl_matrix_free(m2);
	gsl_matrix_free(m3);
	gsl_matrix_free(m4);
	gsl_matrix_free(inv);
	gsl_vector_free(b);
	gsl_vector_free(x);
	gsl_permutation_free(p);
}

void regress(regress_struct* regress_str)
{
	gsl_matrix_view X = gsl_matrix_view_array(regress_str->X,regress_str->nobs,4);
	gsl_vector_view w = gsl_vector_view_array(regress_str->wgts,regress_str->nobs);
	gsl_vector_view y = gsl_vector_view_array(regress_str->obs,regress_str->nobs);
	gsl_vector *c = gsl_vector_alloc(4);
	gsl_matrix *cov = gsl_matrix_alloc(4,4);
	double chisq;
	gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(regress_str->nobs,4);

	gsl_multifit_wlinear(&X.matrix,&w.vector,&y.vector,c,cov,&chisq,work);

	regress_str->predictval[0] = c->data[0] + c->data[1]*regress_str->pt_x + c->data[2]*regress_str->pt_y + c->data[3]*regress_str->pt_z;

	gsl_multifit_linear_free(work);
	gsl_vector_free(c);
	gsl_matrix_free(cov);

}


void predict_tair(tair_struct* tair_str)
{
	int nstns = tair_str->nstns;
	int ndays = tair_str->ndays;
	double pt_x = tair_str->pt_x;
	double pt_y = tair_str->pt_y;
	double pt_z = tair_str->pt_z;
	double chisq;
	double resid_interp;
	int day;

	gsl_matrix_view X = gsl_matrix_view_array(tair_str->X,nstns,4);
	gsl_vector_view w_dist = gsl_vector_view_array(tair_str->stn_wgt_dist,nstns);
	gsl_vector_view w_resid = gsl_vector_view_array(tair_str->stn_wgt_resid,nstns);
	gsl_matrix_view stn_obs = gsl_matrix_view_array(tair_str->stn_obs,ndays,nstns);
	gsl_multifit_linear_workspace *lm_ws = gsl_multifit_linear_alloc(nstns,4);
	gsl_vector *resid = gsl_vector_alloc(nstns);
	gsl_vector *coef = gsl_vector_alloc(4);
	gsl_matrix *cov = gsl_matrix_alloc(4,4);
	gsl_vector_view y;

	setup_svd(&X.matrix,&w_dist.vector,lm_ws);

	for (day=0 ; day < ndays ; ++day)
	{
		y = gsl_matrix_row(&stn_obs.matrix,day);

		solve_multifit_linear(&X.matrix,&w_dist.vector,&y.vector,coef,cov,&chisq,lm_ws);

		gsl_multifit_linear_residuals(&X.matrix,&y.vector,coef,resid);
		resid_interp = gsl_stats_wmean(w_resid.vector.data,w_resid.vector.stride,resid->data,resid->stride,nstns);
		tair_str->predictvals[day] = (coef->data[0] + coef->data[1]*pt_x + coef->data[2]*pt_y + coef->data[3]*pt_z) - resid_interp;
	}

	gsl_multifit_linear_free(lm_ws);
	gsl_vector_free(resid);
	gsl_vector_free(coef);
	gsl_matrix_free(cov);


}

int predict_prcp(prcp_struct* prcp_str)
{

	int nstns = prcp_str->nstns;
	int ndays = prcp_str->ndays;
	double pt_x = prcp_str->pt_x;
	double pt_y = prcp_str->pt_y;
	double pt_z = prcp_str->pt_z;
	double chisq;
	double resid_interp;
	double ttl_prcp_interp; /* the interpolated total prcp for the time period*/
	double daily_prcp_interp; /* the interpolated prcp for a single day*/
	int day,i; /*counters */
	double pop; /* daily prcp occurence probability */
	double popcrit = prcp_str->pop_crit; /* the pop critical value to determine if there is prcp for the day */
	double stn_prcp_val; /* daily prcp value for a stn */
	double val_sum,wgt_sum; /* temporary variables for a weighted average */
	double daily_prcp_ttl; /* the total prcp for the time period from summing daily interpolated values */
	int err = 0;

	gsl_matrix_view X = gsl_matrix_view_array(prcp_str->X,nstns,4);
	gsl_vector_view w_dist = gsl_vector_view_array(prcp_str->stn_wgt_dist,nstns);
	gsl_vector_view w_resid = gsl_vector_view_array(prcp_str->stn_wgt_resid,nstns);
	gsl_matrix_view stn_obs = gsl_matrix_view_array(prcp_str->stn_obs,ndays,nstns);
	gsl_vector_view stn_sum_obs = gsl_vector_view_array(prcp_str->stn_sum_obs,nstns);
	gsl_vector_view predict_vals = gsl_vector_view_array(prcp_str->predictvals,ndays);
	gsl_multifit_linear_workspace *lm_ws = gsl_multifit_linear_alloc(nstns,4);
	gsl_vector *resid = gsl_vector_alloc(nstns);
	gsl_vector *coef = gsl_vector_alloc(4);
	gsl_matrix *cov = gsl_matrix_alloc(4,4);
	gsl_vector_view stn_prcp_day;
	gsl_vector *stn_poccur_day = gsl_vector_alloc(nstns);

	setup_svd(&X.matrix,&w_dist.vector,lm_ws);

	/*build regression model for: total prcp = lon + lat + elev and interpolate total prcp for month*/
	solve_multifit_linear(&X.matrix,&w_dist.vector,&stn_sum_obs.vector,coef,cov,&chisq,lm_ws);
	gsl_multifit_linear_residuals(&X.matrix,&stn_sum_obs.vector,coef,resid);
	resid_interp = gsl_stats_wmean(w_resid.vector.data,w_resid.vector.stride,resid->data,resid->stride,nstns);
	ttl_prcp_interp = (coef->data[0] + coef->data[1]*pt_x + coef->data[2]*pt_y + coef->data[3]*pt_z) - resid_interp;
	if (ttl_prcp_interp < 0) ttl_prcp_interp = 0.0; /* make sure prcp is not < 0.0; */

	if (ttl_prcp_interp > 0)
	{
		daily_prcp_ttl = 0;
		for (day=0 ; day < ndays ; ++day)
		{
			stn_prcp_day = gsl_matrix_row(&stn_obs.matrix,day);

			for (i=0 ; i < nstns ; ++i)
			{
				if (gsl_vector_get(&stn_prcp_day.vector,i) > 0)
				{
					gsl_vector_set(stn_poccur_day,i,1.0);
				}
				else
				{
					gsl_vector_set(stn_poccur_day,i,0.0);
				}

			}

			pop = gsl_stats_wmean(w_resid.vector.data,w_resid.vector.stride,stn_poccur_day->data,stn_poccur_day->stride,nstns);

			if (pop > popcrit)
			{
				val_sum = 0;
				wgt_sum = 0;
				for (i=0 ; i < nstns ; ++i)
				{
					stn_prcp_val = gsl_vector_get(&stn_prcp_day.vector,i);
					if (stn_prcp_val > 0)
					{
						val_sum+=stn_prcp_val/gsl_vector_get(&stn_sum_obs.vector,i);
						wgt_sum+=gsl_vector_get(&w_resid.vector,i);
					}
				}
				daily_prcp_interp = (val_sum/wgt_sum)*ttl_prcp_interp;
				gsl_vector_set(&predict_vals.vector,day,daily_prcp_interp);
				daily_prcp_ttl+=daily_prcp_interp;
			}
		}
		if (daily_prcp_ttl == 0.0)
		{
			/*Total prcp was interpolated to be > 0 by no days had prcp. Mark as error and keep all daily values = 0*/
			err = 1;
			/*printf("WARNING: Total prcp = %f, but no prcp days.\n",ttl_prcp_interp);*/
		}
		else
		{
			gsl_vector_scale(&predict_vals.vector,ttl_prcp_interp/daily_prcp_ttl);
		}

	}

	gsl_multifit_linear_free(lm_ws);
	gsl_vector_free(resid);
	gsl_vector_free(coef);
	gsl_vector_free(stn_poccur_day);
	gsl_matrix_free(cov);

	return err;
}


void calc_topo_disect(topo_disect_struct* td_struct)
{
	gsl_matrix_view dem = gsl_matrix_view_array(td_struct->dem,td_struct->nrows,td_struct->ncols);
	gsl_vector *tds = gsl_vector_alloc(td_struct->nwins);
	gsl_matrix_view win_vals;
	int* wins = td_struct->windows;
	int x,i,r,c;
	int step,str_row,end_row,str_col,end_col,nwinr,nwinc;
	double min_elev,max_elev,pt_elev,a_elev;
	double td_sum;

	for (x=0 ; x < td_struct->npts ; ++x)
	{
		pt_elev = gsl_matrix_get(&dem.matrix,td_struct->rows[x],td_struct->cols[x]);
		td_sum = 0;

		for (i=0 ; i < td_struct->nwins ; ++i)
		{
			step = (wins[i]-1.0)/2.0;

			str_row = td_struct->rows[x]-step;
			end_row = td_struct->rows[x]+step+1;
			str_col = td_struct->cols[x]-step;
			end_col = td_struct->cols[x]+step+1;

			if (str_row < 0) str_row = 0;
			if (str_col < 0) str_col = 0;
			if (end_row > td_struct->nrows) end_row = td_struct->nrows;
			if (end_col > td_struct->ncols) end_col = td_struct->ncols;

			nwinr = end_row-str_row;
			nwinc = end_col-str_col;

			win_vals = gsl_matrix_submatrix(&dem.matrix,str_row,str_col,nwinr,nwinc);

			min_elev = 99999.0;
			max_elev = -99999.0;
			for (r=0 ; r < nwinr ; ++r)
			{
				for (c=0 ; c < nwinc ; ++c)
				{
					a_elev = gsl_matrix_get(&win_vals.matrix,r,c);

					if (! isnan(a_elev))
					{

						if (a_elev < min_elev)
						{
							min_elev = a_elev;
						}

						if (a_elev > max_elev)
						{
							max_elev = a_elev;
						}
					}
				}
			}

			if (min_elev != max_elev)
			{
				td_sum+=((pt_elev-min_elev)/(max_elev-min_elev));
			}
		}

		td_struct->topo_disect[x] = td_sum;
	}
	gsl_vector_free(tds);
}


 /* Main program */
//int svd_test()
//{
//	int m = 6; //# of rows
//	int n = 5; //# of cols
//	int lda = n;
//	int ldu = m;
//	int ldvt = n;
//
//	double info;
//	double superb[min(m,n)-1];
//	double s[n], u[ldu*m], vt[ldvt*n];
//	double a[30] = {
//		8.79,  9.93,  9.83, 5.45,  3.16,
//		6.11,  6.91,  5.04, -0.27,  7.98,
//	   -9.15, -7.93,  4.86, 4.85,  3.01,
//		9.57,  1.64,  8.83, 0.74,  5.80,
//	   -3.49,  4.02,  9.80, 10.00,  4.27,
//		9.84,  0.15, -8.99, -6.02, -5.31
//	};
//        /* Executable statements */
//        printf( "LAPACKE_dgesvd (row-major, high-level) Example Program Results\n" );
//        /* Compute SVD */
//        info = LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'A', 'A', m, n, a, lda,
//                        s, u, ldu, vt, ldvt, superb );
//        /* Check for convergence */
//        if( info > 0 ) {
//                printf( "The algorithm computing SVD failed to converge.\n" );
//                exit( 1 );
//        }
//        /* Print singular values */
//        print_matrix( "Singular values", 1, n, s, 1 );
//        /* Print left singular vectors */
//        print_matrix( "Left singular vectors (stored columnwise)", m, n, u, ldu );
//        /* Print right singular vectors */
//        print_matrix( "Right singular vectors (stored rowwise)", n, n, vt, ldvt );
//        exit( 0 );
//} /* End of LAPACKE_dgesvd Example */
//
//extern void print_matrix( char* desc, int m, int n, double* a, int lda );
//
///* Auxiliary routine: printing a matrix */
//void print_matrix( char* desc, int m, int n, double* a, int lda ) {
//        int i, j;
//        printf( "\n %s\n", desc );
//        for( i = 0; i < m; i++ ) {
//                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
//                printf( "\n" );
//        }
//}

