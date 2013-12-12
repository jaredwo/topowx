/*
 * topomet.h
 *
 *  Created on: Oct 27, 2011
 *      Author: jared.oyler
 */

#ifndef TOPOMET_H_
#define TOPOMET_H_

#endif /* TOPOMET_H_ */

#define NUM_MTHS 12

//#include <clapack.h>
#include <math.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics_double.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_sort_vector.h>

typedef struct
{
	int nstns;
	int ndays;
	double pt_x; /*LON*/
	double pt_y; /*LAT*/
	double pt_z; /*ELEV*/
	double *X; /*the regression design matrix col 0 = int, 1 = LON, 2 = LAT, 3 = ELEV; C row-major order*/
	double *stn_wgt_dist; /*the distance weights to use for the tair=lon+lot+elev regression*/
	double *stn_wgt_resid; /*the weights to use for interpolation of the regression */
	double *stn_obs; /*row-major where each row is a day*/

	/* Results */
	double *predictvals;

} tair_struct;

typedef struct
{
	int nobs;
	double pt_x; /*LON*/
	double pt_y; /*LAT*/
	double pt_z; /*ELEV*/
	double *X; /*the regression design matrix col 0 = int, 1 = LON, 2 = LAT, 3 = ELEV; C row-major order*/
	double *wgts; /*the weights to use for the regression*/
	double *obs;  /*the obs (y) values for the regression*/
	/* Results */
	double *predictval;

} regress_struct;

typedef struct
{
	int nrows;
	int ncols;
	int nvars; /*the number of independent variables for the regression*/
	double *pt_vals; /*the nvars independent variables for the point*/
	int apply_mean; /*Apply mean after PCA*/
	int apply_std; /*Apply standard deviation after PCA*/
	double pt_mean; /*MEAN VALUE*/
	double pt_std; /*STD VALUE*/
	double *X; /*the regression design matrix ncols*nvars; C row-major order*/
	double *wgts; /*the weights to use for the regression*/
	double *obs;  /*the obs matrix for the pca*/
	/* Results */
	double *returnvals;


} prcomp_struct;

typedef struct
{
	int nrows;
	int ncols;
	double *A;  /*the nrows*ncols matrix for the pca*/
	/* Results */
	double *pc_loads; /*the loadings for each PC ncols*ncols */
	double *pc_scores; /*the scores for each PC nrows*ncols */
	double *var_explain; /*variance explained by each PCA ncols vector */

} pca_struct;

typedef struct
{
	int nobs;
	int nstns;
	double pt_x; /*LON*/
	double pt_y; /*LAT*/
	double pt_z; /*ELEV*/
	double *X; /*the regression design matrix col 0 = int, 1 = LON, 2 = LAT, 3 = ELEV; C row-major order*/
	double *wgts; /*the weights to use for the regression*/
	double *obs;  /*the obs matrix for the pca*/
	double *thres; /*a matrix po thres values for each month and station*/
	int *mths; /*an array of mth values */
	/* Results */
	double *returnvals;

} po_interp_struct;

typedef struct
{
	int nboots; /* the number of bootstrap replicates */
	int nobs;
	double pt_x; /*LON*/
	double pt_y; /*LAT*/
	double pt_z; /*ELEV*/
	double *X; /*the regression design matrix col 0 = int, 1 = LON, 2 = LAT, 3 = ELEV; C row-major order*/
	double *wgts; /*the weights to use for the regression*/
	double *obs;  /*the obs (y) values for the regression*/
	/* Results */
	double *returnvals; /*predict val, se, ci lower, ci upper*/


} boot_struct;

typedef struct
{
	int nstns;
	int ndays;
	double pt_x; /*LON*/
	double pt_y; /*LAT*/
	double pt_z; /*ELEV*/
	double *X; /*the regression design matrix col 0 = int, 1 = LON, 2 = LAT, 3 = ELEV; C row-major order*/
	double *stn_wgt_dist; /*the distance weights to use for the tair=lon+lot+elev regression*/
	double *stn_wgt_resid; /*the weights to use for interpolation of the regression */
	double *stn_obs; /*row-major where each row is a day*/
	double *stn_sum_obs; /*the total prcp for each stn for this time period*/

	double pop_crit;

	/* Results */
	double *predictvals;

} prcp_struct;

typedef struct
{
	double *dem;
	int nrows;
	int ncols;
	int nwins;
	int npts;
	int *rows;
	int *cols;
	int *windows;
	double *topo_disect;/* Results */

} topo_disect_struct;

typedef struct
{
	int nYRows;
	int nYCols;
	int nZLen;
	double *Y;
	double *Z;
	double *z;
	double *predictVals;
	double *fitVals;

} RepRegressStruct;

int prcomp_interp(prcomp_struct* pca_str);
void bootstrap(boot_struct* boot_str);
void regress(regress_struct* regress_str);
void predict_tair(tair_struct* tair_str);
int predict_prcp(prcp_struct* prcp_str);
void calc_topo_disect(topo_disect_struct* td_struct);

void solve_multifit_linear(const gsl_matrix * X,
                      	  const gsl_vector * w,
                      	  const gsl_vector * y,
                      	  gsl_vector * c,
                      	  gsl_matrix * cov,
                      	  double *chisq, gsl_multifit_linear_workspace * work);

void setup_svd(const gsl_matrix * X,const gsl_vector * w,gsl_multifit_linear_workspace * work);


