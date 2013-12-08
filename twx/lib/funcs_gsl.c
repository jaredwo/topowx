/*
 * funcs_gsl.c
 *
 *  Created on: Nov 25, 2011
 *      Author: jared.oyler
 */
#include "wxTopo.h"

void solve_multifit_linear(const gsl_matrix * X,
                      const gsl_vector * w,
                      const gsl_vector * y,
                      gsl_vector * c,
                      gsl_matrix * cov,
                      double *chisq, gsl_multifit_linear_workspace * work)
{

    const size_t n = X->size1;
    const size_t p = X->size2;

    size_t i, j;

    gsl_matrix *A = work->A;
    gsl_matrix *QSI = work->QSI;
    gsl_vector *t = work->t;
    gsl_vector *xt = work->xt;
    gsl_vector *D = work->D;

    for (i = 0; i < n; i++)
      {
        double wi = gsl_vector_get (w, i);
        double yi = gsl_vector_get (y, i);
        if (wi < 0)
          wi = 0;
        gsl_vector_set (t, i, sqrt (wi) * yi);
      }

    gsl_blas_dgemv (CblasTrans, 1.0, A, t, 0.0, xt);

    /* Solution */

    gsl_blas_dgemv (CblasNoTrans, 1.0, QSI, xt, 0.0, c);

    /* Unscale the balancing factors */

    gsl_vector_div (c, D);

    /* Form covariance matrix cov = (Q S^-1) (Q S^-1)^T */

    for (i = 0; i < p; i++)
      {
        gsl_vector_view row_i = gsl_matrix_row (QSI, i);
        double d_i = gsl_vector_get (D, i);

        for (j = i; j < p; j++)
          {
            gsl_vector_view row_j = gsl_matrix_row (QSI, j);
            double d_j = gsl_vector_get (D, j);
            double s;

            gsl_blas_ddot (&row_i.vector, &row_j.vector, &s);

            gsl_matrix_set (cov, i, j, s / (d_i * d_j));
            gsl_matrix_set (cov, j, i, s / (d_i * d_j));
          }
      }

    /* Compute chisq, from residual r = y - X c */

    {
      double r2 = 0;

      for (i = 0; i < n; i++)
        {
          double yi = gsl_vector_get (y, i);
          double wi = gsl_vector_get (w, i);
          gsl_vector_const_view row = gsl_matrix_const_row (X, i);
          double y_est, ri;
          gsl_blas_ddot (&row.vector, c, &y_est);
          ri = yi - y_est;
          r2 += wi * ri * ri;
        }

      *chisq = r2;
    }
}

void setup_svd(const gsl_matrix * X,const gsl_vector * w,gsl_multifit_linear_workspace * work)
{
	double tol = GSL_DBL_EPSILON;
	int balance = 1;

    const size_t n = X->size1;
    const size_t p = X->size2;

    size_t i, j, p_eff;

    gsl_matrix *A = work->A;
    gsl_matrix *Q = work->Q;
    gsl_matrix *QSI = work->QSI;
    gsl_vector *S = work->S;
    gsl_vector *xt = work->xt;
    gsl_vector *D = work->D;

    /* Scale X,  A = sqrt(w) X */

    gsl_matrix_memcpy (A, X);

    for (i = 0; i < n; i++)
      {
        double wi = gsl_vector_get (w, i);

        if (wi < 0)
          wi = 0;

        {
          gsl_vector_view row = gsl_matrix_row (A, i);
          gsl_vector_scale (&row.vector, sqrt (wi));
        }
      }

    /* Balance the columns of the matrix A if requested */

    if (balance)
      {
        gsl_linalg_balance_columns (A, D);
      }
    else
      {
        gsl_vector_set_all (D, 1.0);
      }

    /* Decompose A into U S Q^T */

    gsl_linalg_SV_decomp_mod (A, QSI, Q, S, xt);

    /* Scale the matrix Q,  Q' = Q S^-1 */

    gsl_matrix_memcpy (QSI, Q);

    {
      double alpha0 = gsl_vector_get (S, 0);
      p_eff = 0;

      for (j = 0; j < p; j++)
        {
          gsl_vector_view column = gsl_matrix_column (QSI, j);
          double alpha = gsl_vector_get (S, j);

          if (alpha <= tol * alpha0) {
            alpha = 0.0;
          } else {
            alpha = 1.0 / alpha;
            p_eff++;
          }

          gsl_vector_scale (&column.vector, alpha);
        }
    }
}
