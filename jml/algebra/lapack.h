/* lapack.h                                                        -*- C++ -*-
   Jeremy Barnes, 5 November 2004
   Copyright (c) 2004 Jeremy Barnes  All rights reserved.
   $Source$

   Interface to lapack.
*/

#ifndef __boosting__lapack_h__
#define __boosting__lapack_h__

#include <boost/multi_array.hpp>
#include "jml/boosting/config.h"
#include <numeric>


namespace ML {
namespace LAPack {

/** Convert a matrix to a fortran compatible version. */
template<typename Float>
boost::multi_array<Float, 2>
fortran(const boost::multi_array<Float, 2> & A)
{
    /* We just have to make sure that each of the dimensions is at least
       one.  The copying was probably necessary as most of the LAPACK
       routines destroy their arguments.
    */
    boost::multi_array<Float, 2> X(A.shape()[0], std::max<size_t>(A.shape()[1], 1));
    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        for (unsigned j = 0;  j < A.shape()[1];  ++j)
            X[i][j] = A[i][j];
    return X;
}

/** Information on the LAPACK library.  Normally used internally.  See the man
    page for ilaenv for details. */
int ilaenv(int ispec, const char * routine, const char * opts,
           int n1, int n2, int n3, int n4);

/** Full rank least squares problem, float version. */
int gels(char trans, int m, int n, int nrhs, float * A, int lda, float * B,
         int ldb);

/** Full rank least squares problem, double version. */
int gels(char trans, int m, int n, int nrhs, double * A, int lda, double * B,
         int ldb);

/** Generalized least squares problem, generalised to work with rank-deficient
    matrices,  float version.  See man page for sgelsd for details. */
int gelsd(int m, int n, int nrhs, float * A, int lda, float * B, int ldb,
          float * s, float rcond, int & rank);

/** Generalized least squares problem, generalised to work
    with rank-deficient matrices, double version.  See man page for dgelsd for
    details. */
int gelsd(int m, int n, int nrhs, double * A, int lda, double * B, int ldb,
          double * s, double rcond, int & rank);

/** Generalized linear constrained least squares problem, float version.  See
    man page for sgglse for details. */
int gglse(int m, int n, int p, float * A, int lda, float * B, int ldb,
          float * c, float * d, float * result);

/** Generalized linear constrained least squares problem, double version.  See
    man page for dgglse for details. */
int gglse(int m, int n, int p, double * A, int lda, double * B, int ldb,
          double * c, double * d, double * result);

/** Convert matrix to bidiagonal form. */
int gebrd(int m, int n, double * A, int lda,
          double * D, double * E, double * tauq, double * taup);

/** SVD of a bidiagonal matrix. */
int orgbr(const char * vect, int m, int n, int k,
          double * A, int lda, const double * tau);

/** Extract orthoginal matrices from the output of gebrd. */
int bdsdc(const char * uplo, const char * compq, int n,
          double * D, double * E,
          double * U, int ldu,
          double * VT, int ldvt,
          double * Q, int * iq);

/** SVD. */
int gesvd(const char * jobu, const char * jobvt, int m, int n,
          double * A, int lda, double * S, double * U, int ldu,
          double * VT, int ldvt);

/** SVD with improved algorithm */
int gesdd(const char * jobz, int m, int n,
          float * A, int lda, float * S, float * U, int ldu,
          float * vt, int ldvt);

int gesdd(const char * jobz, int m, int n,
          double * A, int lda, double * S, double * U, int ldu,
          double * vt, int ldvt);

/** Solve a system of linear equations. */
int gesv(int n, int nrhs, double * A, int lda, int * pivots, double * B,
         int ldb);

/** Cholesky factorization. */
int spotrf(char uplo, int n, float * A, int lda);

int dpotrf(char uplo, int n, double * A, int lda);

inline int potrf(char uplo, int n, float * A, int lda)
{
    return spotrf(uplo, n, A, lda);
}

inline int potrf(char uplo, int n, double * A, int lda)
{
    return dpotrf(uplo, n, A, lda);
}

/** QR factorization with column pivoting. */
int geqp3(int m, int n, float * A, int lda, int * jpvt, float * tau);

int geqp3(int m, int n, double * A, int lda, int * jpvt, double * tau);


/** Generalized matrix multiply */
int gemm(char transa, char transb, int m, int n, int k, float alpha,
         const float * A, int lda, const float * b, int ldb,
         float beta, float * C, int ldc);

int gemm(char transa, char transb, int m, int n, int k, double alpha,
         const double * A, int lda, const double * b, int ldb,
         double beta, double * C, int ldc);

} // namespace LAPack
} // namespace ML

#endif /* __boosting__lapack_h__ */
