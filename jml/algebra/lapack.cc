/* lapack.cc
   Jeremy Barnes, 5 November 2004
   Copyright (c) 2004 Jeremy Barnes  All rights reserved.
   $Source$

   LAPACK interface.
*/

#include "lapack.h"
#include <vector>
#include <boost/scoped_array.hpp>
#include <cmath>
#include "jml/arch/threads.h"
#include "jml/arch/exception.h"
#include <iostream>


using namespace std;


extern "C" {
    /* Definitions of the FORTRAN routines that implement the functionality. */

    /* Information on the linear algebra environment. */
    int ilaenv_(const int * ispec, const char * routine, const char * opts,
                const int * n1, const int * n2, const int * n3, const int * n4);

    /* Information about the machine precision */
    float slamch_(const char * param);
    double dlamch_(const char * param);

    /* Constrained least squares. */
    void sgglse_(const int * m, const int * n, const int * p,
                 float * A, const int * lda,
                 float * B, const int * ldb,
                 float * c, float * d, float * result,
                 float * workspace, const int * workspace_size, int * info);
    
    /* Constrained least squares. */
    void dgglse_(const int * m, const int * n, const int * p,
                 double * A, const int * lda,
                 double * B, const int * ldb,
                 double * c, double * d, double * result,
                 double * workspace, const int * workspace_size, int * info);

    /* Rank deficient least squares. */
    void sgelsd_(const int * m, const int * n, const int * nrhs,
                 float * A, const int * lda, float * B, const int * ldb,
                 float * S, const float * rcond, int * rank,
                 float * workspace, const int * workspace_size,
                 int * iworkspace, int * info);
    
    /* Rank deficient least squares. */
    void dgelsd_(const int * m, const int * n, const int * nrhs,
                 double * A, const int * lda, double * B, const int * ldb,
                 double * S, const double * rcond, int * rank,
                 double * workspace, const int * workspace_size,
                 int * iworkspace, int * info);
    
    /* Full rank least squares. */
    void sgels_(char * trans, const int * m, const int * n, const int * nrhs,
                float * A, const int * lda, float * B, const int * ldb,
                float * workspace, int * workspace_size, int * info);
    
    /* Full rank least squares. */
    void dgels_(char * trans, const int * m, const int * n, const int * nrhs,
                double * A, const int * lda, double * B, const int * ldb,
                double * workspace, int * workspace_size, int * info);

    /* Convert general matrix to bidiagonal form. */
    void dgebrd_(const int * m, const int * n, double * A, const int * lda,
                 double * D, double * E, double * tauq, double * taup,
                 double * workspace, int * workspace_size, int * info);

    /* Extract orthogonal matrix from output of xgebrd. */
    void dorgbr_(const char * vect, const int * m, const int * n,
                 const int * k, double * A, const int * lda, const double * tau,
                 double * work, int * workspace_size, int * info);

    /* SVD of bidiagonal form. */
    void dbdsdc_(const char * uplo, const char * compq, const int * n,
                 double * D, double * E,
                 double * U, const int * ldu,
                 double * VT, const int * ldvt,
                 double * Q, int * iq, double * workspace, int * iworkspace,
                 int * info);

    /* SVD of matrix. */
    void dgesvd_(const char * jobu, const char * jobvt, const int * m,
                 const int * n,
                 double * A, const int * lda,
                 double * S,
                 double * U, const int * ldu,
                 double * VT, const int * ldvt,
                 double * workspace, int * workspace_size, int * info);

    /* Better SVD of a matrix. */
    void sgesdd_(const char * jobz, const int * m, const int * n,
                 float * A, const int * lda,
                 float * S,
                 float * U, const int * ldu,
                 float * vt, const int * ldvt,
                 float * workspace, int * workspace_size,
                 int * iwork, int * info);

    void dgesdd_(const char * jobz, const int * m, const int * n,
                 double * A, const int * lda,
                 double * S,
                 double * U, const int * ldu,
                 double * vt, const int * ldvt,
                 double * workspace, int * workspace_size,
                 int * iword, int * info);

    /* Solve a system of linear equations. */
    void dgesv_(const int * n, const int * nrhs, double * A, const int * lda,
                int * pivots, double * B, const int * ldb, int * info);

    /* Cholesky factorization */
    void spotrf_(const char * uplo, const int * n, float * A, const int * lda,
                 int * info);

    /* Cholesky factorization */
    void dpotrf_(const char * uplo, const int * n, double * A, const int * lda,
                 int * info);

    /* QR factorization with partial pivoting */
    void sgeqp3_(const int * m, const int * n, float * A, const int * lda,
                 int * jpvt, float * tau, float * work, const int * lwork,
                 int * info);

    /* QR factorization with partial pivoting */
    void dgeqp3_(const int * m, const int * n, double * A, const int * lda,
                 int * jpvt, double * tau, double * work, const int * lwork,
                 int * info);

    /* Matrix multiply */
    void sgemm_(const int * m, const int * n, const int * k, const float * alpha,
                const float * A, const int * lda, const float * b,
                const int * ldb, const float * beta, float * c, const int * ldc,
                int * info);

    /* Matrix multiply */
    void dgemm_(const int * m, const int * n, const int * k, const double * alpha,
                const double * A, const int * lda, const double * b,
                const int * ldb, const double * beta, double * c, const int * ldc,
                int * info);

    /* Elementary reflector.  Used to detect version 3.2 of the LAPACK.  Most
       important thing is that if n < 0, it will return zero in tau. */
    void slarfp_(const int * n, float * alpha, float * X, const int * incx,
                 float * tau);
} // extern "C"

namespace ML {
namespace LAPack {

namespace {

// TODO: Determine if we need to lock the LAPack based upon its version
// For lapack 3.1.x, yes we do
// for lapack 3.2.x, no we don't

// To determine, we look for the routine slarfp which was added in version 3.2.
// If we find it (our weak reference is ignored), then we know that we are in
// version 3.2.  Otherwise, we know that there is a previous version.

bool lapack_version_3_2_or_later()
{
    // We assume that these days we are using 3.2 or later.  The slarfp
    // function appears to have been removed from 3.3 and so this code
    // no longer has the desired effect.
    return true;

    float tau, alpha, x;
    // 234 is the special value for the slarfp_dummy function above
    int n = -234, incx;

    slarfp_(&n, &alpha, &x, &incx, &tau);


    // If the dummy slarfp function ran, it will return 1.0, which means that
    // we are using a version of lapack 3.1 or lower.  If the lapack
    // 3.2 version ran, it will return 0.0.
    if (tau == 1.0) return false;  // dummy version ran
    if (tau == 0.0) return true;
    
    throw Exception("unable to interpret result of slarfp");
}


static bool need_lock = true;

struct Init {
    Init()
    {
        need_lock = !lapack_version_3_2_or_later();
        
        // ilaenv isn't thread safe (it can return different results if it's
        // called for the first time twice from two different threads; see
        // http://icl.cs.utk.edu/lapack-forum/archives/lapack/msg00342.html

        int ispec = 10;
        const char * routine = "BONUS";
        const char * opts = "T";
        int n1 = 0, n2 = 0, n3 = 0, n4 = 0;
        
        // Call this one once so that it's thread safe thereafter
        ilaenv_(&ispec, routine, opts, &n1, &n2, &n3, &n4);

        dlamch_("e");
        slamch_("e");
    }
} init;


static Lock lock;

struct Lapack_Guard {
    Lapack_Guard()
    {
        if (!need_lock) return;
        lock.lock();
    }

    ~Lapack_Guard()
    {
        if (!need_lock) return;
        lock.unlock();
    }
};

} // file scope

int ilaenv(int ispec, const char * routine, const char * opts,
           int n1, int n2, int n3, int n4)
{
    return ilaenv_(&ispec, routine, opts, &n1, &n2, &n3, &n4); 
}

int gels(char trans, int m, int n, int nrhs, float * A, int lda, float * B,
         int ldb)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    float ws_return;
    
    /* Find out how much to allocate. */
    sgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &ws_return,
           &workspace_size, &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    //cerr << "gels: asked for " << workspace_size << " workspace" << endl;

    boost::scoped_array<float> workspace(new float[workspace_size]);

    /* Perform the computation. */
    sgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb,
           workspace.get(), &workspace_size, &info);
    
    return info;
}

int gels(char trans, int m, int n, int nrhs, double * A, int lda, double * B,
         int ldb)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb, &ws_return,
           &workspace_size, &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    //cerr << "gels: asked for " << workspace_size << " workspace" << endl;

    boost::scoped_array<double> workspace(new double[workspace_size]);

    /* Perform the computation. */
    dgels_(&trans, &m, &n, &nrhs, A, &lda, B, &ldb,
           workspace.get(), &workspace_size, &info);
    
    return info;
}

int gelsd(int m, int n, int nrhs, float * A, int lda, float * B, int ldb,
          float * S, float rcond, int & rank)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    float ws_return;

    int smallsz = ilaenv(9, "SGELSD", "", m, n, nrhs, -1); 
    
    //cerr << "smallsz = " << smallsz << endl;

    int minmn = std::min(m, n);
    int nlvl = std::max(0, (int)(log2(minmn/(smallsz + 1))) + 1);
    int intwss = 3 * minmn * nlvl + 11 * minmn;

    boost::scoped_array<int> iwork(new int[intwss]);

    /* Find out how much to allocate. */
    sgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank, &ws_return,
            &workspace_size, iwork.get(), &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    //cerr << "gels: asked for " << workspace_size << " workspace" << endl;

    boost::scoped_array<float> workspace(new float[workspace_size]);

    /* Perform the computation. */
    sgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank,
            workspace.get(), &workspace_size, iwork.get(), &info);
    
    return info;
}

int gelsd(int m, int n, int nrhs, double * A, int lda, double * B, int ldb,
          double * S, double rcond, int & rank)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;

    int smallsz = ilaenv(9, "DGELSD", "", m, n, nrhs, -1); 
    
    //cerr << "smallsz = " << smallsz << endl;

    int minmn = std::min(m, n);
    int nlvl = std::max(0, (int)(log2(minmn/(smallsz + 1))) + 1);
    int intwss = 3 * minmn * nlvl + 11 * minmn;

    boost::scoped_array<int> iwork(new int[intwss]);

    /* Find out how much to allocate. */
    dgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank, &ws_return,
            &workspace_size, iwork.get(), &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    //cerr << "gels: asked for " << workspace_size << " workspace" << endl;

    boost::scoped_array<double> workspace(new double[workspace_size]);

    /* Perform the computation. */
    dgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rcond, &rank,
            workspace.get(), &workspace_size, iwork.get(), &info);
    
    return info;
}

int gglse(int m, int n, int p, float * A, int lda, float * B, int ldb,
          float * c, float * d, float * result)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    float ws_return;
    
    /* Find out how much to allocate. */
    sgglse_(&m, &n, &p, A, &lda, B, &ldb, c, d, result, &ws_return,
             &workspace_size, &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    /* Get the workspace. */
    boost::scoped_array<float> workspace(new float[workspace_size]);

    /* Perform the computation. */
    sgglse_(&m, &n, &p, A, &lda, B, &ldb, c, d, result,
             &workspace[0], &workspace_size, &info);

    return info;
}

int gglse(int m, int n, int p, double * A, int lda, double * B, int ldb,
          double * c, double * d, double * result)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dgglse_(&m, &n, &p, A, &lda, B, &ldb, c, d, result, &ws_return,
             &workspace_size, &info);

    if (info != 0) return info;
    workspace_size = (int)ws_return;

    //cerr << "gglse: asked for " << workspace_size << " workspace" << endl;

    boost::scoped_array<double> workspace(new double[workspace_size]);

    /* Perform the computation. */
    dgglse_(&m, &n, &p, A, &lda, B, &ldb, c, d, result,
             workspace.get(), &workspace_size, &info);
    
    return info;
}

int gebrd(int m, int n, double * A, int lda,
          double * D, double * E, double * tauq, double * taup)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, &ws_return,
            &workspace_size, &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;

    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    /* Perform the computation. */
    dgebrd_(&m, &n, A, &lda, D, E, tauq, taup, workspace.get(),
            &workspace_size, &info);
    
    return info;
}

int orgbr(const char * vect, int m, int n, int k,
          double * A, int lda, const double * tau)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dorgbr_(vect, &m, &n, &k, A, &lda, tau, &ws_return, &workspace_size,
            &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;

    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    /* Perform the computation. */
    dorgbr_(vect, &m, &n, &k, A, &lda, tau, workspace.get(), &workspace_size,
            &info);
    
    return info;
}

int bdsdc(const char * uplo, const char * compq, int n,
          double * D, double * E,
          double * U, int ldu,
          double * VT, int ldvt,
          double * Q, int * iq)
{
    Lapack_Guard guard;

    int workspace_size;
    switch (*compq) {
    case 'N': workspace_size = 2 * n;  break;
    case 'P': workspace_size = 6 * n;  break;
    case 'I': workspace_size = 3 * n * n + 2 * n;  break;
    default: return -2;  // error with param 2 (compq)
    }

    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    int iworkspace_size = 7 * n;

    boost::scoped_array<int> iworkspace(new int[iworkspace_size]);

    int info;

    /* Perform the computation. */
    dbdsdc_(uplo, compq, &n, D, E, U, &ldu, VT, &ldvt, Q, iq, workspace.get(),
            iworkspace.get(), &info);

    return info;
}

int gesvd(const char * jobu, const char * jobvt, int m, int n,
          double * A, int lda, double * S, double * U, int ldu,
          double * VT, int ldvt)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dgesvd_(jobu, jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, &ws_return,
            &workspace_size, &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;
    
    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    /* Perform the computation. */
    dgesvd_(jobu, jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt,
            workspace.get(), &workspace_size, &info);
    
    return info;
}

int gesdd(const char * jobz, int m, int n,
          float * A, int lda, float * S, float * U, int ldu,
          float * vt, int ldvt)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    float ws_return;
    
    int iwork[8 * std::min(m, n)];

    /* Find out how much to allocate. */
    sgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, &ws_return,
            &workspace_size, iwork, &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;
    
    boost::scoped_array<float> workspace(new float[workspace_size]);
    
    /* Perform the computation. */
    sgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, workspace.get(),
            &workspace_size, iwork, &info);
    
    return info;
}

int gesdd(const char * jobz, int m, int n,
          double * A, int lda, double * S, double * U, int ldu,
          double * vt, int ldvt)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    int iwork[8 * std::min(m, n)];

    /* Find out how much to allocate. */
    dgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, &ws_return,
            &workspace_size, iwork, &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;
    
    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    /* Perform the computation. */
    dgesdd_(jobz, &m, &n, A, &lda, S, U, &ldu, vt, &ldvt, workspace.get(),
            &workspace_size, iwork, &info);
    
    return info;
}

int gesv(int n, int nrhs, double * A, int lda, int * pivots, double * B,
         int ldb)
{
    Lapack_Guard guard;

    int info = 0;
    dgesv_(&n, &nrhs, A, &lda, pivots, B, &ldb, &info);
    return info;
}

int spotrf(char uplo, int n, float * A, int lda)
{
    Lapack_Guard guard;

    int info = 0;
    spotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

int dpotrf(char uplo, int n, double * A, int lda)
{
    Lapack_Guard guard;

    int info = 0;
    dpotrf_(&uplo, &n, A, &lda, &info);
    return info;
}

int geqp3(int m, int n, float * A, int lda, int * jpvt, float * tau)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    float ws_return;
    
    /* Find out how much to allocate. */
    sgeqp3_(&m, &n, A, &lda, jpvt, tau, &ws_return, &workspace_size,
            &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;
    
    boost::scoped_array<float> workspace(new float[workspace_size]);
    
    /* Perform the computation. */
    sgeqp3_(&m, &n, A, &lda, jpvt, tau, workspace.get(), &workspace_size,
            &info);
    
    return info;
}

int geqp3(int m, int n, double * A, int lda, int * jpvt, double * tau)
{
    Lapack_Guard guard;

    int info = 0;
    int workspace_size = -1;
    double ws_return;
    
    /* Find out how much to allocate. */
    dgeqp3_(&m, &n, A, &lda, jpvt, tau, &ws_return, &workspace_size,
            &info);
    
    if (info != 0) return info;
    workspace_size = (int)ws_return;
    
    boost::scoped_array<double> workspace(new double[workspace_size]);
    
    /* Perform the computation. */
    dgeqp3_(&m, &n, A, &lda, jpvt, tau, workspace.get(), &workspace_size,
            &info);
    
    return info;
}

} // namespace LAPack
} // namespace ML

