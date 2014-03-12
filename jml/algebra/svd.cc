/* svd.cc
   Jeremy Barnes, 15 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Singular value decomposition functions, implementation.
*/

#if 0

#include "svd.h"
#include "eigenvalues.h"
#include "jml/stats/distribution.h"
#include "jml/stats/distribution_simd.h"
#include "jml/arch/exception.h"
#include <boost/timer.hpp>
#include <iostream>

using namespace std;

namespace ML {


template<class Float>
boost::tuple<distribution<Float>, boost::multi_array<Float, 2>,
             boost::multi_array<Float, 2> >
svd_impl(const boost::multi_array<Float, 2> & A, int nsv)
{
    /* We make the m x n square matrix
       [ 0  A ]
       [ AT 0 ]

       (where AT is A transposed) and take its eigenvalues.  This is basically
       because the more efficient SVD procedure has problems in calculating the
       left singular vectors when A has a large range in its singular values.
       This is also what Matlab does.

       Note that this could be optimised a lot to make use of the sparseness.
       If A is tall and skinny or short and fat, a lot of cycles will be
       wasted multiplying zero by zero.
    */

    bool profile = false;

    boost::timer t;
    double t0 = t.elapsed();

    size_t m = A.shape()[0];
    size_t n = A.shape()[1];
    size_t mnmin = std::min(m, n);

    if (nsv < 0 || nsv > mnmin)
        throw Exception("asked for more singular values than min(m,n)");

    /* Make the array

       [ 0  A ]
       [ A' 0 ]

       to find the eigenvalues with.
    */
    boost::multi_array<Float, 2> A_(m+n, m+n);
    A_.fill(0.0);
    for (unsigned i = 0;  i < m;  ++i) {
        for (unsigned j = 0;  j < n;  ++j) {
            A_[n + i][j] = A[i][j];
            A_[j][n + i] = A[i][j];
        }
    }
    
    if (profile) {
        cerr << "SVD: array construction: " << -t0 + (t.elapsed()) << endl;
        t0 = t.elapsed();
    }

    //cerr << "A_ = " << endl << A_ << endl;

    /* Get the eigeneverything. */
    distribution<Float> E;
    vector<distribution<Float> > W;
    boost::tie(E, W) = eigenvectors(A_, nsv);

    if (profile) {
        cerr << "SVD: eigenvalues: " << -t0 + (t.elapsed()) << endl;
        t0 = t.elapsed();
    }
    
    //cerr << "done eigeneverything" << endl;

    //cerr << "E = " << E << endl;
    
    //cerr << "W = " << endl;
    //for (unsigned i = 0;  i < W.size();  ++i)
    //    cerr << W[i] << endl;
    //cerr << endl;
    
    /* Note: matlab does here some special processing to only select the ones
       that are independent... not really sure so leaving it out.

       >> type svds

       ...

       % Which (left singular) vectors are already orthogonal, with norm
       % 1/sqrt(2)?
       UU = W(1:m,:)' * W(1:m,:);
       dUU = diag(UU);
       VV = W(m+(1:n),:)' * W(m+(1:n),:);
       dVV = diag(VV);
       indpos = find((d > dtol) & (abs(dUU-0.5) <= uvtol)
                     & (abs(dVV-0.5) <= uvtol));
       indpos = indpos(1:min(end,k));
       npos = length(indpos);
       U = sqrt(2) * W(1:m,indpos);
       s = d(indpos);
       V = sqrt(2) * W(m+(1:n),indpos);

       ...
    */
    
    boost::multi_array<Float, 2> U(boost::extents[nsv][m]);
    boost::multi_array<Float, 2> V(boost::extents[nsv][n]);

    for (unsigned i = 0;  i < m;  ++i)
        for (unsigned j = 0;  j < nsv;  ++j)
            U[j][i] = std::sqrt(2.0) * W[j][i];

    for (unsigned i = 0;  i < n;  ++i)
        for (unsigned j = 0;  j < nsv;  ++j)
            V[j][i] = std::sqrt(2.0) * W[j][i+m];

    if (profile) {
        cerr << "SVD: singular vectors: " << -t0 + (t.elapsed()) << endl;
        t0 = t.elapsed();
    }

    //cerr << "U = " << endl << U << endl;
    //cerr << "V = " << endl << V << endl;
    
    //boost::multi_array<Float, 2> D = diag(E);
    //D.fill(0.0);
    //for (unsigned i = 0;  i < std::min(m, n);  ++i)
    //    D[i][i] = E[i];

    //boost::multi_array<Float, 2> VT(std::min(m, n), n);
    //for (unsigned i = 0;  i < n;  ++i)
    //    for (unsigned j = 0;  j < std::min(m, n);  ++j)
    //        VT[j][i] = V[i][j];

    //cerr << "U * D * VT = " << endl << U * D * VT << endl;

    //cerr << "A = " << endl << A << endl;

    if (profile) cerr << "SVD: total: " << t.elapsed() << endl;

    return boost::make_tuple(E, U, V);
}

boost::tuple<distribution<float>, boost::multi_array<float, 2>,
             boost::multi_array<float, 2> >
svd(const boost::multi_array<float, 2> & A)
{
    return svd_impl(A, std::min(A.shape()[0], A.shape()[1]));
}

boost::tuple<distribution<double>, boost::multi_array<double, 2>,
             boost::multi_array<double, 2> >
svd(const boost::multi_array<double, 2> & A)
{
    return svd_impl(A, (A.shape()[0], A.shape()[1]));
}

boost::tuple<distribution<float>, boost::multi_array<float, 2>,
             boost::multi_array<float, 2> >
svd(const boost::multi_array<float, 2> & A, size_t nsv)
{
    return svd_impl(A, nsv);
}

boost::tuple<distribution<double>, boost::multi_array<double, 2>,
             boost::multi_array<double, 2> >
svd(const boost::multi_array<double, 2> & A, size_t nsv)
{
    return svd_impl(A, nsv);
}

} // namespace ML

#endif
