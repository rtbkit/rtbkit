/* svd.h                                                           -*- C++ -*-
   Jeremy Barnes, 15 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Singular value decompositions.
*/

#ifndef __algebra__svd_h__
#define __algebra__svd_h__


#include "jml/stats/distribution.h"
#include <boost/multi_array.hpp>
#include <boost/tuple/tuple.hpp>


namespace ML {

/*****************************************************************************/
/* SVD                                                                       */
/*****************************************************************************/

/** Calculates the "economy size" SVD of the matrix A, including the left and
    right singular vectors.  The output has the relationship

    \f[
        A = U\Sigma V^T
    \f]
    
    where \f$\Sigma\f$ is a diagonal matrix of the singular values:

    \f[
        \Sigma = \left[
            \begin{array}{cccc}
              \sigma_0  &    0   & \cdots &    0   \\
               0   &   \sigma_1  & \cdots &    0   \\
            \vdots & \vdots & \ddots & \vdots \\
               0   &    0   &    0   &   \sigma_n 
            \end{array}
            \right]
    \f]

    Apropos:

    \code
    boost::multi_array<float, 2> A;

    ...

    distribution<float> E;
    boost::multi_array<float, 2> U, V;
    boost::tie(E, U, V) = svd(A);
    \endcode

    \param A        the matrix of which to take the SVD

    \returns E      a vector of the singular values, in order from the
                    highest value to the lowest value
    \returns U      the left-singular vectors
    \returns V      the right-singular vectors.

    \pre            A has full rank
    \post           \f$A = U \Sigma V^T\f$

    Again, a wrapper around ARPACK.  Note that the ARPACK++ manual gives
    pretty good instructions on how to do all of this, although it neglects
    to mention that U and V need to be multiplied by \f$\sqrt{2}\f$!
*/
boost::tuple<distribution<float>, boost::multi_array<float, 2>,
             boost::multi_array<float, 2> >
svd(const boost::multi_array<float, 2> & A);

boost::tuple<distribution<double>, boost::multi_array<double, 2>,
             boost::multi_array<double, 2> >
svd(const boost::multi_array<double, 2> & A);

/** Same as above, but calculates only the first \p n singular values.
 */
boost::tuple<distribution<float>, boost::multi_array<float, 2>,
             boost::multi_array<float, 2> >
svd(const boost::multi_array<float, 2> & A, size_t n);

boost::tuple<distribution<double>, boost::multi_array<double, 2>,
             boost::multi_array<double, 2> >
svd(const boost::multi_array<double, 2> & A, size_t n);

} // namespace ML


#endif /* __algebra__svd_h__ */

