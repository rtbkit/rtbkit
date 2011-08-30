/* matrix_ops.h                                                    -*- C++ -*-
   Jeremy Barnes, 15 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Operations on matrices.
*/

#ifndef __algebra__matrix_ops_h__
#define __algebra__matrix_ops_h__

#include "jml/stats/distribution.h"
#include "jml/arch/exception.h"
#include <boost/multi_array.hpp>
#include <iostream>
#include "jml/utils/float_traits.h"
#include "jml/compiler/compiler.h"
#include "jml/arch/simd_vector.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/cache.h"

namespace boost {

template<class Float>
std::ostream &
operator << (std::ostream & stream, const boost::multi_array<Float, 2> & m)
{
    for (unsigned i = 0;  i < m.shape()[0];  ++i) {
        stream << "    [";
        for (unsigned j = 0;  j < m.shape()[1];  ++j)
            stream << ML::format(" %8.3g", m[i][j]);
        stream << " ]" << std::endl;
    }
    return stream;
}

} // namespace boost

namespace ML {

// Copy a chunk of a matrix transposed to another place
template<typename Float>
void copy_transposed(boost::multi_array<Float, 2> & A,
                     int i0, int i1, int j0, int j1)
{
    // How much cache will be needed to hold the input data?
    size_t mem = (i1 - i0) * (j1 - j0) * sizeof(float);

    // Fits in memory (with some allowance for loss): copy directly
    if (mem * 4 / 3 < l1_cache_size) {
        // 1.  Prefetch everything we need to access with non-unit stride
        //     in cache in order
        for (unsigned i = i0;  i < i1;  ++i)
            warmup_cache_all_levels(&A[i][j0], j1 - j0);

        // 2.  Do the work
        for (unsigned j = j0;  j < j1;  ++j)
            streaming_copy_from_strided(&A[j][i0], &A[i0][j], A.strides()[0],
                                        i1 - i0);
        return;
    }

    // Otherwise, we recurse
    int spliti = (i0 + i1) / 2;
    int splitj = (j0 + j1) / 2;

    // TODO: try to ensure a power of 2

    copy_transposed(A, i0, spliti, j0, splitj);
    copy_transposed(A, i0, spliti, splitj, j1);
    copy_transposed(A, spliti, i1, j0, splitj);
    copy_transposed(A, spliti, i1, splitj, j1);
}

// Copy everything above the diagonal below the diagonal of the given part
// of the matrix.  We do it by divide-and-conquer, sub-dividing the problem
// until we get something small enough to fit in the cache.
template<typename Float>
void copy_lower_to_upper(boost::multi_array<Float, 2> & A,
                         int i0, int i1)
{
    int j0 = i0;
    int j1 = i1;

    // How much cache will be needed to hold the input data?
    size_t mem = (i1 - i0) * (j1 - j0) * sizeof(float) / 2;

    // Fits in memory (with some allowance for loss): copy directly
    if (mem * 4 / 3 < l1_cache_size) {
        // 1.  Prefetch everything in cache in order
        for (unsigned i = i0;  i < i1;  ++i)
            warmup_cache_all_levels(&A[i][j0], i - j0);

        // 2.  Do the work
        for (unsigned j = i0;  j < j1;  ++j)
            streaming_copy_from_strided(&A[j][j], &A[j][j], A.strides()[0],
                                        j1 - j);


        return;
    }

    // Otherwise, we recurse
    int split = (i0 + i1) / 2;
    // TODO: try to ensure a power of 2

    /* i0+
         |\
         | \
         |  \
         |   \
       s +----\
         |    |\
         |    | \
         |    |  \
       i1+----+---+
        i0    s   i1
    */

    copy_lower_to_upper(A, i0, split);
    copy_lower_to_upper(A, split, i1);
    copy_transposed(A, split, i1, j0, split);
}

template<typename Float>
void copy_lower_to_upper(boost::multi_array<Float, 2> & A)
{
    int n = A.shape()[0];
    if (n != A.shape()[1])
        throw Exception("copy_upper_to_lower: matrix is not square");

    copy_lower_to_upper(A, 0, n);
}

template<typename Float>
boost::multi_array<Float, 2>
transpose(const boost::multi_array<Float, 2> & A)
{
    boost::multi_array<Float, 2> X(boost::extents[A.shape()[1]][A.shape()[0]]);
    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        for (unsigned j = 0;  j < A.shape()[1];  ++j)
            X[j][i] = A[i][j];
    return X;
}

template<typename Float>
boost::multi_array<Float, 2>
diag(const distribution<Float> & d)
{
    boost::multi_array<Float, 2> D(boost::extents[d.size()][d.size()]);
    for (unsigned i = 0;  i < d.size();  ++i)
        D[i][i] = d[i];
    return D;
}

template<typename Float>
std::string print_size(const boost::multi_array<Float, 2> & array)
{
    return format("%dx%d", (int)array.shape()[0], (int)array.shape()[1]);
}


/*****************************************************************************/
/* MATRIX VECTOR                                                             */
/*****************************************************************************/

template<typename FloatR, typename Float1, typename Float2>
ML::distribution<FloatR>
multiply_r(const boost::multi_array<Float1, 2> & A,
           const ML::distribution<Float2> & b)
{
    if (b.size() != A.shape()[1])
        throw Exception(format("multiply(matrix, vector): "
                               "shape (%dx%d) x (%dx1) wrong",
                               (int)A.shape()[0], (int)A.shape()[1],
                               (int)b.size()));
    ML::distribution<FloatR> result(A.shape()[0], 0.0);
    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        result[i] = SIMD::vec_dotprod_dp(&A[i][0], &b[0], A.shape()[1]);
    //for (unsigned j = 0;  j < A.shape()[1];  ++j)
    //    result[i] += b[j] * A[i][j];
    return result;
}

template<typename Float1, typename Float2>
//JML_ALWAYS_INLINE
distribution<typename float_traits<Float1, Float2>::return_type>
multiply(const boost::multi_array<Float1, 2> & A,
         const ML::distribution<Float2> & b)
{
    return multiply_r<typename float_traits<Float1, Float2>::return_type>(A, b);
}

template<typename Float1, typename Float2>
//JML_ALWAYS_INLINE
distribution<typename float_traits<Float1, Float2>::return_type>
operator * (const boost::multi_array<Float1, 2> & A,
            const ML::distribution<Float2> & b)
{
    typedef typename float_traits<Float1, Float2>::return_type FloatR;
    return multiply_r<FloatR, Float1, Float2>(A, b);
}


/*****************************************************************************/
/* VECTOR MATRIX                                                             */
/*****************************************************************************/

template<typename FloatR, typename Float1, typename Float2>
ML::distribution<FloatR>
multiply_r(const ML::distribution<Float2> & b,
           const boost::multi_array<Float1, 2> & A)
{
    if (b.size() != A.shape()[0])
        throw Exception(format("multiply(vector, matrix): "
                               "shape (1x%d) x (%dx%d) wrong",
                               (int)b.size(),
                               (int)A.shape()[0], (int)A.shape()[1]));

    ML::distribution<FloatR> result(A.shape()[1], 0.0);
#if 1 // more accurate
    std::vector<double> accum(A.shape()[1], 0.0);
    using namespace std;
    //cerr << "A.shape()[1] = " << A.shape()[1] << endl;
    //cerr << "accum = " << accum << endl;
    //for (unsigned j = 0;  j < A.shape()[1];  ++j)
    //    accum[j] = 0.0;
    for (unsigned i = 0;  i < A.shape()[0];  ++i) {
        //for (unsigned j = 0;  j < A.shape()[1];  ++j)
        //    result[j] += b[i] * A[i][j];
        SIMD::vec_add(&accum[0], b[i], &A[i][0], &accum[0], A.shape()[1]);
    }
    std::copy(accum.begin(), accum.end(), result.begin());
#else
    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        SIMD::vec_add(&result[0], b[i], &A[i][0], &result[0], A.shape()[1]);
#endif
    return result;
}

template<typename Float1, typename Float2>
//JML_ALWAYS_INLINE
distribution<typename float_traits<Float1, Float2>::return_type>
multiply(const ML::distribution<Float2> & b,
         const boost::multi_array<Float1, 2> & A)
{
    return multiply_r<typename float_traits<Float1, Float2>::return_type>(b, A);
}

template<typename Float1, typename Float2>
//JML_ALWAYS_INLINE
distribution<typename float_traits<Float1, Float2>::return_type>
operator * (const ML::distribution<Float2> & b,
            const boost::multi_array<Float1, 2> & A)
{
    typedef typename float_traits<Float1, Float2>::return_type FloatR;
    return multiply_r<FloatR, Float1, Float2>(b, A);
}


/*****************************************************************************/
/* MATRIX MATRIX                                                             */
/*****************************************************************************/

template<typename FloatR, typename Float1, typename Float2>
boost::multi_array<FloatR, 2>
multiply_r(const boost::multi_array<Float1, 2> & A,
           const boost::multi_array<Float2, 2> & B)
{
    if (A.shape()[1] != B.shape()[0])
        throw ML::Exception("Incompatible matrix sizes");

    boost::multi_array<FloatR, 2> X(boost::extents[A.shape()[0]][B.shape()[1]]);
    Float2 bentries[A.shape()[1]];
    for (unsigned j = 0;  j < B.shape()[1];  ++j) {
        for (unsigned k = 0;  k < A.shape()[1];  ++k)
            bentries[k] = B[k][j];

        for (unsigned i = 0;  i < A.shape()[0];  ++i)
            X[i][j] = SIMD::vec_dotprod_dp(&A[i][0], bentries, A.shape()[1]);

        //for (unsigned i = 0;  i < A.shape()[0];  ++i)
        //    for (unsigned k = 0;  k < A.shape()[1];  ++k)
        //        X[i][j] += A[i][k] * B[k][j];

    }
    return X;
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
multiply(const boost::multi_array<Float1, 2> & A,
         const boost::multi_array<Float2, 2> & B)
{
    return multiply_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
operator * (const boost::multi_array<Float1, 2> & A,
            const boost::multi_array<Float2, 2> & B)
{
    return multiply_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

/*****************************************************************************/
/* MULTIPLY_TRANSPOSED                                                       */
/*****************************************************************************/

// Multiply A * transpose(A)
template<typename FloatR, typename Float>
boost::multi_array<FloatR, 2>
multiply_transposed(const boost::multi_array<Float, 2> & A)
{
    int As0 = A.shape()[0];
    int As1 = A.shape()[1];

    boost::multi_array<FloatR, 2> X(boost::extents[As0][As0]);
    for (unsigned i = 0;  i < As0;  ++i) 
        for (unsigned j = 0;  j <= i;  ++j)
            X[i][j] = X[j][i] = SIMD::vec_dotprod_dp(&A[i][0], &A[j][0], As1);
    
    return X;
}

template<typename FloatR, typename Float1, typename Float2>
boost::multi_array<FloatR, 2>
multiply_transposed(const boost::multi_array<Float1, 2> & A,
                    const boost::multi_array<Float2, 2> & BT)
{
    int As0 = A.shape()[0];
    int Bs0 = BT.shape()[0];
    int As1 = A.shape()[1];

    if (A.shape()[1] != BT.shape()[1])
        throw ML::Exception("Incompatible matrix sizes");

    boost::multi_array<FloatR, 2> X(boost::extents[As0][Bs0]);
    for (unsigned j = 0;  j < Bs0;  ++j) {
        for (unsigned i = 0;  i < As0;  ++i)
            X[i][j] = SIMD::vec_dotprod_dp(&A[i][0], &BT[j][0], As1);
    }

    return X;
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
multiply_transposed(const boost::multi_array<Float1, 2> & A,
                    const boost::multi_array<Float2, 2> & B)
{
    // Special case for A * A^T
    if (&A == &B)
        return multiply_transposed<typename float_traits<Float1, Float2>::return_type>(A);

    return multiply_transposed
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

/*****************************************************************************/
/* MATRIX ADDITION                                                           */
/*****************************************************************************/

template<typename FloatR, typename Float1, typename Float2>
boost::multi_array<FloatR, 2>
add_r(const boost::multi_array<Float1, 2> & A,
           const boost::multi_array<Float2, 2> & B)
{
    if (A.shape()[0] != B.shape()[0]
        || A.shape()[1] != B.shape()[1])
        throw ML::Exception("Incompatible matrix sizes");
    
    boost::multi_array<FloatR, 2> X(boost::extents[A.shape()[0]][A.shape()[1]]);

    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        SIMD::vec_add(&A[i][0], &B[i][0], &X[i][0], A.shape()[1]);

    return X;
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
add(const boost::multi_array<Float1, 2> & A,
    const boost::multi_array<Float2, 2> & B)
{
    return add_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
operator + (const boost::multi_array<Float1, 2> & A,
            const boost::multi_array<Float2, 2> & B)
{
    return add_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}


/*****************************************************************************/
/* MATRIX SUBTRACTION                                                        */
/*****************************************************************************/

template<typename FloatR, typename Float1, typename Float2>
boost::multi_array<FloatR, 2>
subtract_r(const boost::multi_array<Float1, 2> & A,
           const boost::multi_array<Float2, 2> & B)
{
    if (A.shape()[0] != B.shape()[0]
        || A.shape()[1] != B.shape()[1])
        throw ML::Exception("Incompatible matrix sizes");
    
    boost::multi_array<FloatR, 2> X(boost::extents[A.shape()[0]][A.shape()[1]]);

    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        SIMD::vec_minus(&A[i][0], &B[i][0], &X[i][0], A.shape()[1]);
    
    return X;
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
subtract(const boost::multi_array<Float1, 2> & A,
    const boost::multi_array<Float2, 2> & B)
{
    return subtract_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
operator - (const boost::multi_array<Float1, 2> & A,
            const boost::multi_array<Float2, 2> & B)
{
    return subtract_r
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}


} // namespace ML


#endif /* __algebra__matrix_ops_h__ */

