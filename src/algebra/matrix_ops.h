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

#include "stats/distribution.h"
#include "arch/exception.h"
#include <boost/multi_array.hpp>
#include <iostream>
#include "utils/float_traits.h"
#include "compiler/compiler.h"

namespace ML {


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
    boost::multi_array<Float, 2> D(d.size(), d.size());
    for (unsigned i = 0;  i < d.size();  ++i)
        D[i][i] = d[i];
    return D;
}

template<class Float>
std::ostream &
operator << (std::ostream & stream, const boost::multi_array<Float, 2> & m)
{
    for (unsigned i = 0;  i < m.shape()[0];  ++i) {
        stream << "    [";
        for (unsigned j = 0;  j < m.shape()[1];  ++j)
            stream << format(" %8.3g", m[i][j]);
        stream << " ]" << std::endl;
    }
    return stream;
}

template<typename FloatR, typename Float1, typename Float2>
ML::distribution<FloatR>
multiply(const boost::multi_array<Float1, 2> & A,
         const ML::distribution<Float2> & b)
{
    if (b.size() == A.shape()[0]) {
        ML::distribution<FloatR> result(A.shape()[1], 0.0);
        for (unsigned i = 0;  i < A.shape()[0];  ++i)
            for (unsigned j = 0;  j < A.shape()[1];  ++j)
                result[j] += b[i] * A[i][j];
        return result;
    }
    else if (b.size() == A.shape()[1]) {
        ML::distribution<FloatR> result(A.shape()[0], 0.0);
        for (unsigned i = 0;  i < A.shape()[0];  ++i)
            for (unsigned j = 0;  j < A.shape()[1];  ++j)
                result[i] += b[j] * A[i][j];
        return result;
    }
    else throw ML::Exception("Incompatible matrix sizes");
}

template<typename Float1, typename Float2>
JML_ALWAYS_INLINE
distribution<typename float_traits<Float1, Float2>::return_type>
multiply(const boost::multi_array<Float1, 2> & A,
         const ML::distribution<Float2> & b)
{
    return multiply<typename float_traits<Float1, Float2>::return_type>(A, b);
}

template<typename Float1, typename Float2>
JML_ALWAYS_INLINE
distribution<double>
//ML::distribution<typename float_traits<Float1, Float2>::return_type>
operator * (const boost::multi_array<Float1, 2> & A,
            const ML::distribution<Float2> & b)
{
    typedef typename float_traits<Float1, Float2>::return_type FloatR;
    return multiply<FloatR, Float1, Float2>(A, b);
}

template<typename FloatR, typename Float1, typename Float2>
boost::multi_array<FloatR, 2>
multiply(const boost::multi_array<Float1, 2> & A,
         const boost::multi_array<Float2, 2> & B)
{
    if (A.shape()[1] != B.shape()[0])
        throw ML::Exception("Incompatible matrix sizes");

    boost::multi_array<FloatR, 2> X(A.shape()[0], B.shape()[1]);
    for (unsigned i = 0;  i < A.shape()[0];  ++i)
        for (unsigned j = 0;  j < B.shape()[1];  ++j)
            for (unsigned k = 0;  k < A.shape()[1];  ++k)
                X[i][j] += A[i][k] * B[k][j];
    return X;
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
multiply(const boost::multi_array<Float1, 2> & A,
         const boost::multi_array<Float2, 2> & B)
{
    return multiply
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

template<typename Float1, typename Float2>
boost::multi_array<typename float_traits<Float1, Float2>::return_type, 2>
operator * (const boost::multi_array<Float1, 2> & A,
            const boost::multi_array<Float2, 2> & B)
{
    return multiply
        <typename float_traits<Float1, Float2>::return_type, Float1, Float2>
        (A, B);
}

} // namespace ML


#endif /* __algebra__matrix_ops_h__ */

