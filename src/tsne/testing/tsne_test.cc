/* tsne_test.cc
   Jeremy Barnes, 16 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Unit tests for the tsne software.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "tsne/tsne.h"
#include <boost/assign/list_of.hpp>
#include <limits>
#include <boost/test/floating_point_comparison.hpp>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

template<typename X>
X sqr(X x)
{
    return x * x;
}

BOOST_AUTO_TEST_CASE( test_vectors_to_distances )
{
    distribution<float> vecs[4] = {
        boost::assign::list_of<float>(1.0)(0.0)(-1.0)(0.0),
        boost::assign::list_of<float>(1.0)(1.0)(-1.0)(0.0),
        boost::assign::list_of<float>(-1.0)(2.0)(-2.0)(0.0),
        boost::assign::list_of<float>(1.0)(0.0)(-1.0)(4.0) };

    boost::multi_array<float, 2> vectors(boost::extents[4][4]);

    for (unsigned i = 0;  i < 4;  ++i)
        for (unsigned j = 0;  j < 4;  ++j) 
            vectors[i][j] = vecs[i].at(j);

    boost::multi_array<float, 2> distances
        = vectors_to_distances(vectors);

    double tolerance = 0.00001;

    for (unsigned i = 0;  i < 4;  ++i)
        for (unsigned j = 0;  j < 4;  ++j) 
            BOOST_CHECK_CLOSE(distances[i][j],
                              sqr(float((vecs[i] - vecs[j]).two_norm())),
                              tolerance);
}
