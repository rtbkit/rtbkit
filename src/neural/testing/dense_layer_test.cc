/* dense_layer_test.cc
   Jeremy Barnes, 28 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the dense layer class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils/parse_context.h"
#include "utils/file_functions.h"
#include "utils/guard.h"
#include "db/persistent.h"
#include "db/compact_size_types.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <sstream>
#include <boost/multi_array.hpp>
#include "algebra/matrix_ops.h"
#include "stats/distribution.h"
#include "boosting/thread_context.h"
#include "neural/dense_layer.h"
#include "utils/testing/serialize_reconstitute_include.h"

using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;


BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer1 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_ZERO, context);
    test_serialize_reconstitute(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer2 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_INPUT, context);
    test_serialize_reconstitute(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer3 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_DENSE, context);
    test_serialize_reconstitute(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer_double )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 200, 400, TF_TANH, MV_DENSE, context);
    test_serialize_reconstitute(layer);
}

