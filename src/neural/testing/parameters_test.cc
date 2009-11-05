/* parameters_test.cc
   Jeremy Barnes, 5 November October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the parameters class.
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
using namespace ML::Stats;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
}
