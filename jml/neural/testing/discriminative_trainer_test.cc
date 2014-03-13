/* discriminative_trainer_test.cc
   Jeremy Barnes, 28 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for discriminative training of a neural netrowk.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "jml/neural/dense_layer.h"
#include "jml/utils/testing/serialize_reconstitute_include.h"
#include <boost/assign/list_of.hpp>
#include <limits>

using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( xor_test )
{
}
