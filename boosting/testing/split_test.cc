/* decision_tree_xor_test.cc
   Jeremy Barnes, 25 February 2008
   Copyright (c) 2008 Jeremy Barnes.  All rights reserved.

   Test of the decision tree class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <limits>

#include "jml/boosting/split.h"
#include "jml/boosting/feature_set.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
//#include <boost/math/>

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;


BOOST_AUTO_TEST_CASE( test_split_less )
{
    float NaN = std::numeric_limits<float>::quiet_NaN();
    float infinity = std::numeric_limits<float>::infinity();

    Split split(MISSING_FEATURE, 20, Split::LESS);
    BOOST_CHECK_EQUAL(split.apply(18), true);
    BOOST_CHECK_EQUAL(split.apply(20), false);
    BOOST_CHECK_EQUAL(split.apply(21), false);
    BOOST_CHECK_EQUAL(split.apply(NaN), (int)MISSING);
    BOOST_CHECK_EQUAL(split.apply(-infinity), true);
    BOOST_CHECK_EQUAL(split.apply(infinity), false);
}

BOOST_AUTO_TEST_CASE( test_split_equal )
{
    float NaN = std::numeric_limits<float>::quiet_NaN();
    float infinity = std::numeric_limits<float>::infinity();

    Split split(MISSING_FEATURE, 20, Split::EQUAL);
    BOOST_CHECK_EQUAL(split.apply(18), false);
    BOOST_CHECK_EQUAL(split.apply(20), true);
    BOOST_CHECK_EQUAL(split.apply(21), false);
    BOOST_CHECK_EQUAL(split.apply(NaN), (int)MISSING);
    BOOST_CHECK_EQUAL(split.apply(-infinity), false);
    BOOST_CHECK_EQUAL(split.apply(infinity), false);
}

