/* rmse_test.cc
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the RMSE functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "jml/stats/rmse.h"
#include <boost/assign/list_of.hpp>

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test )
{
    distribution<float> targets
        = boost::assign::list_of<float>(1.0)(0.0);
    distribution<float> outputs 
        = boost::assign::list_of<float>(2.0)(1.0);

    BOOST_CHECK_EQUAL(calc_rmse(outputs, outputs), 0.0);

    BOOST_CHECK_EQUAL(calc_rmse(outputs, targets), 1.0);

    distribution<float> weights(2, 0.1);
    BOOST_CHECK_EQUAL(calc_rmse(outputs, targets, weights), 1.0);

    weights.push_back(1.0);
    BOOST_CHECK_THROW(calc_rmse(outputs, targets, weights), Exception);
}
