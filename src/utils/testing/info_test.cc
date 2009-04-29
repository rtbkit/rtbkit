/* info_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the info functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils/info.h"
#include "utils/environment.h"

#include <boost/test/unit_test.hpp>
#include <iostream>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
    BOOST_CHECK_EQUAL(userid_to_username(0), "root");
    BOOST_CHECK_EQUAL(userid_to_username(getuid()),
                      Environment::instance()["USER"]);
    BOOST_CHECK(num_cpus() > 0 && num_cpus() < 1024);
    cerr << "num_cpus = " << num_cpus() << endl;
}
