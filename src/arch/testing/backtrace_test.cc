/* bitops_test.cc
   Jeremy Barnes, 20 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the bit operations class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "arch/backtrace.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>


using namespace ML;
using namespace std;

using boost::unit_test::test_suite;


BOOST_AUTO_TEST_CASE( test1 )
{
    backtrace(cerr);
}
