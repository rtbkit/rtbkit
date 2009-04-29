/* tick_counter_test.cc                                            -*- C++ -*-
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of tick counter functionality.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "arch/tick_counter.h"

#include <boost/test/unit_test.hpp>

using namespace ML;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test1 )
{
    uint64_t before = ticks();
    uint64_t after = ticks();

    BOOST_CHECK(after > before);
}

BOOST_AUTO_TEST_CASE( test2 )
{
    double overhead = calc_ticks_overhead();

    BOOST_CHECK(overhead > 1.0);
    BOOST_CHECK(overhead < 100.0);
}
