/** print_utils_test.cc                                            -*- C++ -*-
    Rémi Attab, 2 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/utils/print_utils.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_randomString )
{
    string a = randomString(6);
    string b = randomString(6);
    BOOST_CHECK_NE(a, b);
}
