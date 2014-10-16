/* bitops_test.cc
   Jeremy Barnes, 20 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test of the bit operations class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/backtrace.h"
#include "jml/arch/exception.h"

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
    cerr << "\n\nan explicit backtrace:\n";
    backtrace(cerr);
}

/* ensures that the char * version of backtrace behaves like snprintf */
BOOST_AUTO_TEST_CASE( test2 )
{
    /* buffer overflows */
    {
        char buffer[12];
        ::memset(buffer, 0xaf, sizeof(buffer));
        size_t bufferSize = 10;

        size_t written = backtrace(buffer, bufferSize);
        BOOST_CHECK_GE(written, bufferSize);
        BOOST_CHECK_EQUAL(buffer[9], '\0');
        BOOST_CHECK_EQUAL(buffer[10], (char) 0xaf);
        BOOST_CHECK_EQUAL(buffer[11], (char) 0xaf);
    }

    /* buffer with correct size */
    {
        char buffer[10240];
        ::memset(buffer, 0xaf, sizeof(buffer));

        size_t written = backtrace(buffer, sizeof(buffer));
        BOOST_CHECK_LT(written, sizeof(buffer));
        ::fprintf(stderr, "written: %lu\n", written);

        /* buffer ends with '\0' */
        BOOST_CHECK_EQUAL(buffer[written], '\0');

        /* buffer is kept untouched after last written byte */
        BOOST_CHECK_EQUAL(buffer[written+1], (char) 0xaf);

        ::fprintf(stderr, "\n\nconst char * backtrace:\n%s\n", buffer);
    }
}

BOOST_AUTO_TEST_CASE (exception_backtrace)
{
    cerr << "\n\nan exception with a backtrace:\n";
    try {
        throw ML::Exception("this exception produces a backtrace");
    }
    catch (...) {
    }
}
