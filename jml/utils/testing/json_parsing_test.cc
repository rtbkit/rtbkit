/* json_parsing_test.cc
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Test for the environment functions.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/utils/json_parsing.h"
#include "jml/utils/info.h"

#include <boost/test/unit_test.hpp>
#include <boost/test/auto_unit_test.hpp>
#include <math.h>

using namespace ML;

using boost::unit_test::test_suite;

void testUnsigned(const std::string & str, unsigned long long expected)
{
    Parse_Context context(str, str.c_str(), str.c_str() + str.size());
    auto val = expectJsonNumber(context);
    context.expect_eof();
    BOOST_CHECK_EQUAL(val.uns, expected);
    BOOST_CHECK_EQUAL(val.type, JsonNumber::UNSIGNED_INT);
}

void testSigned(const std::string & str, long long expected)
{
    Parse_Context context(str, str.c_str(), str.c_str() + str.size());
    auto val = expectJsonNumber(context);
    context.expect_eof();
    BOOST_CHECK_EQUAL(val.uns, expected);
    BOOST_CHECK_EQUAL(val.type, JsonNumber::SIGNED_INT);
}

void testFp(const std::string & str, double expected)
{
    Parse_Context context(str, str.c_str(), str.c_str() + str.size());
    auto val = expectJsonNumber(context);
    context.expect_eof();
    BOOST_CHECK_EQUAL(val.fp, expected);
    BOOST_CHECK_EQUAL(val.type, JsonNumber::FLOATING_POINT);
}

void testHex4(const std::string & str, long long expected)
{
    Parse_Context context(str, str.c_str(), str.c_str() + str.size());
    auto val = context.expect_hex4();
    context.expect_eof();
    BOOST_CHECK_EQUAL(val, expected);
}

BOOST_AUTO_TEST_CASE( test1 )
{
    testUnsigned("0", 0);
    testSigned("-0", 0);
    testSigned("-1", -1);
    testFp("0.", 0.0);
    testFp(".1", 0.1);
    testFp("-.1", -0.1);
    testFp("0.0", 0.0);
    testFp("1e0", 1e0);
    testFp("-1e0", -1e0);
    testFp("-1e+0", -1e+0);
    testFp("-1e-0", -1e-0);
    testFp("-1E+3", -1e+3);
    testFp("1.0E-3", 1.0E-3);
    testFp("Inf", INFINITY);
    testFp("-Inf", -INFINITY);

    testHex4("0026", 38);
    testHex4("001A", 26);
    
    JML_TRACE_EXCEPTIONS(false);
    BOOST_CHECK_THROW(testFp(".", 0.1), std::exception);
    BOOST_CHECK_THROW(testFp("", 0.1), std::exception);
    BOOST_CHECK_THROW(testFp("e3", 0.1), std::exception);
    BOOST_CHECK_THROW(testFp("3e", 0.1), std::exception);
    BOOST_CHECK_THROW(testFp("3.1aade", 0.1), std::exception);
    
    BOOST_CHECK_THROW(testHex4("002", 2), std::exception);
    BOOST_CHECK_THROW(testHex4("002G", 2), std::exception);
    BOOST_CHECK_THROW(testHex4("002.", 2), std::exception);
}
