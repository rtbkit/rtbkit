/* json_handling_test.cc
   Jeremy Barnes, 29 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test of the functionality to handle JSON.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "jml/db/persistent.h"
#include "soa/types/string.h"
#include "soa/types/json_parsing.h"
#include "soa/types/json_printing.h"

using namespace Datacratic;
using namespace std;

BOOST_AUTO_TEST_CASE(test_utf8_round_trip_streaming_binary)
{
    Utf8String str("\xe2\x80\xa2skin");
    
    std::ostringstream stream;

    {
        StreamJsonPrintingContext context(stream);
        context.writeUtf8 = true;
        context.writeStringUtf8(str);
    }

    cerr << stream.str() << endl;

    BOOST_CHECK_EQUAL(stream.str(), "\"\xe2\x80\xa2skin\"");

    {
        StringJsonParsingContext context(stream.str());
        Utf8String str2 = context.expectStringUtf8();
        BOOST_CHECK_EQUAL(str, str2);
    }
}

BOOST_AUTO_TEST_CASE(test_utf8_round_trip_streaming_ascii)
{
    Utf8String str("\xe2\x80\xa2skin");
    
    std::ostringstream stream;

    {
        StreamJsonPrintingContext context(stream);
        context.writeUtf8 = false;
        context.writeStringUtf8(str);
    }

    cerr << stream.str() << endl;

    BOOST_CHECK_EQUAL(stream.str(), "\"\\u2022skin\"");

    {
        StringJsonParsingContext context(stream.str());
        Utf8String str2 = context.expectStringUtf8();
        BOOST_CHECK_EQUAL(str, str2);
    }
}
