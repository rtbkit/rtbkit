/* exchange_parsing_from_file_test.cc
   Jean-Sebastien Bejeau, 27 May 2014
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Allow to test batch of Bid Request parsing from a file.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtbkit/testing/exchange_parsing_from_file.cc"

using namespace std;

BOOST_AUTO_TEST_CASE( test_exchange_parsing_multi_requests )
{
    Exchange_parsing_from_file myTest = Exchange_parsing_from_file("./rtbkit/testing/exchange_parsing_from_file_config.json");

    myTest.run();

    BOOST_CHECK_EQUAL(myTest.getNumError() , 0 );
}
