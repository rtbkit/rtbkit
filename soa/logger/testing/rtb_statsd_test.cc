/* rtb_client_test.cc
   Jeremy Barnes, 31 January 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Test for the RTB client.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "rtb/statsd_connector.h"


using namespace std;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_rtb_statsd )
{
    StatsdConnector x("127.0.0.1:4567");
    for(int i=0; i<30; i++) x.incrementCounter("test", 0.1);
    for(int i=0; i<300; i++) x.recordGauge("testGauge", 0.1, 5.2);
    BOOST_CHECK_EQUAL(2, 2);
}

BOOST_AUTO_TEST_CASE( test_graphite_connector )
{
    GraphiteConnector x("127.0.0.1:2003",
                        "graphiteTest");
}
