#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <soa/utils/variadic_hash.h>
#include <iostream>

using namespace std;

BOOST_AUTO_TEST_CASE( logger_metrics_mongo )
{
    BOOST_CHECK_EQUAL(hashAll("pwel", "-", "caribou", 123, ""), 
                      1790627249794637977);
}
