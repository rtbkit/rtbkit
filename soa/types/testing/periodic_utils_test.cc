#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/types/periodic_utils.h"
#include <boost/test/unit_test.hpp>

using namespace std;
using namespace ML;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE(time_period_test)
{
    TimePeriod tp("20s");
    BOOST_CHECK_EQUAL(tp.interval, 20);
    BOOST_CHECK_EQUAL(tp.toString(), "20s");

    tp.parse("1m");
    BOOST_CHECK_EQUAL(tp.interval, 60);
    BOOST_CHECK_EQUAL(tp.toString(), "1m");

    tp.parse("90s");
    BOOST_CHECK_EQUAL(tp.interval, 90);
    BOOST_CHECK_EQUAL(tp.toString(), "90s");

    bool threw = false;
    try {
        JML_TRACE_EXCEPTIONS(false);
        tp.parse("1m25s");
    }
    catch (...) {
        threw = true;
    }
    BOOST_CHECK(threw);

    tp.parse("1h");
    BOOST_CHECK_EQUAL(tp.interval, 3600);
    BOOST_CHECK_EQUAL(tp.toString(), "1h");


    tp.parse("1d");
    BOOST_CHECK_EQUAL(tp.interval, 3600 * 24);
    BOOST_CHECK_EQUAL(tp.toString(), "1d");
}
