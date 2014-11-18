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

#if 1
BOOST_AUTO_TEST_CASE(time_period_granularity_multiplier)
{
    JML_TRACE_EXCEPTIONS(false);

    /* different families */
    BOOST_CHECK_THROW(granularityMultiplier(YEARS, MINUTES), ML::Exception);

    /* seconds cannot be translated to minutes */
    BOOST_CHECK_THROW(granularityMultiplier(SECONDS, MINUTES), ML::Exception);

    int mult = granularityMultiplier(MILLISECONDS, MILLISECONDS);
    BOOST_CHECK_EQUAL(mult, 1);

    mult = granularityMultiplier(MINUTES, MILLISECONDS);
    BOOST_CHECK_EQUAL(mult, 60000);

    mult = granularityMultiplier(MINUTES, MINUTES);
    BOOST_CHECK_EQUAL(mult, 1);

    mult = granularityMultiplier(WEEKS, MINUTES);
    BOOST_CHECK_EQUAL(mult, 10080);

    mult = granularityMultiplier(YEARS, YEARS);
    BOOST_CHECK_EQUAL(mult, 1);

    mult = granularityMultiplier(YEARS, MONTHS);
    BOOST_CHECK_EQUAL(mult, 12);
}
#endif

#if 1
/* Ensure that operators + and += works well for TimePeriod */
BOOST_AUTO_TEST_CASE(time_period_op_plus_equal)
{
    /* same unit */
    {
        TimePeriod period1("2m");
        TimePeriod period2("5m");

        TimePeriod total = period1 + period2;

        BOOST_CHECK_EQUAL(total.granularity, MINUTES);
        BOOST_CHECK_EQUAL(total.number, 7);
        BOOST_CHECK_EQUAL(total.interval, 420);
    }

    /* distinct compatible units */
    {
        TimePeriod period1("1h");
        TimePeriod period2("2s");

        TimePeriod total = period1 + period2;

        BOOST_CHECK_EQUAL(total.granularity, SECONDS);
        BOOST_CHECK_EQUAL(total.number, 3602);
        BOOST_CHECK_EQUAL(total.interval, 3602);

        /* operator += */
        period1 += period2;

        BOOST_CHECK_EQUAL(period1.granularity, SECONDS);
        BOOST_CHECK_EQUAL(period1.number, 3602);
        BOOST_CHECK_EQUAL(period1.interval, 3602);
    }

    /* same as above, in reverse order */
    {
        TimePeriod period1("2s");
        TimePeriod period2("1h");

        TimePeriod total = period1 + period2;

        BOOST_CHECK_EQUAL(total.granularity, SECONDS);
        BOOST_CHECK_EQUAL(total.number, 3602);
        BOOST_CHECK_EQUAL(total.interval, 3602);

        /* operator += */
        period1 += period2;

        BOOST_CHECK_EQUAL(period1.granularity, SECONDS);
        BOOST_CHECK_EQUAL(period1.number, 3602);
        BOOST_CHECK_EQUAL(period1.interval, 3602);
    }

    /* incompatible units */
    {
        TimePeriod yearly;
        yearly.granularity = YEARS;
        yearly.number = 1;
        yearly.interval = -1; // years do not have a fixed set of seconds
        TimePeriod minutely("2m");

        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(yearly + minutely, ML::Exception);
    }

    {
        TimePeriod t;

        t += "1s";
        BOOST_CHECK_EQUAL(t.granularity, SECONDS);
        BOOST_CHECK_EQUAL(t.number, 1);
        BOOST_CHECK_EQUAL(t.interval, 1);
    }
}
#endif
