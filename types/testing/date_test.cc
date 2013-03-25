/* date_test.cc
   Copyright (c) 2010 Datacratic.  All rights reserved.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include "soa/types/date.h"
#include <boost/test/unit_test.hpp>
#include "soa/jsoncpp/json.h"
#include "jml/arch/format.h"
#include "jml/utils/parse_context.h"
#include "ace/Time_Value.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

#if 0
BOOST_AUTO_TEST_CASE(test_date_parse_iso8601)
{
    // Issue PLAT-151 should fix this test
    BOOST_CHECK_NO_THROW(Date::parseIso8601("2012-12-20T14:57:57.187775+00:00")) ;
}
#endif
BOOST_AUTO_TEST_CASE( test_microsecond_date )
{
    Date d = Date::now();
    double x;
    BOOST_CHECK_GT(modf(d.secondsSinceEpoch(), &x), 0.000001);
}

BOOST_AUTO_TEST_CASE( test_quantize )
{
    Date d = Date::fromSecondsSinceEpoch(0.1);
    BOOST_CHECK_EQUAL(d.quantized(1.0), Date::fromSecondsSinceEpoch(0.0));
    BOOST_CHECK_EQUAL(d.quantized(0.5), Date::fromSecondsSinceEpoch(0.0));
    BOOST_CHECK_EQUAL(d.quantized(0.2), Date::fromSecondsSinceEpoch(0.2));
    BOOST_CHECK_EQUAL(d.quantized(0.1), Date::fromSecondsSinceEpoch(0.1));
    BOOST_CHECK_EQUAL(d.quantized(0.05), Date::fromSecondsSinceEpoch(0.1));
    BOOST_CHECK_EQUAL(d.quantized(0.01), Date::fromSecondsSinceEpoch(0.1));

    d = Date::fromSecondsSinceEpoch(0.11);
    BOOST_CHECK_EQUAL(d.quantized(1.0), Date::fromSecondsSinceEpoch(0.0));
    BOOST_CHECK_EQUAL(d.quantized(0.5), Date::fromSecondsSinceEpoch(0.0));
    BOOST_CHECK_EQUAL(d.quantized(0.2), Date::fromSecondsSinceEpoch(0.2));
    BOOST_CHECK_EQUAL(d.quantized(0.1), Date::fromSecondsSinceEpoch(0.1));
    BOOST_CHECK_EQUAL(d.quantized(0.05), Date::fromSecondsSinceEpoch(0.1));
    BOOST_CHECK_EQUAL(d.quantized(0.01), Date::fromSecondsSinceEpoch(0.11));
}

BOOST_AUTO_TEST_CASE( test_fractional_date_print )
{
    Date d1 = Date::fromSecondsSinceEpoch(0);
    Date d2 = d1.plusSeconds(0.33336);

    BOOST_CHECK_EQUAL(d1.print(), d2.print());
    BOOST_CHECK_EQUAL(d1.print() + ".3", d2.print(1));
    BOOST_CHECK_EQUAL(d1.print() + ".33", d2.print(2));
    BOOST_CHECK_EQUAL(d1.print() + ".333", d2.print(3));
    BOOST_CHECK_EQUAL(d1.print() + ".3334", d2.print(4));
    BOOST_CHECK_EQUAL(d1.print() + ".33336", d2.print(5));
}


BOOST_AUTO_TEST_CASE( test_date_parse_roundtrip )
{
    Date d1 = Date::now().quantized(0.01);

    string s = d1.print(6);

    Date d2(s);

    cerr << "s = " << d1.print(9) << " s2 = " << d2.print(9) << endl;

    BOOST_CHECK_EQUAL(d1.quantized(0.000001), d2.quantized(0.000001));

    for (unsigned i = 0;  i < 1000;  ++i) {
        double sec = random();
        double frac = random() / 1000000000.0;

        Date d1 = Date::fromSecondsSinceEpoch(sec + frac);
        Date d2(d1.print(6));

        if (d1.print(6) != d2.print(6)) {
            cerr << "d1:6: " << d1.print(6) << endl;
            cerr << "d2:6: " << d2.print(6) << endl;
            cerr << "d1:9: " << d1.print(9) << endl;
            cerr << "d2:9: " << d2.print(9) << endl;
            cerr << "sec: " << ML::format("%32.10f", sec + frac) << endl;
        }

        BOOST_CHECK_EQUAL(d1.print(6), d2.print(6));
    }
}

BOOST_AUTO_TEST_CASE( test_stream_print_equality )
{
    Date d = Date::fromSecondsSinceEpoch(4516978241);

    std::stringstream ss;
    ss << d;

    BOOST_CHECK_EQUAL(ss.str(), "2113-Feb-19 20:10:41");
}

BOOST_AUTO_TEST_CASE( test_ace )
{
    ACE_Time_Value ace(1, 0);

    BOOST_CHECK_EQUAL(Date(ace).secondsSinceEpoch(), 1.0);
    BOOST_CHECK_EQUAL(Date(ace).toAce(),  ace);
}

BOOST_AUTO_TEST_CASE( test_print_format )
{
    BOOST_CHECK_EQUAL(Date(ACE_Time_Value(0)).print("%c"), "Thu Jan  1 00:00:00 1970");
    BOOST_CHECK_EQUAL(Date(ACE_Time_Value(1)).print("%c"), "Thu Jan  1 00:00:01 1970");
}

BOOST_AUTO_TEST_CASE( test_utc_parse )
{
    Date d(2012, 06, 06, 15, 15, 38.380);
    string s = "2012-Jun-06 15:15:38.380";
    Date d2 = Date::parseDefaultUtc(s);
    BOOST_CHECK_EQUAL(d2.print(3), s);
}

BOOST_AUTO_TEST_CASE( test_now )
{
    cerr << "new: " << Date::now().print(6) << endl;
    cerr << "old: " << Date::nowOld().print(6) << endl;
}

BOOST_AUTO_TEST_CASE( test_date_equality )
{
    BOOST_CHECK_EQUAL(Date(), Date());
}

BOOST_AUTO_TEST_CASE( test_date_parse_no_delimiter )
{
    const char * s = "20120624";
    Parse_Context context(s, s, s + strlen(s));
    Date date = Date::expect_date(context, "%y%m%d");

    BOOST_CHECK_EQUAL(date, Date(2012, 06, 24));
}

BOOST_AUTO_TEST_CASE( test_date_hour_of_week )
{
    BOOST_CHECK_EQUAL(Date(2012, 06, 24, 0, 0, 0).hourOfWeek(), 0);
    BOOST_CHECK_EQUAL(Date(2012, 06, 24, 1, 0, 0).hourOfWeek(), 1);
    BOOST_CHECK_EQUAL(Date(2012, 06, 25, 0, 0, 0).hourOfWeek(), 24);
    BOOST_CHECK_EQUAL(Date(2012, 06, 25, 23, 59, 59).hourOfWeek(), 47);
    BOOST_CHECK_EQUAL(Date(2012, 06, 26, 0, 0, 0).hourOfWeek(), 48);
    BOOST_CHECK_EQUAL(Date(2012, 06, 30, 23, 59, 59).hourOfWeek(), 167);
    BOOST_CHECK_EQUAL(Date(2012, 07, 1, 0, 0, 0).hourOfWeek(), 0);
}

BOOST_AUTO_TEST_CASE( test_date_difference )
{
    BOOST_CHECK_EQUAL(Date(2012, 06, 24, 0, 0, 0) -
                      Date(2012, 06, 24, 0, 0, 10), -10);
    BOOST_CHECK_EQUAL(Date(2012, 06, 24, 0, 1, 0) -
                      Date(2012, 06, 24, 0, 0, 0), 60);
}

BOOST_AUTO_TEST_CASE( test_rfc_2616 )
{
    BOOST_CHECK_EQUAL(Date(1994, 11, 6, 8, 49, 37).printRfc2616(),
                      "Sun, 06 Nov 1994 08:49:37 GMT");
}

BOOST_AUTO_TEST_CASE( test_addFromString )
{
    Date d(2013, 1, 1, 0, 0, 0);
    d.addFromString("1d");
    BOOST_CHECK_EQUAL(d, Date(2013, 1, 2, 0, 0, 0));
    d.addFromString("2S");
    BOOST_CHECK_EQUAL(d, Date(2013, 1, 2, 0, 0, 2));
    d.addFromString("3M");
    BOOST_CHECK_EQUAL(d, Date(2013, 1, 2, 0, 3, 2));
    d.addFromString("4H");
    BOOST_CHECK_EQUAL(d, Date(2013, 1, 2, 4, 3, 2));
    if(false){
        //this test leaks, skip it
        bool failureFailed = false;
        try{
            d.addFromString("pwel");
        }catch(const ML::Exception& e){
            failureFailed = true;
        }
        if(!failureFailed){
            throw ML::Exception("Invalid string should fail with a ML exception");
        }
    }
}
