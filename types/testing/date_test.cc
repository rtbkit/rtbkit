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

BOOST_AUTO_TEST_CASE(test_date_parse_iso8601_date_time)
{
    JML_TRACE_EXCEPTIONS(false);
    Date date;
    string expected;

    {
        /* YYYY-MM-DD  or  YYYYMMDD
           YYYY-MM  (but not YYYYMM) */
        vector<string> dateStrs = {"2013-04-01", "20130401", "2013-04"};
        expected = "2013-Apr-01 00:00:00.000";
        for (const string & dateStr: dateStrs) {
            date = Date::parseIso8601DateTime(dateStr);
            BOOST_CHECK_EQUAL(date.print(3), expected);
        }

        date = Date::parseIso8601DateTime("2013-04-01T09:08:07");
        expected = "2013-Apr-01 09:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("2013-04-01 09:08:07");
        expected = "2013-Apr-01 09:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("2013-04-01T09:08:07Z");
        expected = "2013-Apr-01 09:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("2013-04-01T09:08:07-04:00");
        expected = "2013-Apr-01 05:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("20130401T090807");
        expected = "2013-Apr-01 09:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("20130401T090807Z");
        expected = "2013-Apr-01 09:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);

        date = Date::parseIso8601DateTime("20130401T090807-0400");
        expected = "2013-Apr-01 05:08:07.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
    }

    {
        /* YYYY-Www  or  YYYYWww
           YYYY-Www-D  or  YYYYWwwD */
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-W00"), ML::Exception);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-W60"), ML::Exception);
        date = Date::parseIso8601DateTime("2013-W23");
        expected = "2013-Jun-03 00:00:00.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
        date = Date::parseIso8601DateTime("2013W23");
        expected = "2013-Jun-03 00:00:00.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-W23-8"), ML::Exception);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-W23-0"), ML::Exception);
        date = Date::parseIso8601DateTime("2013-W23-6");
        expected = "2013-Jun-08 00:00:00.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013W238"), ML::Exception);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013W230"), ML::Exception);
        date = Date::parseIso8601DateTime("2013W236");
        expected = "2013-Jun-08 00:00:00.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
    }

    {
        /* YYYY-DDD or YYYYDDD */
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-000"), ML::Exception);
        BOOST_CHECK_THROW(Date::parseIso8601DateTime("2013-367"), ML::Exception);
        date = Date::parseIso8601DateTime("2013-006");
        expected = "2013-Jan-06 00:00:00.000";
        BOOST_CHECK_EQUAL(date.print(3), expected);
    }

    {
        // Issue PLAT-151 should fix this test
        string dt = "2012-12-20T14:57:57.187+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.187";
        BOOST_CHECK_EQUAL(date.print(3), expected);
    }

    /* fractional seconds */
    {
        string dt = "2012-12-20T14:57:57.1+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.100000";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.12+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.120000";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.123+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.123000";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.1234+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.123400";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.12345+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.123450";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.123456+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.123456";
        BOOST_CHECK_EQUAL(date.print(6), expected);

        dt = "2012-12-20T14:57:57.1234567+00:00";
        date = Date::parseIso8601DateTime(dt);
        expected = "2012-Dec-20 14:57:57.1234567";
        BOOST_CHECK_EQUAL(date.print(7), expected);
    }

    /* negative seconds */
    {
        string dt = "1969-12-31T23:59:58.984375";
        date = Date::parseIso8601DateTime(dt);
        double result = date.secondsSinceEpoch();

        /* 0.015625 = 1/2 + 1/4 + ... + 1/64 */
        BOOST_CHECK_EQUAL(result, -1.015625);
    }
}

#if 0
BOOST_AUTO_TEST_CASE(test_date_parse_iso8601_time)
{
    JML_TRACE_EXCEPTIONS(false);
    Date date;
    string expected;

    BOOST_CHECK_THROW(Date::parseIso8601Time("24:59:01"), ML::Exception);
    BOOST_CHECK_THROW(Date::parseIso8601Time("23:69:01"), ML::Exception);
    BOOST_CHECK_THROW(Date::parseIso8601Time("23:09:61"), ML::Exception);
    date = Date::parseIso8601Time("02:23:43");
    expected = "1970-Jan-01 02:23:43.000";
    BOOST_CHECK_EQUAL(date.print(3), expected);
    date = Date::parseIso8601Time("022343");
    expected = "1970-Jan-01 02:23:43.000";
    BOOST_CHECK_EQUAL(date.print(3), expected);
    date = Date::parseIso8601Time("01:23");
    expected = "1970-Jan-01 01:23:00.000";
    BOOST_CHECK_EQUAL(date.print(3), expected);
    date = Date::parseIso8601Time("0123");
    expected = "1970-Jan-01 01:23:00.000";
    BOOST_CHECK_EQUAL(date.print(3), expected);
    date = Date::parseIso8601Time("23");
    expected = "1970-Jan-01 23:00:00.000";
    BOOST_CHECK_EQUAL(date.print(3), expected);
}
#endif

BOOST_AUTO_TEST_CASE( test_from_timespec )
{
    struct timespec ts{ 1, 987123456 };
    Date testDate = Date::fromTimespec(ts);
    double secs = testDate.secondsSinceEpoch();
    BOOST_CHECK_EQUAL(secs, 1.987123456);
}

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
    {
        Date d(2012, 06, 06, 15, 15, 38.380);
        string s = "2012-Jun-06 15:15:38.380";
        Date d2 = Date::parseDefaultUtc(s);
        BOOST_CHECK_EQUAL(d2.print(3), s);
    }

    {
        string s = "2012-Jun-06 15:15:38";
        Date d2 = Date::parseDefaultUtc(s);
        BOOST_CHECK_EQUAL(d2.print(0), s);
    }
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

// BOOST_AUTO_TEST_CASE( test_constructor )
// {
//     Date d1(2113, 2, 19, 20, 10, 41);
//     Date d2 = Date::fromSecondsSinceEpoch(4516978241);
//     BOOST_CHECK_EQUAL(d1, d2);
// }

BOOST_AUTO_TEST_CASE( test_strptime_parse )
{
    Date d(2013,1,1,0,0,0);
    vector<string> fmts = {
        "%d-%m-%Y %H:%M:%S",
        "%d %m %Y",
        "%d-%b-%Y",
    };
    for (auto & fmt : fmts) {
        cerr << d.print(fmt) << endl;
        cerr << Date::parse(d.print(fmt), fmt).print(fmt) << endl;
        string formattedDate = d.print(fmt);
        Date parsedDate = Date::parse(formattedDate, fmt);
        BOOST_CHECK_EQUAL(parsedDate, d);
    }
}

#if 1
BOOST_AUTO_TEST_CASE( test_parse_date_time )
{
    {
        Date date = Date::parse_date_time("2013-01-01-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.secondsSinceEpoch(), 1356998400);
    }

    {
        Date date = Date::parse_date_time("2012-07-01-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.secondsSinceEpoch(), 1341100800);
    }
}
#endif

BOOST_AUTO_TEST_CASE( test_weekday )
{
    {
        /* "2012-12-30-00" = sunday */
        Date date = Date::parse_date_time("2012-12-30-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.weekday(), 0);
    }

    {
        /* "2012-12-31-00" = monday */
        Date date = Date::parse_date_time("2012-12-31-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.weekday(), 1);
    }
}

BOOST_AUTO_TEST_CASE( test_iso8601Weekday )
{
    {
        /* "2011-01-01-00" = saturday */
        Date date = Date::parse_date_time("2011-01-01-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.iso8601Weekday(), 6);
    }

    {
        /* "2011-01-02-00" = sunday */
        Date date = Date::parse_date_time("2011-01-02-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.iso8601Weekday(), 7);
    }

    {
        /* "2012-12-30-00" = sunday */
        Date date = Date::parse_date_time("2012-12-30-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.iso8601Weekday(), 7);
    }

    {
        /* "2012-12-31-00" = monday */
        Date date = Date::parse_date_time("2012-12-31-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.iso8601Weekday(), 1);
    }
}

BOOST_AUTO_TEST_CASE( test_dayOfYear )
{
    {
        /* "2012-01-01-00" = day 0 */
        Date date = Date::parse_date_time("2012-01-01-00", "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.dayOfYear(), 0);
    }
}

BOOST_AUTO_TEST_CASE( test_Date_fromIso8601Week )
{
    Date sept130909 = Date::fromIso8601Week(2013, 37);
    BOOST_CHECK_EQUAL(sept130909.year(), 2013);
    BOOST_CHECK_EQUAL(sept130909.monthOfYear(), 9);
    BOOST_CHECK_EQUAL(sept130909.dayOfMonth(), 9);
    BOOST_CHECK_EQUAL(sept130909.iso8601WeekStart(), sept130909);
}

BOOST_AUTO_TEST_CASE( test_iso8601WeekOfYear )
{
    map<string, int> weeks;
    
    weeks.insert({"2010-12-31-00", 52});
    weeks.insert({"2011-01-01-00", 52});
    weeks.insert({"2011-01-02-00", 52});
    weeks.insert({"2011-01-03-00", 1});
    weeks.insert({"2013-01-01-00", 1});
    weeks.insert({"2013-01-05-00", 1});
    weeks.insert({"2013-01-06-00", 1});
    weeks.insert({"2013-01-07-00", 2});
    weeks.insert({"2013-01-08-00", 2});
    weeks.insert({"2013-09-08-23", 36});
    weeks.insert({"2013-09-09-00", 37});
    weeks.insert({"2013-09-15-23", 37});
    weeks.insert({"2013-09-16-00", 38});

    for (const auto & entry: weeks) {
        cerr << "testing " + entry.first << endl;

        Date date = Date::parse_date_time(entry.first, "%y-%m-%d-", "%H");
        BOOST_CHECK_EQUAL(date.iso8601WeekOfYear(), entry.second);
    }
}

BOOST_AUTO_TEST_CASE( test_printIso8601 )
{
    Date testDate = Date::fromSecondsSinceEpoch(1348089400.416978);

    string expected = "2012-09-19T21:16:40.417Z";
    string result = testDate.printIso8601();
    BOOST_CHECK_EQUAL(result, expected);

    expected = "2012-09-19T21:16:40.416978Z";
    result = testDate.printIso8601(6);
    BOOST_CHECK_EQUAL(result, expected);

    /* ensure that negative seconds do not append a extra digit to the number
     * of seconds */
    testDate = Date::fromSecondsSinceEpoch(-60);
    expected = "1969-12-31T23:59:00.000000Z";
    result = testDate.printIso8601(6);
    BOOST_CHECK_EQUAL(result, expected);
}

BOOST_AUTO_TEST_CASE( test_weekStart )
{
    {
        /* "2013-09-13-14" -> "2013-09-08-00"*/
        Date date = Date::parse_date_time("2013-09-13-14", "%y-%m-%d-", "%H");
        Date start = date.weekStart();
        BOOST_CHECK_EQUAL(start.printIso8601(), "2013-09-08T00:00:00.000Z");
    }
    {
        /* "2013-09-08-00" -> "2013-09-08-00"*/
        Date date = Date::parse_date_time("2013-09-08-00", "%y-%m-%d-", "%H");
        Date start = date.weekStart();
        BOOST_CHECK_EQUAL(start.printIso8601(), "2013-09-08T00:00:00.000Z");
    }
}

#if 1
BOOST_AUTO_TEST_CASE( test_iso8601WeekStart )
{
    /* "2013-09-13-14" -> "2013-09-09-00"*/
    Date date = Date::parse_date_time("2013-09-13-14", "%y-%m-%d-", "%H");
    Date start = date.iso8601WeekStart();
    BOOST_CHECK_EQUAL(start.printIso8601(), "2013-09-09T00:00:00.000Z");

    Date newStart = start.iso8601WeekStart();
    BOOST_CHECK_EQUAL(newStart, start);
}
#endif

// for PLAT-274
// BOOST_AUTO_TEST_CASE( test_patate) {
    // string x = "01 01 2013";
    // string fmt = "%d %m %Y";
    // // const char * x = "2013-Jan-01";
    // // const char * fmt = "%Y-%b-%d";

    // struct tm tm;
    // memset(&tm, 0, sizeof(struct tm));
    // strptime(x.c_str(), fmt.c_str(), &tm);
// }

BOOST_AUTO_TEST_CASE( test_date_iostream_print )
{
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::positiveInfinity()),
                      "Inf");
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::negativeInfinity()),
                      "-Inf");
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::notADate()),
                      "NaD");
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::fromSecondsSinceEpoch((uint64_t)-1)),
                      "Inf");

    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::fromSecondsSinceEpoch(9.22337e+18)), "Inf");
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::fromSecondsSinceEpoch(std::numeric_limits<int64_t>::min())), "-Inf");
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::fromSecondsSinceEpoch(9223372036854775807ULL)), "Inf");
    
    BOOST_CHECK_EQUAL(boost::lexical_cast<std::string>(Date::fromSecondsSinceEpoch(-9.22337e+18)), "-Inf");
}

BOOST_AUTO_TEST_CASE( test_minute )
{
    {
        // parse_date_time("2013-05-13/21:00:00", "%y-%m-%d/","%H:%M:%S")
        Date date = Date::parse_date_time("2012-12-30-00:01:02", "%y-%m-%d-", "%H:%M:%S");
        BOOST_CHECK_EQUAL(date.minute(), 1);
    }

}

BOOST_AUTO_TEST_CASE( test_second )
{
    {
        // parse_date_time("2013-05-13/21:00:00", "%y-%m-%d/","%H:%M:%S")
        Date date = Date::parse_date_time("2012-12-30-00:01:02", "%y-%m-%d-", "%H:%M:%S");
        BOOST_CHECK_EQUAL(date.second(), 2);
    }

}
