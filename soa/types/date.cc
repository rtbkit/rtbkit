/* date.cc
   Jeremy Barnes, 18 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "date.h"
#include <cmath>
#include <limits>
#include "jml/arch/format.h"
#include "soa/js/js_value.h"
#include "soa/jsoncpp/json.h"
#include <cmath>
#include "ace/Time_Value.h"
#include "jml/arch/exception.h"
#include "jml/db/persistent.h"
#include <boost/regex.hpp>


using namespace std;
using namespace ML;
using namespace Datacratic;

namespace {

bool
matchFixedWidthInt(ML::Parse_Context & context,
                   int minLength, int maxLength,
                   int min, int max, int & value)
{
    ML::Parse_Context::Revert_Token token(context);

    char buf[maxLength + 1];
    unsigned i = 0;
    for (;  i < maxLength && context;  ++i, ++context) {
        char c = *context;
        if (c < '0' || c > '9') {
            if (i >= minLength)
                break;
            else
                return false;
        }
        buf[i] = c;
    }
    if (i < minLength)
        return false;
    buf[i] = 0;

    char * endptr = 0;
    errno = 0;
    int result = strtol(buf, &endptr, 10);
    
    if (errno || *endptr != 0 || endptr != buf + i)
        context.exception("expected fixed width int");

    // This WILL bite us some time.  640k anyone?
    if (result < min || result > max) {
        return false;
    }

    token.ignore();

    value = result;

    return true;
}

int
expectFixedWidthInt(ML::Parse_Context & context,
                    int minLength, int maxLength,
                    int min, int max, const char * message)
{
    int result;

    if (!matchFixedWidthInt(context, minLength, maxLength,
                            min, max, result)) {
        context.exception(message);
    }

    return result;
}

}

namespace Datacratic {


/*****************************************************************************/
/* DATE                                                                      */
/*****************************************************************************/

Date::
Date(const ptime & date)
    : secondsSinceEpoch_((date - epoch).total_microseconds()/1000000.0)
{
}

Date::
Date(int year, int month, int day,
     int hour, int minute, int second,
     double fraction)
    : secondsSinceEpoch_((boost::posix_time::ptime
                            (boost::gregorian::date(year, month, day)) - epoch)
                           .total_seconds()
                           + 3600 * hour + 60 * minute + second
                           + fraction)
{
}

Date::
Date(JS::JSValue & value)
{
    throw Exception("Date::Date(JSValue): not done");
}

Date::
Date(const Json::Value & value)
{
    if (value.isConvertibleTo(Json::realValue)) {
        secondsSinceEpoch_ = value.asDouble();
    }
    else if (value.isConvertibleTo(Json::stringValue)) {
        Date parsed = parse_date_time(value.asString(), "%y-%M-%d", "%H:%M:%S");
        secondsSinceEpoch_ = parsed.secondsSinceEpoch_;
    }
    else throw Exception("Date::Date(Json): JSON value "
                         + value.toStyledString()
                         + "not convertible to date");
}

Date::
Date(const ACE_Time_Value & value)
{
    uint64_t msec;
    value.msec(msec);
    secondsSinceEpoch_ = msec / 1000.0;
}

Date
Date::
fromIso8601Week(int year, int week, int day)
{
    Date newDate(year, 1, 1);

    int currentWeek = newDate.iso8601WeekOfYear();
    if (currentWeek == 1) {
        newDate.addWeeks(week - 1);
    }
    else {
        newDate.addWeeks(week);
    }

    return newDate.iso8601WeekStart().plusDays(day - 1);
}

Date
Date::
parseSecondsSinceEpoch(const std::string & date)
{
    errno = 0;
    char * end = 0;
    double seconds = strtod(date.c_str(), &end);
    if (errno != 0)
        throw ML::Exception(errno, "date parseSecondsSinceEpoch: " + date);
    if (end != date.c_str() + date.length())
        throw ML::Exception("couldn't convert " + date + " to date");
    return fromSecondsSinceEpoch(seconds);
}

Date
Date::
parseDefaultUtc(const std::string & date)
{
    if (date == "NaD" || date == "NaN")
        return notADate();
    else if (date == "Inf")
        return positiveInfinity();
    else if (date == "-Inf")
        return negativeInfinity();

    return parse_date_time(date, "%y-%M-%d", "%H:%M:%S");
}

Date
Date::
parseIso8601(const std::string & date)
{
    if (date == "NaD" || date == "NaN")
        return notADate();
    else if (date == "Inf")
        return positiveInfinity();
    else if (date == "-Inf")
        return negativeInfinity();
    else {
        return parse_date_time(date, "%y-%m-%d", "T%H:%M:%SZ");
    }
}

Date
Date::
parseIso8601DateTime(const std::string & dateTimeStr)
{
    if (dateTimeStr == "NaD" || dateTimeStr == "NaN")
        return notADate();
    else if (dateTimeStr == "Inf")
        return positiveInfinity();
    else if (dateTimeStr == "-Inf")
        return negativeInfinity();
    else
        return Iso8601Parser::parseDateTimeString(dateTimeStr);
}

Date
Date::
notADate()
{
    Date result;
    result.secondsSinceEpoch_ = std::numeric_limits<double>::quiet_NaN();
    return result;
}

Date
Date::
positiveInfinity()
{
    Date result;
    result.secondsSinceEpoch_ = INFINITY;
    return result;
}

Date
Date::
negativeInfinity()
{
    Date result;
    result.secondsSinceEpoch_ = -INFINITY;
    return result;
}

Date
Date::
now()
{
    timespec time;
    int res = clock_gettime(CLOCK_REALTIME, &time);
    if (res == -1)
        throw ML::Exception(errno, "clock_gettime");
    return fromSecondsSinceEpoch(time.tv_sec + time.tv_nsec * 0.000000001);
}

Date
Date::
nowOld()
{
    boost::posix_time::ptime
        t(boost::posix_time::microsec_clock::universal_time());
    Date result(t);
    return result;
}

bool
Date::
isADate() const
{
    return std::isfinite(secondsSinceEpoch_);
}

std::string
Date::
print(unsigned seconds_digits) const
{
    if (!std::isfinite(secondsSinceEpoch_)) {
        if (std::isnan(secondsSinceEpoch_)) {
            return "NaD";
        }
        else if (secondsSinceEpoch_ > 0) {
            return "Inf";
        }
        else return "-Inf";
    }

    string result = print("%Y-%b-%d %H:%M:%S");
    if (seconds_digits == 0) return result;

    double partial_seconds = fractionalSeconds();
    string fractional = format("%0.*f", seconds_digits, partial_seconds);

    result.append(fractional, 1, -1);
    return result;
}

std::string
Date::
printRfc2616() const
{
    if (!std::isfinite(secondsSinceEpoch_)) {
        if (std::isnan(secondsSinceEpoch_)) {
            return "NaD";
        }
        else if (secondsSinceEpoch_ > 0) {
            return "Inf";
        }
        else return "-Inf";
    }

    return print("%a, %d %b %Y %H:%M:%S GMT");
}

std::string
Date::
printIso8601(unsigned int fraction) const
{
    if (!std::isfinite(secondsSinceEpoch_)) {
        if (std::isnan(secondsSinceEpoch_)) {
            return "NaD";
        }
        else if (secondsSinceEpoch_ > 0) {
            return "Inf";
        }
        else return "-Inf";
    }

    string result = print("%Y-%m-%dT%H:%M:%S");

    if (result == "Inf" || result == "-Inf" || result == "NaD")
        return result;
    
    double partial_seconds = fractionalSeconds();
    string fractional = format("%.*fZ", fraction, partial_seconds);

    result.append(fractional, 1, -1);

    return result;
}

std::string
Date::
printClassic() const
{
    if (!std::isfinite(secondsSinceEpoch_)) {
        if (std::isnan(secondsSinceEpoch_)) {
            return "NaD";
        }
        else if (secondsSinceEpoch_ > 0) {
            return "Inf";
        }
        else return "-Inf";
    }

    return print("%Y-%m-%d %H:%M:%S");
}

Date
Date::
quantized(double fraction) const
{
    Date result = *this;
    return result.quantize(fraction);
}

Date &
Date::
quantize(double fraction)
{
    if (fraction <= 0.0)
        throw Exception("Date::quantize(): "
                        "fraction cannot be zero or negative");

    if (fraction <= 1.0) {
        // Fractions of a second; split off to avoid loss of precision
        double whole_seconds, partial_seconds;
        partial_seconds = modf(secondsSinceEpoch_, &whole_seconds);

        double periods_per_second = round(1.0 / fraction);
        partial_seconds = round(partial_seconds * periods_per_second)
            / periods_per_second;

        secondsSinceEpoch_ = whole_seconds + partial_seconds;
    }
    else {
        // Fractions of a second; split off to avoid loss of precision
        double whole_seconds;
        // double partial_seconds = modf(secondsSinceEpoch_, &whole_seconds);
        modf(secondsSinceEpoch_, &whole_seconds);

        uint64_t frac = fraction;
        if (frac != fraction)
            throw ML::Exception("non-integral numbers of seconds not supported");
        uint64_t whole2 = whole_seconds;
        whole2 /= fraction;
        secondsSinceEpoch_ = whole2 * fraction;
    }

    return *this;
}

std::string
Date::
print(const std::string & format) const
{
    size_t buffer_size = format.size() + 1024;
    char buffer[buffer_size];

    if (secondsSinceEpoch() >= 100000000000) {
        return "Inf";
    }
    if (secondsSinceEpoch() <= -1000000000000) {
        return "-Inf";
    }

    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time)) {
        cerr << strerror(errno) << endl;
        cerr << t << endl;
        cerr << secondsSinceEpoch() << endl;
        throw Exception("problem with gmtime_r");
    }
    size_t nchars = strftime(buffer, buffer_size, format.c_str(),
                             &time);
    
    if (nchars == 0)
        throw Exception("couldn't print date format " + format);
    
    return string(buffer, buffer + nchars);
}

int
Date::
hour() const
{
    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time))
        throw Exception("problem with gmtime_r");

    return time.tm_hour;
}

int
Date::
minute() const
{
    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time))
        throw Exception("problem with gmtime_r");

    return time.tm_min;
}

int
Date::
second() const
{
    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time))
        throw Exception("problem with gmtime_r");

    return time.tm_sec;
}

int
Date::
weekday()
    const
{
    using namespace boost::gregorian;

    int day_of_week;
    try {
        day_of_week = from_string(print()).day_of_week();
    } catch (...) {
        cerr << "order_date = " << *this << endl;
        throw;
    }

    return day_of_week;
}

int
Date::
iso8601Weekday()
    const
{
    int weekDay = weekday();

    if (weekDay == 0) {
        weekDay = 7;
    }

    return weekDay;
}

int
Date::
dayOfMonth() const
{
    return boost::gregorian::from_string(print()).day();
}

int
Date::
dayOfYear()
    const
{
    time_t t = secondsSinceEpoch_;
    struct tm time;

    ::gmtime_r(&t, &time);

    return time.tm_yday;
}

int
Date::
iso8601WeekOfYear()
    const
{
    int yearDay = dayOfYear();
    int weekDay = iso8601Weekday();

    if (yearDay == 0) {
        /* first week = the week with the year's first Thursday in it */
        if (weekDay <= 4) {
            return 1;
        }
        else {
            /* else: Jan 1 is part of week 52 or 53 of the previous year */
            Date prevDec31 = plusSeconds(-24 * 3600);
            return prevDec31.iso8601WeekOfYear();
        }
    }

    Date jan1 = plusSeconds(24 * 3600 * -yearDay);
    int jan1Week = jan1.iso8601WeekOfYear();
    int weeks = (1 + yearDay - weekDay + jan1.iso8601Weekday()) / 7;
    if (weeks == 0) {
        weeks = jan1Week;
    }
    else if (jan1Week == 1) {
        weeks++;
    }

    return weeks;
}

int
Date::
monthOfYear() const
{
    return boost::gregorian::from_string(print()).month();
}

int
Date::
year() const
{
    return boost::gregorian::from_string(print()).year();
}

int
Date::
hourOfWeek() const
{
    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time))
        throw Exception("problem with gmtime_r");

    return time.tm_wday * 24 + time.tm_hour;
}

std::string
Date::
printMonth() const
{
    throw Exception("Date: stub method");
}

std::string
Date::
printWeekday() const
{
    throw Exception("Date: stub method");
}

std::string
Date::
printYearAndMonth() const
{
    throw Exception("Date: stub method");
}

const boost::posix_time::ptime
Date::epoch(boost::gregorian::date(1970, 1, 1));

std::ostream & operator << (std::ostream & stream, const Date & date)
{
    return stream << date.print();
}

std::istream & operator >> (std::istream & stream, Date & date)
{
    throw Exception("date istream parsing doesn't work yet");
}

bool
Date::
match_date(ML::Parse_Context & context,
           Date & result,
           const std::string & format)
{
    Parse_Context::Revert_Token token(context);
    
    int year = -1, month = 1, day = 1;
    
    for (const char * f = format.c_str();  context && *f;  ++f) {
        if (*f != '%') {
            if (!context.match_literal(*f)) return false;
            continue;
        }
        
        ++f;
        switch (*f) {
        case '%':
            if (!context.match_literal('%')) return false;
            break;
        case 'd':
            if (!context.match_int(day, 1, 31)) return false;
            break;
        case 'm':
            if (!context.match_int(month, 1, 12)) return false;
            break;
        case 'M':
            switch(tolower(*context)) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': {
                if (!context.match_int(month, 1, 12))
                    return false;
                break;
            }
            case 'j': {
                ++context;
                if (context.match_literal("an")) {
                    month = 1;
                    break;
                }
                else if (context.match_literal("un")) {
                    month = 6;
                    break;
                }
                else if (context.match_literal("ul")) {
                    month = 7;
                    break;
                }
                else return false;
            }
            case 'f': {
                ++context;
                if (!context.match_literal("eb")) return false;
                month = 2;
                break;
            }
            case 'm': {
                ++context;
                if (context.match_literal("ar")) {
                    month = 3;
                    break;
                }
                else if (context.match_literal("ay")) {
                    month = 5;
                    break;
                }
                else return false;
                break;
            }
            case 'a':
                ++context;
                if (context.match_literal("pr")) {
                    month = 4;
                    break;
                }
                else if (context.match_literal("ug")) {
                    month = 8;
                    break;
                }
                else return false;
                break;
            case 's':
                ++context;
                if (!context.match_literal("ep")) return false;
                month = 9;
                break;
            case 'o':
                ++context;
                if (!context.match_literal("ct")) return false;
                month = 10;
                break;
            case 'n': {
                ++context;
                if (!context.match_literal("ov")) return false;
                month = 11;
                break;
            }
            case 'd': {
                ++context;
                if (!context.match_literal("ec")) return false;
                month = 12;
                break;
            }
            default:
                return false;
            }
            break;
        case 'y':
            if (!context.match_int(year, 1400, 9999));
            break;
        default:
            throw Exception("expect_date: format " + string(1, *f)
                            + " not implemented yet");
        }
    }
    
    try {
        boost::gregorian::date date(year, month, day);
        boost::posix_time::ptime time(date);
        result = Date(time);
    } catch (const std::exception & exc) {
        return false;
    }
    token.ignore();
    return true;
}

Date
Date::
expect_date(ML::Parse_Context & context, const std::string & format)
{
    int year = -1, month = 1, day = 1;

    for (const char * f = format.c_str();  context && *f;  ++f) {
        if (*f != '%') {
            context.expect_literal(*f);
            continue;
        }
        
        ++f;
        switch (*f) {
        case '%':
            context.expect_literal('%');
            break;
        case 'd':
            day = expectFixedWidthInt(context, 1, 2, 1, 31,
                                      "expected day of month");
            break;
        case 'm':
            month = expectFixedWidthInt(context, 1, 2, 1, 12,
                                        "expected month of year");
            break;
        case 'M':
            switch(tolower(*context)) {
            case '0':
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9': {
                month = expectFixedWidthInt(context, 1, 2, 1, 12,
                                            "expected month of year");
                break;
            }
            case 'j': {
                ++context;
                if (context.match_literal("an")) {
                    month = 1;
                    break;
                }
                else if (context.match_literal("un")) {
                    month = 6;
                    break;
                }
                else if (context.match_literal("ul")) {
                    month = 7;
                    break;
                }
                else context.exception("expected month name");
            }
            case 'f': {
                ++context;
                context.expect_literal("eb", "expected Feb");
                month = 2;
                break;
            }
            case 'm': {
                ++context;
                if (context.match_literal("ar")) {
                    month = 3;
                    break;
                }
                else if (context.match_literal("ay")) {
                    month = 5;
                    break;
                }
                else context.exception("expected month name");
                break;
            }
            case 'a':
                ++context;
                if (context.match_literal("pr")) {
                    month = 4;
                    break;
                }
                else if (context.match_literal("ug")) {
                    month = 8;
                    break;
                }
                else context.exception("expected month name");
                break;
            case 's':
                ++context;
                context.expect_literal("ep", "expected Sep");
                month = 9;
                break;
            case 'o':
                ++context;
                context.expect_literal("ct", "expected Oct");
                month = 10;
                break;
            case 'n': {
                ++context;
                context.expect_literal("ov", "expected Nov");
                month = 11;
                break;
            }
            case 'd': {
                ++context;
                context.expect_literal("ec", "expected Dec");
                month = 12;
                break;
            }
            default:
                context.exception("expected month name for %M");
            }
            break;
        case 'y':
            year = expectFixedWidthInt(context, 4, 4, 1400, 2999,
                                       "expected year");
            break;
        default:
            throw Exception("expect_date: format " + string(1, *f)
                            + " not implemented yet");
        }
    }
    
    //cerr << "year = " << year << " month = " << month
    //     << " day = " << day << endl;
    
    try {
        boost::gregorian::date date(year, month, day);
        boost::posix_time::ptime time(date);
        return Date(time);
        //result = (time - DateHandler::epoch).total_seconds();
        //cerr << "result = " << result << endl;
    } catch (const std::exception & exc) {
        context.exception("error parsing date: " + string(exc.what()));
        throw Exception("not reached");
    }
    //return result;
}

#if 0
Date
Date::
expect_date(ML::Parse_Context & context,
           const std::string & format)
{
    Date result;
    if (!match_date(context, format))
        context.exception("expected date");
    return result;
}
#endif

bool
Date::
match_time(ML::Parse_Context & context,
           double & result,
           const std::string & format)
{
    Parse_Context::Revert_Token token(context);

    int hour = 0, minute = 0, offset = 0;
    double second = 0;
    bool twelve_hour = false;
    
    for (const char * f = format.c_str();  context && *f;  ++f) {
        if (*f != '%') {
            if (!context.match_literal(*f)) return false;
            continue;
        }
        
        ++f;
        switch (*f) {
        case '%':
            if (!context.match_literal('%')) return false;;
            break;
        case 'h':
            twelve_hour = true;
            if (!context.match_int(hour, 1, 12))
                return false;
            break;
        case 'H':
            twelve_hour = false;
            if (!context.match_int(hour, 0, 24))
                return false;
            if (hour >= 24) return false;
            break;
        case 'M':
            if (!context.match_int(minute, 0, 60))
                return false;
            break;
        case 'S':
            if (!context.match_double(second, 0, 60))
                return false;
            if (second >= 60.0) return false;
            break;
        case 'p':
            twelve_hour = true;
            if (context.match_literal('A')) {
                if (!context.match_literal('M')) return false;
                offset = 0;
            }
            else if (context.match_literal('P')) {
                if (!context.match_literal('M')) return false;
                offset = 12;
            }
            else return false;
            break;
            
        default:
            throw Exception("expect_time: format " + string(1, *f)
                            + " not implemented yet");
        }
    }
    
    if (twelve_hour) {
        if (hour < 1 || hour > 12)
            return false;
        if (hour == 12) hour = 0;
        hour += offset;
    }

    double fractional_sec, full_sec;
    fractional_sec = modf(second, &full_sec);
    
    using namespace boost::posix_time;
    result = (hours(hour) + minutes(minute) + seconds(full_sec)).total_seconds()
        + fractional_sec;
    token.ignore();
    return true;
}

double
Date::
expect_time(ML::Parse_Context & context, const std::string & format)
{
    int hour = 0, minute = 0, offset = 0;
    double second = 0;
    bool twelve_hour = false;
    
    for (const char * f = format.c_str();  context && *f;  ++f) {
        if (*f != '%') {
            context.expect_literal(*f);
            continue;
        }
        
        ++f;
        switch (*f) {
        case '%':
            context.expect_literal('%');
            break;
        case 'h':
            twelve_hour = true;
            hour = context.expect_int(1, 12, "expected hours");
            break;
        case 'H':
            twelve_hour = false;
            hour = context.expect_int(0, 24, "expected hours");
            if (hour >= 24)
                context.exception("expected 24-hour hour");
            break;
        case 'M':
            minute = context.expect_int(0, 60, "expected minutes");
            break;
        case 'S':
            second = context.expect_double(0, 60, "expected seconds");
            if (second >= 60.0)
                context.exception("seconds cannot be 60");
            break;
        case 'p':
            twelve_hour = true;
            if (context.match_literal('A')) {
                context.expect_literal('M', "expected AM");
                offset = 0;
            }
            else if (context.match_literal('P')) {
                context.expect_literal('M', "expected AM");
                offset = 12;
            }
            else context.exception("expected AM or PM");
            break;
            
        default:
            throw Exception("expect_time: format " + string(1, *f)
                            + " not implemented yet");
        }
    }
    
    if (twelve_hour) {
        if (hour < 1 || hour > 12)
            context.exception("invalid hour after 12 hours");
        if (hour == 12) hour = 0;
        hour += offset;
    }

    double fractional_sec, full_sec;
    fractional_sec = modf(second, &full_sec);
    
    using namespace boost::posix_time;
    return (hours(hour) + minutes(minute) + seconds(full_sec)).total_seconds()
        + fractional_sec;
}

#if 0
double
Date::
expect_time(ML::Parse_Context & context,
           const std::string & format)
{
    double time;
    if (!context.match_time(context, time, format))
        context.exception("expected time");
    return time;
}
#endif



/** 
    DEPRECATED FUNCTION documentation:
    This function takes a string expected to contain a date that matches the
    provided date pattern, followed by a time. The two patterns in the
    string can be separated by whitespace but anything else has to appear in
    the patterns. 
    
    example:

        parse_date_time("2013-05-13/21:00:00", "%y-%m-%d/","%H:%M:%S")

    returns 2013-May-13 21:00:00.
    
    symbols meanings:
        date_format:
            %d      day of month as digit 1-31
            %m      month as digit 1-12
            %M      month as 3-letter abbreviation
            %y      year with century 1400-2999
        time_format:
            %h      hour as digit 1-12
            %H      hour as digit 0-24
            %M      minute as digit 0-60
            %S      second as digit 0-60
            %p      'AM' or 'PM'
**/
Date
Date::
parse_date_time(const std::string & str,
                const std::string & date_format,
                const std::string & time_format)
{
    using namespace boost::posix_time;
    
    if (str == "") return Date::notADate();
    
    Date result;
    try {
        ML::Parse_Context context(str,
                                  str.c_str(), str.c_str() + str.length());
        result = expect_date_time(context, date_format, time_format);
        
        context.expect_eof();
    }
    catch (const std::exception & exc) {
        //cerr << "Error parsing date string:\n'" << str << "'" << endl;
        throw;
    }
    
    return result;
}

Date
Date::
expect_date_time(ML::Parse_Context & context,
                 const std::string & date_format,
                 const std::string & time_format)
{
    Date date;
    double time = 0.0;
    
    date = expect_date(context, date_format);
    
    if (!context.eof()) {
        Parse_Context::Revert_Token token(context);
        context.match_whitespace();
        if (match_time(context, time, time_format))
            token.ignore();
    }
    
    return date.plusSeconds(time);
}

bool
Date::
match_date_time(ML::Parse_Context & context,
                Date & result,
                const std::string & date_format,
                const std::string & time_format)
{
    Date date;
    double time = 0.0;
    
    if (!match_date(context, date, date_format)) return false;
    context.match_whitespace();
    match_time(context, time, time_format);
    result = date.plusSeconds(time);

    return true;
}


Date Date::parse(const std::string & date,
                 const std::string & format)
{
    tm time;
    memset(&time, 0, sizeof(time));
    if(strptime(date.c_str(), format.c_str(), &time) == NULL)
        throw ML::Exception("strptime error. format='" + format + "', string='" + date + "'");

    //not using fromTm because I don't want it to assume it's local time
    return Date(1900 + time.tm_year, 1 + time.tm_mon, time.tm_mday,
                time.tm_hour, time.tm_min, time.tm_sec);
}

ACE_Time_Value
Date::
toAce() const
{
    ACE_Time_Value result;
    result.set(secondsSinceEpoch_);
    return result;
}

tm
Date::
toTm() const
{
    tm result;
    errno = 0;
    time_t t = toTimeT();
    if (gmtime_r(&t, &result) == 0)
        throw ML::Exception("error converting time: t = %lld (%s)",
                            (long long)t,
                            strerror(errno));
    return result;
}

Date
Date::
fromTm(const tm & t)
{
    tm t2 = t;
    time_t t3 = mktime(&t2);
    if (t3 == (time_t)-1)
        throw ML::Exception("couldn't construct from invalid time");
    return fromTimeT(t3);
}


void Date::addFromString(string str){
    {
        using namespace boost;
        string format = "^[1-9][0-9]*[SMHd]$";
        regex e(format);
        if(!regex_match(str, e)){
            throw ML::Exception("String " + str + " did not match format "
                + format);
        }
    }
    char unit = str[str.length() - 1];
    int length;
    {
        stringstream tmp;
        tmp << str.substr(0, -1);
        tmp >> length;
    }
    switch(unit){
        case 'S':
            this->addSeconds(length);
            break;
        case 'M':
            this->addMinutes(length);
            break;
        case 'H':
            this->addHours(length);
            break;
        case 'd':
            this->addDays(length);
            break;
        default:
            throw ML::Exception("Should never get here with string: " + str);
            break;
    }
}

DB::Store_Writer & operator << (ML::DB::Store_Writer & store, Date date)
{
    return store << date.secondsSinceEpoch();
}

DB::Store_Reader & operator >> (ML::DB::Store_Reader & store, Date & date)
{
    double d;
    store >> d;
    date = Date::fromSecondsSinceEpoch(d);
    return store;
}


/*****************************************************************************/
/* ISO8601PARSER                                                             */
/*****************************************************************************/

Date
Iso8601Parser::
expectDateTime()
{
    /*
      parse date
      try eof
      or {
        parse 'T' || ' '
        parse time
      }
      try eof
      or {
        parse tz
      }
    */

    Date date = expectDate();
    if (!eof() && (match_literal('T') || match_literal(' '))) {
        date.addSeconds(expectTime().secondsSinceEpoch());
    }

    return date;
}
    
Date
Iso8601Parser::
expectDate()
{
    int year = expectYear();

    match_literal('-');
    if (match_literal('W')) {
        int week = expectWeekNumber();
        match_literal('-');
        int day;
        if (eof()) {
            day = 1;
        }
        else {
            day = expectWeekDay();
        }

        return Date::fromIso8601Week(year, week, day);
    }
    else {
        int day(1);

        {
            ML::Parse_Context::Revert_Token token(*this);

            if (matchYearDay(day) && (eof() || !isdigit(*(*this)))) {
                Date date(year, 1, 1);
                date.addDays(day - 1);
                token.ignore();

                return date;
            }
        }

        int month(1);
        if (!eof()) {
            month = expectMonth();
            match_literal('-');
            if (!eof() && isdigit(*(*this))) {
                day = expectMonthDay();
            }
        }

        return Date(year, month, day);
    }
}

bool
Iso8601Parser::
matchYearDay(int & result)
{
    return matchFixedWidthInt(*this, 3, 3, 1, 366, result);
}

int
Iso8601Parser::
expectYear()
{
    return expectFixedWidthInt(*this, 4, 4, 1400, 9999, "bad year");
}

int
Iso8601Parser::
expectMonth()
{
    return expectFixedWidthInt(*this, 2, 2, 1, 12, "bad month");
}

int
Iso8601Parser::
expectWeekNumber()
{
    return expectFixedWidthInt(*this, 2, 2, 1, 53, "bad week number");
}

int
Iso8601Parser::
expectWeekDay()
{
    return expectFixedWidthInt(*this, 1, 1, 1, 7, "bad week day");
}

int
Iso8601Parser::
expectMonthDay()
{
    return expectFixedWidthInt(*this, 2, 2, 1, 31, "bad month day");
}

int
Iso8601Parser::
expectYearDay()
{
    return expectFixedWidthInt(*this, 3, 3, 1, 366, "bad year day");
}

Date
Iso8601Parser::
expectTime()
{
    Date date;

    int hours = expectHours();
    date.addHours(hours);
    if (eof()) {
        return date;
    }

    match_literal(':');

    int minutes = expectMinutes();
    date.addMinutes(minutes);
    if (eof()) {
        return date;
    }

    match_literal(':');
    int seconds = expectSeconds();
    date.addSeconds(seconds);
    if (eof()) {
        return date;
    }

    if (match_literal('.')) {
        size_t start = get_offset();
        int millis = expect_int();
        size_t end = get_offset();
        double seconds = double(millis) / pow(10, end-start);
        date.addSeconds(seconds);
    }

    if (eof()) {
        return date;
    }
    int tzminutes(0);
    if (match_literal('+')) {
        tzminutes = expectTimeZoneMinutes();
    }
    else if (match_literal('-')) {
        tzminutes = -expectTimeZoneMinutes();
    }
    else {
        expect_literal('Z');
    }
    date.addMinutes(tzminutes);

    return date;
}

int
Iso8601Parser::
expectHours()
{
    return expectFixedWidthInt(*this, 2, 2, 0, 23, "wrong hour value");
}

int
Iso8601Parser::
expectMinutes()
{
    return expectFixedWidthInt(*this, 2, 2, 0, 59, "bad minute value");
}

bool
Iso8601Parser::
matchMinutes(int & result)
{
    return matchFixedWidthInt(*this, 2, 2, 0, 59, result);
}

int
Iso8601Parser::
expectSeconds()
{
    return expectFixedWidthInt(*this, 2, 2, 0, 60, "bad second value");
}

int
Iso8601Parser::
expectTimeZoneMinutes()
{
    int minutes(0);

    int hours = expectHours();
    match_literal(':');
    matchMinutes(minutes);
    minutes += hours * 60;

    return minutes;
}

} // namespace Datacratic
