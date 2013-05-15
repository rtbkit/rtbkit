/* date.cc
   Jeremy Barnes, 18 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "date.h"
#include <limits>
#include "jml/utils/parse_context.h"
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
    return parse_date_time(date, "%y-%M-%d", "%H:%M:%S");
}

Date
Date::
parseIso8601(const std::string & date)
{
    return parse_date_time(date, "%y-%m-%d", "T%H:%M:%SZ");
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
    return print("%a, %d %b %Y %H:%M:%S GMT");
}

std::string
Date::
printIso8601() const
{
    return print("%Y-%m-%dT%H:%M:%S.000Z");
}

std::string
Date::
printClassic() const
{
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
        double whole_seconds, partial_seconds;
        partial_seconds = modf(secondsSinceEpoch_, &whole_seconds);

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

    time_t t = secondsSinceEpoch();
    tm time;

    if (!gmtime_r(&t, &time))
        throw Exception("problem with gmtime_r");
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
    throw Exception("Date: stub method");
}

int
Date::
minute() const
{
    throw Exception("Date: stub method");
}

int
Date::
second() const
{
    throw Exception("Date: stub method");
}

int
Date::
weekday() const
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
dayOfMonth() const
{
    return boost::gregorian::from_string(print()).day();
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

    auto expectFixedWidthInt = [&] (int length, int min, int max,
                                     const char * message)
        {
            char buf[length + 1];
            unsigned i = 0;
            for (;  i < length && context;  ++i, ++context) {
                char c = *context;
                if (c < '0' || c > '9') {
                    break;
                }
                buf[i] = c;
            }
            if (i == 0)
                context.exception(message);
            buf[i] = 0;
            int result = boost::lexical_cast<int>((const char *)buf);

            // This WILL bite us some time.  640k anyone?
            if (result < min || result > max) {
                context.exception(message);
            }
            return result;
        };
    
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
            day = expectFixedWidthInt(2, 1, 31, "expected day of month");
            break;
        case 'm':
            month = expectFixedWidthInt(2, 1, 12, "expected month of year");
            break;
        case 'M':
            switch(tolower(*context)) {
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
            year = expectFixedWidthInt(4, 1400, 2999, "expected year");
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
        cerr << "date was " << str << endl;
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
        throw ML::Exception("error converting time: t = %lld",
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


} // namespace Datacratic
