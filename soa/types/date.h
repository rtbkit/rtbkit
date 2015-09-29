/* date.h                                                          -*- C++ -*-
   Jeremy Barnes, 18 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Basic class that holds and manipulates a date.  Not designed for ultimate
   accuracy, but shouldn't be too bad.
*/

#pragma once

#include <chrono>
#include <string>
#include "boost/date_time/posix_time/posix_time.hpp"
#include "jml/utils/parse_context.h"
#include "jml/db/persistent_fwd.h"

struct ACE_Time_Value;

namespace Json {

struct Value;

} // namespace Json

namespace Datacratic {

typedef std::chrono::duration<double, std::ratio<1>> Seconds;

using boost::posix_time::ptime;
struct Opaque;

namespace JS {
struct JSValue;
} // namespace JS


/*****************************************************************************/
/* DATE                                                                      */
/*****************************************************************************/

struct Date {

    Date()
        : secondsSinceEpoch_(0.0)
    {
    }

    explicit Date(const ptime & date);
    explicit Date(const Opaque & value);
    Date(int year, int month, int day,
         int hour = 0, int minute = 0, int second = 0,
         double fraction = 0.0);
    explicit Date(JS::JSValue & value);
    explicit Date(const Json::Value & value);
    explicit Date(const ACE_Time_Value & time);

    static Date fromSecondsSinceEpoch(double numSeconds)
    {
        Date result;
        result.secondsSinceEpoch_ = numSeconds;
        return result;
    }

    static Date fromTimespec(const struct timespec & ts)
    {
        Date result;
        result.secondsSinceEpoch_ = (double(ts.tv_sec)
                                     + double(ts.tv_nsec) / 1000000000);
        return result;
    }

    static Date fromIso8601Week(int year, int week, int day = 1);

    static Date parseSecondsSinceEpoch(const std::string & date);

    static Date parseDefaultUtc(const std::string & date);
    static Date parseIso8601DateTime(const std::string & date);

    // Deprecated
    static Date parseIso8601(const std::string & date);

    static Date notADate();
    static Date positiveInfinity();
    static Date negativeInfinity();
    static Date now();
    static Date nowOld();

    bool isADate() const;

    double secondsSinceEpoch() const
    {
        return secondsSinceEpoch_;
    }

    std::string print(unsigned seconds_digits = 0) const;
    std::string print(const std::string & format) const;
    std::string printIso8601(unsigned int fraction = 3) const;
    std::string printRfc2616() const;
    std::string printClassic() const;

    bool operator == (const Date & other) const
    {
        return secondsSinceEpoch_ == other.secondsSinceEpoch_;
    }

    bool operator != (const Date & other) const
    {
        return ! operator == (other);
    }

    bool operator <  (const Date & other) const
    {
        return secondsSinceEpoch_ < other.secondsSinceEpoch_;
    }

    bool operator <= (const Date & other) const
    {
        return secondsSinceEpoch_ <= other.secondsSinceEpoch_;
    }

    bool operator >  (const Date & other) const
    {
        return secondsSinceEpoch_ > other.secondsSinceEpoch_;
    }

    bool operator >= (const Date & other) const
    {
        return secondsSinceEpoch_ >= other.secondsSinceEpoch_;
    }

    double operator - (const Date & other) const
    {
        return secondsSinceEpoch_ - other.secondsSinceEpoch_;
    }
    
    Date & setMin(Date other)
    {
        secondsSinceEpoch_ = std::min(secondsSinceEpoch_,
                                      other.secondsSinceEpoch_);
        return *this;
    }

    Date & setMax(Date other)
    {
        secondsSinceEpoch_ = std::max(secondsSinceEpoch_,
                                      other.secondsSinceEpoch_);
        return *this;
    }

    /** Quantize to the given fraction of a second.  For example,
        quantize(0.1) leaves only tenths of a second, whereas quantize(1)
        quantizes to the nearest second. */
    Date quantized(double fraction) const;
    Date & quantize(double fraction);

    Date & addSeconds(double interval)
    {
        secondsSinceEpoch_ += interval;
        return *this;
    }

    Date & addMinutes(double interval)
    {
        secondsSinceEpoch_ += interval * 60.0;
        return *this;
    }

    Date & addHours(double interval)
    {
        secondsSinceEpoch_ += interval * 3600.0;
        return *this;
    }

    Date & addDays(double interval)
    {
        secondsSinceEpoch_ += interval * 3600.0 * 24.0;
        return *this;
    }

    Date & addWeeks(double interval)
    {
        addDays(interval * 7.0);
        return *this;
    }

    Date plusSeconds(double interval) const
    {
        Date result = *this;
        result.addSeconds(interval);
        return result;
    }

    Date plusMinutes(double interval) const
    {
        Date result = *this;
        result.addSeconds(interval * 60.0);
        return result;
    }

    Date plusHours(double interval) const
    {
        Date result = *this;
        result.addSeconds(interval * 3600.0);
        return result;
    }

    Date plusDays(double interval) const
    {
        Date result = *this;
        result.addSeconds(interval * 3600.0 * 24.0);
        return result;
    }

    Date plusWeeks(double interval) const
    {
        return plusDays(interval * 7.0);
    }

    double secondsUntil(const Date & other) const
    {
        return other.secondsSinceEpoch_ - secondsSinceEpoch_;
    }

    double minutesUntil(const Date & other) const
    {
        static const double factor = 1.0 / 60.0;
        return secondsUntil(other) * factor;
    }

    double hoursUntil(const Date & other) const
    {
        static const double factor = 1.0 / 3600.0;
        return secondsUntil(other) * factor;
    }

    double daysUntil(const Date & other) const
    {
        static const double factor = 1.0 / 24.0 / 3600.0;
        return secondsUntil(other) * factor;
    }

    double secondsSince(const Date & other) const
    {
        return -secondsUntil(other);
    }

    double minutesSince(const Date & other) const
    {
        return -minutesUntil(other);
    }

    double hoursSince(const Date & other) const
    {
        return -hoursUntil(other);
    }

    double daysSince(const Date & other) const
    {
        return -daysUntil(other);
    }

    bool sameDay(const Date & other) const
    {
        return dayStart() == other.dayStart();
    }

    Date weekStart() const
    {
        int delta = weekday();
        return plusDays(-delta).dayStart();
    }
    Date iso8601WeekStart() const
    {
        int nbr = iso8601Weekday();
        return (nbr == 1
                ? dayStart()
                : plusDays(1-nbr).dayStart());
    }
    Date dayStart() const
    {
        static const double secPerDay = 24.0 * 3600.0;
        double day = secondsSinceEpoch_ / secPerDay;
        double startOfDay = floor(day);
        return fromSecondsSinceEpoch(startOfDay * secPerDay);
    }
    Date hourStart() const
    {
        static const double secPerHour = 3600.0;
        double hour = secondsSinceEpoch_ / secPerHour;
        double startOfHour = floor(hour);
        return fromSecondsSinceEpoch(startOfHour * secPerHour);
    }

    int hour() const;
    int minute() const;
    int second() const;
    int weekday() const;
    int iso8601Weekday() const;
    int dayOfMonth() const;
    int dayOfYear() const;
    int iso8601WeekOfYear() const;
    int monthOfYear() const;
    int year() const;

    int hourOfWeek() const;

    double fractionalSeconds() const
    {
        double whole_seconds;
        return modf(secondsSinceEpoch_ >= 0
                    ? secondsSinceEpoch_ : -secondsSinceEpoch_,
                    &whole_seconds);
    }

    long long wholeSecondsSinceEpoch() const
    {
        double whole_seconds;
        modf(secondsSinceEpoch_, &whole_seconds);
        return whole_seconds;
    }

    std::string printMonth() const;
    std::string printWeekday() const;
    std::string printYearAndMonth() const;

    static const boost::posix_time::ptime epoch;

    static Date expect_date(ML::Parse_Context & context,
                            const std::string & format);
    static bool match_date(ML::Parse_Context & context, Date & date,
                           const std::string & format);
    
    static double expect_time(ML::Parse_Context & context,
                              const std::string & format);
    static bool match_time(ML::Parse_Context & context,
                           double & time,
                           const std::string & format);

    static Date expect_date_time(ML::Parse_Context & context,
                                 const std::string & date_format,
                                 const std::string & time_format);

    static bool match_date_time(ML::Parse_Context & context,
                                Date & result,
                                const std::string & date_format,
                                const std::string & time_format);

    // DO NOT USE THIS FUNCTION. IT IS DEPRECATED.
    static Date parse_date_time(const std::string & date_time,
                                const std::string & date_format,
                                const std::string & time_format);

    // parse using strptime function. more compatible with the `print` format
    static Date parse(const std::string & date,
                      const std::string & format);

    size_t hash() const
    {
        return std::hash<double>() (secondsSinceEpoch_);
    }

    /** Convert to an ACE compatible time value. */
    ACE_Time_Value toAce() const;

    /** Convert to a boost posix time */
    boost::posix_time::ptime toBoost() const
    {
        return boost::posix_time::from_time_t(toTimeT())
            + boost::posix_time::microseconds
            (1000000 * fractionalSeconds());
    }
    
    /** Convert to a std::chrono::time_point<std::chrono::system_clock>
        for the standard timing functions.
    */
    std::chrono::time_point<std::chrono::system_clock>
    toStd() const
    {
        return std::chrono::system_clock::from_time_t(toTimeT())
            + std::chrono::microseconds(static_cast<long>(1000000 * fractionalSeconds()));
    }

    /** Convert to a time_t value (seconds).  Rounds fractional seconds
        down.
    */
    time_t toTimeT() const
    {
        return secondsSinceEpoch();
    }

    /** Construct from a time_t value. */
    static Date fromTimeT(const time_t & time)
    {
        return Date::fromSecondsSinceEpoch(time);
    }

    /** Convert to a tm structure. */
    tm toTm() const;

    /** Construct from a tm structure. */
    static Date fromTm(const tm & t);

    /** expects a ^[1-9][0-9]*[SMHd]$ string */
    void addFromString(std::string);

private:
    double secondsSinceEpoch_;
    Date(double);
};

std::ostream & operator << (std::ostream & stream, const Date & date);
std::istream & operator >> (std::istream & stream, Date & date);

ML::DB::Store_Writer & operator << (ML::DB::Store_Writer & store, Date date);
ML::DB::Store_Reader & operator >> (ML::DB::Store_Reader & store, Date & date);

namespace JS {

void to_js(JSValue & jsval, Date value);
Date from_js(const JSValue & val, Date *);

inline Date
from_js_ref(const JSValue & val, Date *)
{
    return from_js(val, (Date *)0);
}


} // namespace JS


/*****************************************************************************/
/* ISO8601PARSER                                                             */
/*****************************************************************************/

struct Iso8601Parser : public ML::Parse_Context
{
    static Date parseDateTimeString(const std::string & dateTimeStr)
    {
        Iso8601Parser parser(dateTimeStr);
        return parser.expectDateTime();
    }

    static Date parseTimeString(const std::string & timeStr)
    {
        Iso8601Parser parser(timeStr);
        return parser.expectTime();
    }

    Iso8601Parser(const std::string & dateStr)
        : Parse_Context(dateStr, dateStr.c_str(),
                        dateStr.c_str() + dateStr.size())
    {}

    Date expectDateTime();

    Date expectDate();
    bool matchYearDay(int & result);
    int expectYear();
    int expectMonth();
    int expectWeekNumber();
    int expectWeekDay();
    int expectMonthDay();
    int expectYearDay();

    int expectHours();
    int expectMinutes();
    bool matchMinutes(int & result);
    int expectSeconds();

    Date expectTime();
    int expectTimeZoneMinutes();
};

} // namespace Datacratic

namespace std {

template<>
struct hash<Datacratic::Date> {
    size_t operator () (const Datacratic::Date & date) const
    {
        return date.hash();
    }
};

} // namespace std
