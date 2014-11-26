/* time_arithmetic.h                                               -*- C++ -*-
   Jeremy Barnes, 4 April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Arithmetic functions for time.
*/

#ifndef __logger__periodic_utils_h__
#define __logger__periodic_utils_h__

#include "date.h"
#include "value_description_fwd.h"

namespace Datacratic {

enum TimeGranularity {
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS,
    DAYS,
    WEEKS,
    MONTHS,
    YEARS
};

TimeGranularity operator + (TimeGranularity granularity, int steps);
inline TimeGranularity operator - (TimeGranularity granularity, int steps)
{
    return (granularity + (-steps));
}

/** Returns whether one granularity unit can be translated to the other. */
bool canTranslateGranularity(TimeGranularity sourceGranularity,
                             TimeGranularity destGranularity);

/** Number of units of one granularity that first in the other granularity. */
int granularityMultiplier(TimeGranularity sourceGranularity,
                          TimeGranularity destGranularity);

/** Returns the number of units of "destGranularity" when translated from
 * "sourceGranularity". */
inline int translateGranularity(TimeGranularity sourceGranularity,
                                int sourcePeriod,
                                TimeGranularity destGranularity)
{
    return sourcePeriod * granularityMultiplier(sourceGranularity,
                                                destGranularity);
}

/** Calculate when the next period will be. */
std::pair<Date, double>
findPeriod(Date now, TimeGranularity granularity, double interval);

std::pair<Date, double>
findPeriod(Date now, const std::string & granularityName);

/** Generate the filename for the given date.  The pattern is interpreted
    like the DATE command:

    %%     a literal %
    %a     locale's abbreviated weekday name (e.g., Sun)
    %A     locale's full weekday name (e.g., Sunday)
    %b     locale's abbreviated month name (e.g., Jan)
    %B     locale's full month name (e.g., January)
    %c     locale's date and time (e.g., Thu Mar  3 23:05:25 2005)
    %C     century; like %Y, except omit last two digits (e.g., 20)
    %d     day of month (e.g, 01)
    %D     date; same as %m/%d/%y
    %e     day of month, space padded; same as %_d
    %F     full date; same as %Y-%m-%d
    %g     last two digits of year of ISO week number (see %G)
    %G     year of ISO week number (see %V); normally useful only with %V
    %h     same as %b
    %H     hour (00..23)
    %I     hour (01..12)
    %j     day of year (001..366)
    %k     hour ( 0..23)
    %l     hour ( 1..12)
    %m     month (01..12)
    %M     minute (00..59)
    %n     a newline
    %p     locale's equivalent of either AM or PM; blank if not known
    %P     like %p, but lower case
    %r     locale's 12-hour clock time (e.g., 11:11:04 PM)
    %R     24-hour hour and minute; same as %H:%M
    %s     seconds since 1970-01-01 00:00:00 UTC
    %S     second (00..60)
    %t     a tab
    %T     time; same as %H:%M:%S
    %u     day of week (1..7); 1 is Monday
    %U     week number of year, with Sunday as first day of week (00..53)
    %V     ISO week number, with Monday as first day of week (01..53)
    %w     day of week (0..6); 0 is Sunday
    %W     week number of year, with Monday as first day of week (00..53)
    %x     locale's date representation (e.g., 12/31/99)
    %X     locale's time representation (e.g., 23:13:48)
    %y     last two digits of year (00..99)
    %Y     year
    %z     +hhmm numeric timezone (e.g., -0400)
    %:z    +hh:mm numeric timezone (e.g., -04:00)
    %::z   +hh:mm:ss numeric time zone (e.g., -04:00:00)
    %:::z  numeric time zone with : to necessary precision (e.g., -04,
    +05:30)
    %Z     alphabetic time zone abbreviation (e.g., EDT)

    By default, date pads numeric fields with zeroes.  The following
    optional flags may follow `%':
        
    -      (hyphen) do not pad the field
    _      (underscore) pad with spaces
    0      (zero) pad with zeros
    ^      use upper case if possible
    #      use opposite case if possible
*/
std::string filenameFor(const Date & date,
                        const std::string & pattern);
    
std::pair<TimeGranularity, double>
parsePeriod(const std::string & pattern);


/*****************************************************************************/
/* TIME PERIOD                                                               */
/*****************************************************************************/

struct TimePeriod {
    TimePeriod(const std::string & periodName);
    TimePeriod(const char * periodName);
    TimePeriod(TimeGranularity granularity = SECONDS, double number = 0);

    Date current(Date now = Date::now()) const;
    Date next(Date now) const;

    std::string toString() const;

    void parse(const std::string & val);

    TimePeriod operator + (const TimePeriod & other) const;

    TimePeriod & operator += (const TimePeriod & other)
    {
        TimePeriod newPeriod = *this + other;
        *this = newPeriod;
        return *this;
    }

    bool operator == (const TimePeriod & other) const
    {
        return interval == other.interval;
    }

    bool operator != (const TimePeriod & other) const
    {
        return ! operator == (other);
    }

    TimeGranularity granularity;
    double number;
    double interval;
};

// NOTE: this is defined in the value description library
ValueDescriptionT<TimePeriod> * getDefaultDescription(TimePeriod *);

inline Date operator + (Date d, TimePeriod p)
{
    return d.plusSeconds(p.interval);
}

inline Date & operator += (Date & d, TimePeriod p)
{
    d.addSeconds(p.interval);
    return d;
}

inline Date operator + (TimePeriod p, Date d)
{
    return d.plusSeconds(p.interval);
}

inline Date operator - (Date d, TimePeriod p)
{
    return d.plusSeconds(-p.interval);
}

inline Date & operator -= (Date & d, TimePeriod p)
{
    d.addSeconds(-p.interval);
    return d;
}

inline TimePeriod operator * (TimePeriod p, int factor)
{
    TimePeriod result = p;
    result.number *= factor;
    result.interval *= factor;
    return result;
}

inline Date operator % (Date d, TimePeriod p)
{
    return d.quantize(p.interval);
}


} // namespace Datacratic

#endif /*  __logger__periodic_utils_h__ */

