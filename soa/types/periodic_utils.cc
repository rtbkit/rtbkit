/** periodic_utils.cc
    Jeremy Barnes, 4 April 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "periodic_utils.h"
#include "jml/utils/parse_context.h"
#include <boost/tuple/tuple.hpp>
#include "jml/utils/exc_assert.h"

using namespace std;
using namespace Datacratic;


namespace {

TimeGranularity granularities[] = {
    MILLISECONDS,
    SECONDS,
    MINUTES,
    HOURS,
    DAYS,
    WEEKS,
    MONTHS,
    YEARS
};

size_t maxGranularity = (sizeof(granularities) / sizeof(TimeGranularity));

int granularityIndex(TimeGranularity granularity)
{
    for (size_t i = 0; i < maxGranularity; i++) {
        if (granularities[i] == granularity) {
            return i;
        }
    }

    throw ML::Exception("granularity not found");
}

}

namespace Datacratic {

TimeGranularity operator + (TimeGranularity granularity, int steps)
{
    int i = granularityIndex(granularity);

    if (steps > 0) {
        if (i + steps >= maxGranularity) {
            throw ML::Exception("granularity index would be too high");
        }
    }
    else if (steps < 0) {
        if (i + steps < 0) {
            throw ML::Exception("granularity index would be negative");
        }
    }

    return granularities[i + steps];
}

/** Returns whether one granularity unit can be translated to the other. */
bool canTranslateGranularity(TimeGranularity sourceGranularity,
                             TimeGranularity destGranularity)
{
    /* We have 2 families of granularities: one based on
       milliseconds and the other based on months. They are incompatible with
       one another but they share characteristics which enables the
       simplification of this code :
       - the source granularity unit must be >= to the destination unit
       - both units must be from the same family
    */
    if (sourceGranularity == destGranularity) {
        return true;
    }
    else if (destGranularity > sourceGranularity) {
        return false;
    }
    else if (sourceGranularity < MONTHS) {
        return true;
    }
    else if (sourceGranularity == YEARS) {
        return destGranularity == MONTHS;
    }

    throw ML::Exception("we should never get here");
}

/** Number of units of one granularity that first in the other granularity. */
int granularityMultiplier(TimeGranularity sourceGranularity,
                          TimeGranularity destGranularity)
{
    if (!canTranslateGranularity(sourceGranularity, destGranularity)) {
        throw ML::Exception("specified granularities are incompatible with"
                            " each other");
    }

    int fromIndex = granularityIndex(destGranularity);
    int toIndex = granularityIndex(sourceGranularity);
    if (fromIndex > toIndex) {
        throw ML::Exception("the source granularity must be bigger that the"
                            " destination");
    }

    int multiplier(1);

    const int msMults[] = { 1000, 60, 60, 24, 7 };
    const int monthsMults[] = { 12 };
    const int * multipliers;
    if (sourceGranularity < MONTHS) {
        multipliers = msMults;
    }
    else {
        multipliers = monthsMults;
    }

    int offset = fromIndex >= 6 ? 6 : 0;
    for (int i = fromIndex; i < toIndex; i++) {
        multiplier *= multipliers[i - offset];
    }

    return multiplier;
}

std::pair<TimeGranularity, double>
parsePeriod(const std::string & pattern)
{
    TimeGranularity granularity;
    double number;

    ML::Parse_Context context(pattern,
                              pattern.c_str(),
                              pattern.c_str() + pattern.length());

    number = context.expect_double();
    
    //if (number <= 0)
    //    context.exception("invalid time number: must be > 0");

    if (context.match_literal('x') || context.match_literal("ms"))
	granularity = MILLISECONDS;
    else if (context.match_literal('s'))
        granularity = SECONDS;
    else if (context.match_literal('m'))
        granularity = MINUTES;
    else if (context.match_literal('h'))
        granularity = HOURS;
    else if (context.match_literal('d'))
        granularity = DAYS;
    else if (context.match_literal('w'))
        granularity = WEEKS;
    else if (context.match_literal('M'))
        granularity = MONTHS;
    else if (context.match_literal('y'))
        granularity = YEARS;
    else context.exception("excepted h, m, s, d, w, M, y or x");

    context.expect_eof();

    return make_pair(granularity, number);
}

std::pair<Date, double>
findPeriod(Date current, const std::string & period)
{
    TimeGranularity p;
    double n;
    boost::tie(p, n) = parsePeriod(period);
    return findPeriod(current, p, n);
}

std::pair<Date, double>
findPeriod(Date current, TimeGranularity granularity, double number_)
{
    if (number_  == 0)
        return make_pair(current, 0);

    int64_t number = number_;

    // Make sure it's an integer number of seconds
    ExcAssertEqual(number, number_);

    // Find where the current period starts

    tm t = current.toTm();

    // Remove fractional seconds
    Date result = current;
    int64_t seconds = result.secondsSinceEpoch();
    result = Date::fromSecondsSinceEpoch(seconds);

    //Date unadjusted = result;

    double interval;

    switch (granularity) {
    case DAYS:
        current.quantize(3600 * 24);
        interval = 3600 * 24 * number;
        break;

        if (number != 1)
            throw ML::Exception("only 1d is supported for days");
        // Go to the start of the day
        result.addSeconds(-(3600 * t.tm_hour + 60 * t.tm_min + t.tm_sec));
        
        // Go to day
        interval = 3600 * 24 * number;

        break;
    case HOURS:
        // Go to the start of the hour
        result.addSeconds(-(60 * t.tm_min + t.tm_sec));

        // Go to hour number n of the day
        result.addSeconds(-3600 * (t.tm_hour % number));

        interval = 3600 * number;

        break;

    case MINUTES:
        // Go to the start of the minute
        result.addSeconds(-t.tm_sec);

        // Go to the next multiple of minutes
        result.addSeconds(-60 * (t.tm_min % number));

        interval = 60 * number;

        break;

    case SECONDS:
        //cerr << "seconds: t.tm_sec = " << t.tm_sec    
        //     << " number = " << number << " before = "
        //     << result;
        result.addSeconds(-(t.tm_sec % number));
        //cerr <<" after = " << result << " current = "
        //     << current << endl;

        interval = number;
        break;

    case MILLISECONDS:
	interval = number / 1000.0;

	result = current.quantized(interval);
	if (result > current)
	    result.addSeconds(-interval);

	break;

    default:
        throw ML::Exception("that granularity is not supported");
    }

    return make_pair(result, interval);
}

std::string
filenameFor(const Date & date, const std::string & pattern)
{
    return date.print(pattern);
}

/*****************************************************************************/
/* TIME PERIOD                                                               */
/*****************************************************************************/

TimePeriod::
TimePeriod(const std::string & periodName)
{
    parse(periodName);
}

TimePeriod::
TimePeriod(const char * periodName)
{
    parse(periodName);
}

TimePeriod::
TimePeriod(TimeGranularity granularity, double number)
    : granularity(granularity), number(number),
      interval(findPeriod(Date(), granularity, number).second)
{
}

Date
TimePeriod::
current(Date now) const
{
    return findPeriod(now, granularity, number).first;
}

Date
TimePeriod::
next(Date now) const
{
    return current(now).plusSeconds(interval);
}

std::string
TimePeriod::
toString() const
{
    string result = boost::lexical_cast<string>(number);//ML::format("%f", number);
    switch (granularity) {
    case MILLISECONDS:  result += "ms";  return result;
    case SECONDS:       result += 's';   return result;
    case MINUTES:       result += 'm';   return result;
    case HOURS:         result += 'h';   return result;
    case DAYS:          result += 'd';   return result;
    case WEEKS:         result += 'w';   return result;
    case MONTHS:        result += 'M';   return result;
    case YEARS:         result += 'y';   return result;
    default:
        throw ML::Exception("unknown time period");
    }
}

void
TimePeriod::
parse(const std::string & str)
{
    std::tie(granularity, number) = parsePeriod(str);
    interval = findPeriod(Date(), granularity, number).second;
}

TimePeriod
TimePeriod::
operator + (const TimePeriod & other)
    const
{
    TimePeriod result;

    int thisIndex = granularityIndex(granularity);
    int otherIndex = granularityIndex(other.granularity);

    if (thisIndex < otherIndex) {
        int multiplier
            = granularityMultiplier(other.granularity, granularity);
        result.number = number + multiplier * other.number;
        result.granularity = granularity;
    }
    else {
        int multiplier
            = granularityMultiplier(granularity, other.granularity);
        result.number = other.number + multiplier * number;
        result.granularity = other.granularity;
    }

    result.interval = interval + other.interval;

    return result;
}

} // namespace Datacratic
