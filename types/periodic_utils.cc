/** periodic_utils.cc
    Jeremy Barnes, 4 April 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "periodic_utils.h"
#include "jml/utils/parse_context.h"
#include <boost/tuple/tuple.hpp>

using namespace std;


namespace Datacratic {

std::pair<TimeGranularity, int>
parsePeriod(const std::string & pattern)
{
    TimeGranularity granularity;
    int number;

    ML::Parse_Context context(pattern,
                              pattern.c_str(),
                              pattern.c_str() + pattern.length());

    number = context.expect_int();
    
    if (number <= 0)
        context.exception("invalid time number: must be > 0");

    if (context.match_literal('x'))
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
    int n;
    boost::tie(p, n) = parsePeriod(period);
    return findPeriod(current, p, n);
}

std::pair<Date, double>
findPeriod(Date current, TimeGranularity granularity, int number)
{
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
        throw ML::Exception("that granualrity is not supported");
    }

    return make_pair(result, interval);
}

std::string
filenameFor(const Date & date, const std::string & pattern)
{
    return date.print(pattern);
}

} // namespace Datacratic
