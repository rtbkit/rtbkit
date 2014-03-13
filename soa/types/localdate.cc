/* localdate.cc
   Wolfgang Sourdeau, 10 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/


#include <time.h>
#include <mutex>
#include <boost/date_time/local_time/tz_database.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <jml/arch/exception.h>

#include "date.h"

#include "localdate.h"

using namespace boost::local_time;
using namespace boost::posix_time;
using namespace std;
// using namespace ML;

namespace
{
    static tz_database tz_db;
    std::once_flag once ;
}

namespace Datacratic {

/*****************************************************************************/
/* DATE                                                                      */
/*****************************************************************************/

LocalDate::
LocalDate(double secondsSinceEpoch, const std::string & tzName)
    : secondsSinceEpoch_(secondsSinceEpoch), tzName_(tzName), tzOffset_(0)
{
    recomputeTZOffset();
}

double
LocalDate::
secondsSinceEpoch() const
{
    return secondsSinceEpoch_;
}

const string
LocalDate::
timezone() const
{
    return tzName_;
}

int
LocalDate::
tzOffset() const
{
    return tzOffset_;
}

void
LocalDate::
fillTM(struct tm & time) const
{
    time_t t = secondsSinceEpoch_ + tzOffset_;

    if (!gmtime_r(&t, &time))
        throw ML::Exception("problem with gmtime_r");
}

int
LocalDate::
hour() const
{
    tm time;

    fillTM(time);

    return time.tm_hour;
}

int
LocalDate::
dayOfMonth() const
{
    tm time;

    fillTM(time);

    return time.tm_mday;
}

int
LocalDate::
dayOfWeek() const
{
    tm time;

    fillTM(time);

    return time.tm_wday;
}

int
LocalDate::
year() const
{
    tm time;

    fillTM(time);

    return time.tm_year + 1900;
}

void
LocalDate::
recomputeTZOffset()
{
    if (tzName_ == "UTC") {
        tzOffset_ = 0;
        // dstOffset_= 0;
    }
    else {
        call_once(once, [&] {
                string zoneSpecFile = BOOST_TIMEZONES_DIR "/date_timezone_spec.csv";
                tz_db.load_from_file(zoneSpecFile);
            });

        time_zone_ptr tz = tz_db.time_zone_from_region(tzName_);
        if (tz == 0) {
            throw ML::Exception("time zone named '" + tzName_ + "' is not known");
        }
        time_duration offset = tz->base_utc_offset();
        tzOffset_ = offset.total_seconds();

        if (tz->has_dst()) {
            int yearOfLocalDate = year();
            int dst_start((tz->dst_local_start_time(yearOfLocalDate) - Date::epoch)
                          .total_seconds());
            int dst_end((tz->dst_local_end_time(yearOfLocalDate) - Date::epoch)
                        .total_seconds());

            // the southern areas of the world have their DST during northern
            // winter
            if (dst_start < dst_end) {
                if (secondsSinceEpoch_ >= dst_start
                    && dst_end > secondsSinceEpoch_)
                    tzOffset_ += tz->dst_offset().total_seconds();
            }
            else {
                if (secondsSinceEpoch_ >= dst_end
                    && dst_start > secondsSinceEpoch_)
                    tzOffset_ += tz->dst_offset().total_seconds();
            }
        }
    }

    // cerr << "timezone: " << tzName_
    //      << "; seconds: " << secondsSinceEpoch_
    //      << "; offset: " << tzOffset_
    //      << "; dst offset: " << dstOffset_
    //      << endl;
}

} // namespace Datacratic
