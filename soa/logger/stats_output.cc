/* stats_output.cc
   Jeremy Barnes, 8 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Various outputs to write stats.
*/

#include "stats_output.h"
#include <boost/make_shared.hpp>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* CONSOLE STATS OUTPUT                                                      */
/*****************************************************************************/

ConsoleStatsOutput::
ConsoleStatsOutput (bool debug)
    : debug(debug),
      lastSeconds(Date::now())
{
}

void
ConsoleStatsOutput::
logMessage(const string & channel, const string & message)
{
    if (debug) {
        cerr << channel << " => " << message << endl;
    }

    lock_guard<mutex> guard (lock);

    auto it = logStats.find(channel);
    if (it == logStats.end()) {
        it = logStats.insert(make_pair(channel, StatItem(channel))).first;
    }

    it->second.messages++;
    it->second.bytes += message.size();
}

void
ConsoleStatsOutput::
dumpStats ()
{
    vector<StatItem> stats;

    {
        lock_guard<mutex> guard (lock);
        for (auto it = logStats.begin(), end = logStats.end(); it != end; ++it) {
            stats.push_back(it->second);
        }
        logStats.clear();
    }

    double seconds = Date::now().secondsSince(lastSeconds);
    sort(stats.begin(), stats.end(), [](const StatItem& lhs, const StatItem& rhs) -> bool {
            return lhs.bytes > rhs.bytes;
        });

    cerr << Date::now() << endl;
    for (auto it = stats.begin(), end = stats.end(); it != end; ++it) {
        uint64_t count = it->messages;
        double kb = it->bytes / 1024.0;

        cerr << ML::format("%-20s: %6lld msgs %8.2fkb\trate: %8.2f/s %8.2fkb/s",
                           it->channel.c_str(), (long long)count,
                           kb, count/seconds, kb/seconds) << endl;
    }

    lastSeconds = Date::now();
}


/*****************************************************************************/
/* CARBON STATS OUTPUT                                                       */
/*****************************************************************************/

CarbonStatsOutput::
CarbonStatsOutput(const string& carbonConnection,
                  const string& carbonPrefix)
    : EventRecorder("",
                    std::make_shared<CarbonEventService>
                    (carbonConnection, carbonPrefix))
{
    recordHit("loggerUp");        
}

CarbonStatsOutput::
CarbonStatsOutput(std::shared_ptr<EventService> events,
                  string prefix)
    : EventRecorder(prefix, events)
{
    recordHit("loggerUp");        
}

void
CarbonStatsOutput::
logMessage(const string & channel, const string & message)
{
    recordHit(channel);
}

void
CarbonStatsOutput::
recordBytesWrittenToFile (const string& file, size_t bytes)
{
    static const string prefix ("bytesWrittenToDisk.");
    EventRecorder::recordCount(bytes, prefix + file);
}

void
CarbonStatsOutput::
recordLevel (const string& name, double val)
{
    EventRecorder::recordLevel(val, name);
}

} // namespace Datacratic
