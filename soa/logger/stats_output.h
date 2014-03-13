/* stats_output.h                                                -*- C++ -*-
   Jeremy Barnes, 8 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Output that logs stats to the console.
*/

#ifndef __logger__stats_output_h__
#define __logger__stats_output_h__

#include "logger.h"
#include "soa/service/carbon_connector.h"
#include "soa/service/service_base.h"
#include "soa/types/date.h"
#include <thread>


namespace Datacratic {


/*****************************************************************************/
/* CONSOLE STATS OUTPUT                                                      */
/*****************************************************************************/

//! \todo Gotta figure out the dump to console scheme.
struct ConsoleStatsOutput : public LogOutput {
    ConsoleStatsOutput (bool debug = false);

    virtual ~ConsoleStatsOutput ()  {}

private:

        struct StatItem {
            StatItem(const std::string& channel) :
                channel(channel),
                messages(0),
                bytes(0)
            {}

            std::string channel;
            uint64_t messages;
            uint64_t bytes;
        };

public:

    virtual void logMessage(const std::string & channel, const std::string & message);

    void dumpStats ();

    virtual void close() {}
    virtual Json::Value stats() const { return Json::Value (); }
    virtual void clearStats() {}

private :
    bool debug;
    std::map<std::string, StatItem> logStats;

    Date lastSeconds;
    std::mutex lock;
};


/*****************************************************************************/
/* CARBON STATS OUTPUT                                                       */
/*****************************************************************************/

struct CarbonStatsOutput : public LogOutput, public EventRecorder {

    CarbonStatsOutput (const std::string& carbonConnection,
                       const std::string& carbonPrefix);

    CarbonStatsOutput(std::shared_ptr<EventService> eventService,
                      std::string prefix);

    virtual ~CarbonStatsOutput ()  {}

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    void recordBytesWrittenToFile (const std::string& file, size_t bytes);

    void recordLevel (const std::string& name, double val);

    virtual void close() {}
    virtual Json::Value stats() const { return Json::Value (); }
    virtual void clearStats() {}

private:
    std::shared_ptr<EventService> eventService;
};




} // namespace Datacratic

#endif
