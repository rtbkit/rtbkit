/* statsd_connector.h                                              -*- C++ -*-
   Send messages to statsd.
*/

#pragma once

#include "ace/SOCK_Dgram.h"
#include <string>

namespace Datacratic {


/*****************************************************************************/
/* STATSD CONNECTOR                                                          */
/*****************************************************************************/

/** Class that sends UDP packets to statsd for monitoring purposes.
*/

class StatsdConnector {
    ACE_SOCK_Dgram sckt;
    ACE_INET_Addr addr;

public:
    StatsdConnector();
    StatsdConnector(const std::string & statsdAddr);
    ~StatsdConnector();

    void open(const std::string & statsdAddr);

    void incrementCounter(const char* counterName, float sampleRate, int value=1 );
    void recordGauge(const char* counterName, float sampleRate, float gauge );
};


} // namespace Datacratic
