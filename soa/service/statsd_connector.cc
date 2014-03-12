/* statsd_connector.cc
   Nicolas Kruchten
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/service/statsd_connector.h"
#include "ace/INET_Addr.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include <iostream>


using namespace std;
using namespace ML;

namespace Datacratic {


/*****************************************************************************/
/* STATSD CONNECTOR                                                          */
/*****************************************************************************/

StatsdConnector::
StatsdConnector()
{
}

StatsdConnector::
StatsdConnector(const string& statsdAddr)
{
    open(statsdAddr);
}

StatsdConnector::
~StatsdConnector()
{
    sckt.close();
}

void
StatsdConnector::
open(const string& statsdAddr)
{
    sckt.close();
    addr = ACE_INET_Addr(statsdAddr.c_str());

    if(sckt.open(addr) == -1)
        throw Exception("could not create statsd udp socket");
}
    
void
StatsdConnector::
incrementCounter(const char* counterName, float sampleRate, int value)
{
    if (sampleRate < 1.0 && ((random() % 10000) / 10000.0) >= sampleRate)
        return;

    char msgBuf[1024];
    int res = snprintf(msgBuf, 1024, "%s:%d|c|@%.2f", counterName, value,
                       sampleRate);
    if (res >= 1024) {
        cerr << "invalid statsd counter name: " << counterName << endl;
        return;
    }
    
    int sendRes = sckt.send(msgBuf, res, addr, MSG_DONTWAIT);
    if (sendRes == -1) {
        cerr << "statsd message failure: " << strerror(errno)
             << endl;
    }
}

void
StatsdConnector::
recordGauge(const char* counterName, float sampleRate, float value)
{
    if (sampleRate < 1.0 && ((random() % 10000) / 10000.0) >= sampleRate)
        return;

    char msgBuf[1024];
    int res = snprintf(msgBuf, 1024, "%s:%f|ms", counterName, value);

    if (res >= 1024) {
        cerr << "invalid statsd counter name: " << counterName << endl;
        return;
    }
    
    int sendRes = sckt.send(msgBuf, res, addr, MSG_DONTWAIT);
    if (sendRes == -1) {
        cerr << "statsd message failure: " << strerror(errno)
             << endl;
    }
}

} // namespace Datacratic
