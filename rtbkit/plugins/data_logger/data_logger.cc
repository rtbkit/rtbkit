/* data_logger.cc
   Jeremy Barnes, March 2011
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2012, 2013 Datacratic.  All rights reserved.

   Launches the router's logger.
*/


#include "data_logger.h"


using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

DataLogger::
DataLogger(const string & serviceName, std::shared_ptr<ServiceProxies> proxies,
           bool monitor, size_t bufferSize)
    : ServiceBase(serviceName, proxies),
      Logger(proxies->zmqContext, bufferSize),
      multipleSubscriber(proxies->zmqContext),
      monitorProviderClient(proxies->zmqContext),
      monitor_(monitor),
      loopMonitor_(*this)
{
    monitorProviderClient.addProvider(this);
}

DataLogger::
~DataLogger()
{
    monitorProviderClient.shutdown();
    shutdown();
}

void
DataLogger::
init()
{
    Logger::init();
    monitorProviderClient.init(getServices()->config);

    multipleSubscriber.init(getServices()->config);
    multipleSubscriber.messageHandler
        = [&] (vector<zmq::message_t> && msg) {
        // forward to logger class
        vector<string> s;
        s.reserve(msg.size());
        for (auto & m: msg)
            s.push_back(m.toString());
        this->logMessageNoTimestamp(s);
    };
    loopMonitor_.init();
    loopMonitor_.addMessageLoop("logger", &messageLoop);
    //messageLoop.addSource("DataLogger::multipleSubscriber",
    //                      multipleSubscriber);
}

void
DataLogger::
start(std::function<void ()> onStop)
{
    Logger::start(onStop);
    multipleSubscriber.start();
    if(monitor_)
      monitorProviderClient.start();

    loopMonitor_.start();
}

void
DataLogger::
shutdown()
{
    loopMonitor_.shutdown();
    monitorProviderClient.shutdown();
    Logger::shutdown();
    multipleSubscriber.shutdown();
    
}

void
DataLogger::
connectAllServiceProviders(const string & serviceClass, const string & epName)
{
    multipleSubscriber.connectAllServiceProviders(serviceClass, epName);
}

/** MonitorProvider interface */
string
DataLogger::
getProviderClass()
    const
{
    return "rtbDataLogger";
}

MonitorIndicator
DataLogger::
getProviderIndicators()
    const
{
    if (providerIndicatorFn)
        return providerIndicatorFn();

    MonitorIndicator ind;

    ind.serviceName = serviceName();
    ind.status = true;
    ind.message = "Alive";

    return ind;
}
