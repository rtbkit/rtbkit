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
      monitorProviderClient(proxies->zmqContext, *this),
      monitor_(monitor)
{}

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
}

void
DataLogger::
shutdown()
{
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
getProviderName()
    const
{
    return serviceName();
}

Json::Value
DataLogger::
getProviderIndicators()
    const
{
    bool status(true);

    for (const auto & pair: multipleSubscriber.subscribers) {
        if (pair.second->getConnectionState()
            == ZmqNamedSocket::ConnectionState::DISCONNECTED) {
            status = false;
            break;
        }
    }

    Json::Value indicators;
    indicators["status"] = status ? "ok" : "failure";

    return indicators;
}
