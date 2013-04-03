/* router_logger.cc
   Rémi Attab and Jeremy Barnes, March 2011
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2012, 2013 Datacratic.  All rights reserved.

   Launches the router's logger.
*/


#include "router_logger.h"


using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

RouterLogger::
RouterLogger(std::shared_ptr<ServiceProxies> proxies)
    : ServiceBase("router_logger", proxies),
      Logger(proxies->zmqContext),
      multipleSubscriber(proxies->zmqContext),
      monitorProviderClient(proxies->zmqContext, *this)
{}

RouterLogger::
~RouterLogger()
{
    monitorProviderClient.shutdown();
    shutdown();
}

void
RouterLogger::
init(std::shared_ptr<ConfigurationService> config)
{
    Logger::init();
    monitorProviderClient.init(config);

    multipleSubscriber.init(config);
    multipleSubscriber.messageHandler
        = [&] (vector<zmq::message_t> && msg) {
        // forward to logger class
        vector<string> s;
        s.reserve(msg.size());
        for (auto & m: msg)
            s.push_back(m.toString());
        this->logMessageNoTimestamp(s);
    };

    //messageLoop.addSource("RouterLogger::multipleSubscriber",
    //                      multipleSubscriber);
}

void
RouterLogger::
start(std::function<void ()> onStop)
{
    Logger::start(onStop);
    multipleSubscriber.start();
    monitorProviderClient.start();
}

void
RouterLogger::
shutdown()
{
    monitorProviderClient.shutdown();
    Logger::shutdown();
    multipleSubscriber.shutdown();
}

void
RouterLogger::
connectAllServiceProviders(const string & serviceClass, const string & epName)
{
    multipleSubscriber.connectAllServiceProviders(serviceClass, epName);
}

/** MonitorProvider interface */
string
RouterLogger::
getProviderName()
    const
{
    return serviceName();
}

Json::Value
RouterLogger::
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
