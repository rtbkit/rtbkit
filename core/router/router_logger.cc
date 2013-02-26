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
      monitorProviderEndpoint(*this, *this)
{}

RouterLogger::
~RouterLogger()
{
    monitorProviderEndpoint.shutdown();
    shutdown();
}

void
RouterLogger::
init(std::shared_ptr<ConfigurationService> config)
{
    Logger::init();
    monitorProviderEndpoint.init();

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
start()
{
    Logger::start();
    multipleSubscriber.start();
    monitorProviderEndpoint.bindTcp();
    monitorProviderEndpoint.start();
}

void
RouterLogger::
shutdown()
{
    Logger::shutdown();
    multipleSubscriber.shutdown();
}

void
RouterLogger::
connectAllServiceProviders(const string & serviceClass, const string & epName)
{
    multipleSubscriber.connectAllServiceProviders(serviceClass, epName);
}

Json::Value
RouterLogger::
getMonitorIndicators()
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
