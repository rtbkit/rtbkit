/* router_logger.h
   Rémi Attab and Jeremy Barnes, March 2011
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2012, 2013 Datacratic.  All rights reserved.

   RouterLogger class
*/

#pragma once

#include <string>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "soa/service/service_base.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "rtbkit/core/monitor/monitor_provider.h"

#include "soa/logger/logger.h"

namespace RTBKIT {

struct RouterLogger : public Datacratic::ServiceBase,
                      public MonitorProvider,
                      public Datacratic::Logger {
    RouterLogger(std::shared_ptr<Datacratic::ServiceProxies> proxies);
    ~RouterLogger();

    void init(std::shared_ptr<Datacratic::ConfigurationService> config);
    void shutdown();

    void start(std::function<void ()> onStop = 0);

    void connectAllServiceProviders(const std::string & serviceClass,
                                    const std::string & epName);

    // Subscription object to the named services
    Datacratic::ZmqNamedMultipleSubscriber multipleSubscriber;

    MonitorProviderClient monitorProviderClient;

    /* MonitorProvider interface */
    std::string getProviderName() const;
    Json::Value getProviderIndicators() const;
};

} // namespace RTKBIT
