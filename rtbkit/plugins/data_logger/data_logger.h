/* data_logger.h                                 -*- C++ -*-
   Jeremy Barnes, March 2011
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2012, 2013 Datacratic.  All rights reserved.

   DataLogger class
*/

#pragma once

#include <string>
#include <boost/function.hpp>
#include <boost/shared_ptr.hpp>

#include "soa/service/service_base.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/loop_monitor.h"
#include "rtbkit/core/monitor/monitor_provider.h"

#include "soa/logger/logger.h"

namespace RTBKIT {

struct DataLogger : public Datacratic::ServiceBase,
                    public MonitorProvider,
                    public Datacratic::Logger {
  DataLogger(const std::string & serviceName, 
             std::shared_ptr<Datacratic::ServiceProxies> proxies,
             bool monitor = true,
             size_t bufferSize = 65536);
    ~DataLogger();

    void init();
    void shutdown();

    void start(std::function<void ()> onStop = 0);

    void connectAllServiceProviders(const std::string & serviceClass,
                                    const std::string & epName);

    void unsafeDisableMonitor() {
        monitorProviderClient.disable();
    }

    // Subscription object to the named services
    Datacratic::ZmqNamedMultipleSubscriber multipleSubscriber;

    MonitorProviderClient monitorProviderClient;

    /* MonitorProvider interface */
    std::string getProviderClass() const;

    /** Used to overide the default getProviderIndicators implementation. */
    std::function<MonitorIndicator()> providerIndicatorFn;

    /** Returns a dumb all clear status. */
    virtual MonitorIndicator getProviderIndicators() const;

    bool monitor_;
    LoopMonitor loopMonitor_;
};

} // namespace RTKBIT
