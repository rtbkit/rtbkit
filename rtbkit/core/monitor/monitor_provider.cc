/* monitor_provider.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#include <iostream>
#include <boost/algorithm/string/trim.hpp>
#include <jml/arch/exception_handler.h>
#include "soa/service/service_base.h"
#include "monitor_provider.h"

using namespace std;

namespace RTBKIT {

MonitorProviderClient::
MonitorProviderClient(const std::shared_ptr<zmq::context_t> & context,
                      MonitorProvider & provider)
        : MultiRestProxy(context),
          provider_(provider),
          inhibit_(false)
{
    restUrlPath_ = "/v1/services/" + provider.getProviderClass();
}

MonitorProviderClient::
~MonitorProviderClient()
{
    shutdown();
}

void
MonitorProviderClient::
init(std::shared_ptr<ConfigurationService> & config,
     const std::string & serviceClass,
     bool localized)
{
    addPeriodic("MonitorProviderClient::postStatus", 1.0,
                std::bind(&MonitorProviderClient::postStatus, this),
                true);

    MultiRestProxy::init(config);
    MultiRestProxy::connectAllServiceProviders(serviceClass, "zeromq", localized);
}

void
MonitorProviderClient::
shutdown()
{
    MultiRestProxy::shutdown();
}

void
MonitorProviderClient::
postStatus()
{
    if (inhibit_) return;

    string payload = provider_.getProviderIndicators().toJson().toString();

    MultiRestProxy::OnResponse onResponse; // no-op
    push(onResponse, "POST", restUrlPath_, RestParams(), payload);
}

} // namespace RTBKIT
