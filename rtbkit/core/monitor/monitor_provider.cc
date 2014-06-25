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
MonitorProviderClient(const std::shared_ptr<zmq::context_t> & context)
        : MultiRestProxy(context),
          inhibit_(false)
{
}

MonitorProviderClient::
~MonitorProviderClient()
{
    shutdown();
}

void
MonitorProviderClient::
addProvider(const MonitorProvider *provider)
{
    providers.push_back(provider);
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
disable()
{
    inhibit_ = true;
}

void
MonitorProviderClient::
postStatus()
{
    if (inhibit_) return;

    for (const MonitorProvider *provider: providers) {
        const string payload = provider->getProviderIndicators().toJson().toString();
        const string url = "/v1/services/" + provider->getProviderClass();

        MultiRestProxy::OnResponse onResponse; // no-op
        push(onResponse, "POST", url, RestParams(), payload);
    }
}

} // namespace RTBKIT
