/* monitor_provider_proxy.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#include "monitor_provider_proxy.h"

using namespace std;

namespace RTBKIT {

MonitorProviderProxy::
~MonitorProviderProxy()
{
    shutdown();
}

void
MonitorProviderProxy::
init(std::shared_ptr<ConfigurationService> & config,
     const vector<string> & serviceNames)
{
    addPeriodic("MonitorProviderProxy::checkStatus", 1.0,
                bind(&MonitorProviderProxy::checkStatus, this),
                true);

    vector<string> monitorServiceNames;
    for (const string & serviceName: serviceNames) {
        string providerName = serviceName + "/monitor-provider";
        monitorServiceNames.push_back(providerName);
        providerStatus.insert(providerName);
    }

    RestMultiProxy::init(config, monitorServiceNames);
}

void
MonitorProviderProxy::
shutdown()
{
    sleepUntilIdle();
    RestMultiProxy::shutdown();
}

void
MonitorProviderProxy::
checkStatus()
{
    Guard(requestLock);

    if (pendingRequests > 0) {
        cerr << ML::format("MonitorProviderProxy::checkStatus: %d requests are"
                           " still active\n", pendingRequests);
        for (const string & epName: providerStatus) {
            cerr << "  request to endpoint '" << epName << "' still active"
                 << endl;
        }
    }
    else {
        pendingRequests = connections.size();
        responses.clear();
        for (auto & it: connections) {
            RestMultiProxy::OnDone onDone
                = bind(&MonitorProviderProxy::onResponseReceived,
                       this, it.first,
                       placeholders::_1, placeholders::_2, placeholders::_3);
            push(it.first, onDone, "GET", "/status");
        }
    }
}

void
MonitorProviderProxy::
onResponseReceived(string serviceName, exception_ptr ext,
                   int responseCode, const string & body)
{
    Guard(requestLock);

    responses.emplace_back(MonitorProviderResponse(serviceName,
                                                   responseCode,
                                                   body));
    pendingRequests--;
    providerStatus.erase(serviceName);

    if (pendingRequests == 0) {
        subscriber_.onProvidersStatusLoaded(responses);
    }
}

} // RTB
