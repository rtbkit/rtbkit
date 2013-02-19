/* monitor_provider_proxy.h                                        -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include <mutex>
#include <set>
#include <vector>
#include "soa/types/date.h"

#include "rest_multi_proxy.h"

namespace RTBKIT {

struct MonitorProviderResponse
{
    MonitorProviderResponse(const std::string & newServiceName,
                            int newCode,
                            const std::string & newBody)
        : serviceName(newServiceName), code(newCode), body(newBody)
        {};
    std::string serviceName;
    int code;
    std::string body;

    bool operator ==(const MonitorProviderResponse & otherResponse) const
    {
        return (this == &otherResponse
                || (serviceName == otherResponse.serviceName
                    && code == otherResponse.code
                    && body == otherResponse.body));
    }
};


typedef std::vector<MonitorProviderResponse> MonitorProviderResponses;

struct MonitorProvidersSubscriber
{
    virtual void onProvidersStatusLoaded(const MonitorProviderResponses &
                                         responses) = 0;
};

/* This class connects to a list of monitor providers and pass their status at
 * once to an instance of MonitorProviderSubscriber. It is meant to be used
 * only from the Monitor service. */
struct MonitorProviderProxy : RestMultiProxy
{
    MonitorProviderProxy(const std::shared_ptr<zmq::context_t> & context,
                         MonitorProvidersSubscriber & subscriber)
        : RestMultiProxy(context),
          subscriber_(subscriber),
          pendingRequests(0)
        {};

    ~MonitorProviderProxy();

    void init(std::shared_ptr<ConfigurationService> & config,
              const std::vector<std::string> & serviceNames);
    void shutdown();

    void checkStatus();
    void onResponseReceived(std::string serviceName, std::exception_ptr ext,
                            int responseCode, const std::string & body);

    MonitorProvidersSubscriber & subscriber_;

    typedef std::unique_lock<std::mutex> Guard;
    mutable std::mutex requestLock;
    int pendingRequests;
    std::set<std::string> providerStatus;

    MonitorProviderResponses responses;
};

} // RTB
