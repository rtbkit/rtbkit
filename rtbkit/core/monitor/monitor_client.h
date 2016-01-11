/* monitor_client.h                                        -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#pragma once

#include <mutex>
#include <vector>
#include "soa/types/date.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/logs.h"

namespace RTBKIT {
    using namespace Datacratic;

/* This class connects to the Monitor service and queries it periodically to
 * deduce whether the current service (probably the Router) can continue
 * processing its client requests. */
struct MonitorClient : public RestProxy
{

    enum {
        DefaultCheckTimeout = 1,
        DefaultTolerance = 10,
    };

    MonitorClient(const std::shared_ptr<zmq::context_t> & context,
                  int checkTimeout = DefaultCheckTimeout)
        : RestProxy(context),
          checkTimeout_(checkTimeout),
          testMode(false), testResponse(false)
    {
        onDone = std::bind(&MonitorClient::onResponseReceived, this,
                           std::placeholders::_1, std::placeholders::_2,
                           std::placeholders::_3);
    };

    ~MonitorClient();
 
    void init(std::shared_ptr<ConfigurationService> & config,
              const std::string & serviceName = "monitor");

    /** shutdown the MessageLoop but make sure all requests have been
        completed beforehand. */
    void shutdown();

    /** this method tests whether the last status obtained by the Monitor is
        positive and fresh enough to continue operations */
    bool getStatus(double toleranceSec = DefaultTolerance) const;

    /* private members */

    /** method invoked periodically to trigger a request to the Monitor */
    void checkStatus();

    /** method executed when we receive the response from the Monitor */
    void onResponseReceived(std::exception_ptr ext,
                            int responseCode, const std::string & body);
    
    /** bound instance of onResponseReceived */
    RestProxy::OnDone onDone;

    /** the mutex used when pendingRequest is tested and modified */
    typedef std::unique_lock<std::mutex> Guard;
    mutable std::mutex requestLock;

    /** the timeout that determines whether the last check is too old */
    int checkTimeout_;

    /** the timestamp when "lastStatus" was last updated */
    Date lastCheck;

    /** the timestamp of the last successful check */
    Date lastSuccess;

    /** helper members to make testing of dependent services easier */
    bool testMode;
    bool testResponse;

    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;
};

} // RTB
