/** analytics_endpoint.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics endpoint used to publish messages on different channels.
*/
#pragma once

#include <string>
#include <unordered_map>
#include <sstream>
#include <utility>

#include "rtbkit/common/analytics_publisher.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/service_utils.h"
#include "soa/service/port_range_service.h"
#include "soa/jsoncpp/value.h"
#include "soa/service/rest_request_router.h"

#include "boost/thread/locks.hpp"
#include "boost/thread/shared_mutex.hpp"

/********************************************************************************/
/* ANALYTICS REST ENDPOINT                                                      */
/********************************************************************************/

struct AnalyticsRestEndpoint : public Datacratic::ServiceBase,
                               public Datacratic::RestServiceEndpoint {

    AnalyticsRestEndpoint(std::shared_ptr<Datacratic::ServiceProxies> proxies,
                          const std::string & serviceName);

    void init();

    void initChannels(ChannelFilter & channels) ;

    std::pair<std::string, std::string> bindTcp(int port = 0);

    void start();

    void shutdown();

    Json::Value listChannels() const;

    Json::Value enableChannel(const std::string & channel);
    Json::Value disableChannel(const std::string & channel);
    Json::Value enableAllChannels();
    Json::Value disableAllChannels();

private:
    std::string addEvent(const std::string & channel,
                         const std::string & event) const;

    std::string print(const std::string & channel,
                      const std::string & event) const;

    Datacratic::RestRequestRouter router;

    ChannelFilter channelFilter;
    mutable boost::shared_mutex access;
};

