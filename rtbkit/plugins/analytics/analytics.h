/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/
#pragma once

#include <string>
#include <unordered_map>
#include <sstream>
#include <utility>

#include "soa/service/rest_service_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/service_utils.h"
#include "soa/service/service_utils.h"
#include "soa/service/port_range_service.h"
#include "soa/jsoncpp/value.h"
#include "soa/service/rest_request_router.h"

#include "boost/thread/locks.hpp"
#include "boost/thread/shared_mutex.hpp"

typedef std::unordered_map< std::string, bool > ChannelFilter;

/********************************************************************************/
/* ANALYTICS CLIENT                                                             */
/********************************************************************************/

struct AnalyticsClient : public Datacratic::MessageLoop {

    void init(const std::string & baseUrl);

    void start();

    void shutdown();

    void syncChannelFilters();
    
    template<typename... Args>
    void publish(const std::string & channel, const Args & ... args)
    {
        if (!live) return;
        if (!channelFilter[channel]) return;

        std::stringstream ss;
        make_message(ss, args...);
        sendEvent(channel, ss.str());
    }

private:
    std::shared_ptr<Datacratic::HttpClient> client;
    bool live;
    ChannelFilter channelFilter;

    void sendEvent(const std::string & channel, const std::string & event);

    void checkHeartbeat();

    template<typename Head>
    void make_message(std::stringstream & ss, const Head & head)
    {
        ss << head;
    }

    template<typename Head, typename... Tail>
    void make_message(std::stringstream & ss, const Head & head, const Tail & ... tail)
    {
        ss << head << " ";
        make_message(ss, tail...);
    }

};


struct AnalyticsSubscriber {
    void subscribe(const std::string & channel);
    void unsubscribe(const std::string & channel);
};


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

    Json::Value listChannels();

    Json::Value enableChannel(const std::string & channel);
    Json::Value disableChannel(const std::string & channel);
    Json::Value enableAllChannels();
    Json::Value disableAllChannels();

private:
    std::string addEvent(const std::string & channel,
                         const std::string & event);

    std::string print(const std::string & channel,
                      const std::string & event);

    Datacratic::RestRequestRouter router;

    ChannelFilter channelFilter;
    boost::shared_mutex access;
};

