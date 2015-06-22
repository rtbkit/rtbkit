/** analytics_publisher.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics publisher used to send data to an analytics endpoint.
*/
#pragma once

#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>

#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/service_utils.h"

typedef std::unordered_map< std::string, bool > ChannelFilter;

/********************************************************************************/
/* ANALYTICS PUBLISHER                                                          */
/********************************************************************************/

struct AnalyticsPublisher : public Datacratic::MessageLoop {

    AnalyticsPublisher() : initialized(false) {}

    void init(const std::string & baseUrl, const int numConnections);
    bool initialized;

    void start();

    void shutdown();

    void syncChannelFilters();

    template<typename... Args>
    void publish(const std::string & channel, const Args & ... args)
    {
        if (!live) return;

        std::lock_guard<std::mutex> lock(mu);
        auto it = channelFilter.find(channel);
        if (it == channelFilter.end() || !it->second) return;

        std::stringstream ss;
        make_message(ss, args...);
        sendEvent(channel, ss.str());
    }

private:
    std::mutex mu;
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
