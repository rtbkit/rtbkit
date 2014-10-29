/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/
#pragma once

#include <string>
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

struct AnalyticsClient : public Datacratic::MessageLoop {

    std::string baseUrl;
    std::shared_ptr<Datacratic::HttpClient> client;

    AnalyticsClient(const std::string & baseUrl);
    AnalyticsClient(int port = 40000, const std::string & address = "http://127.0.0.1");

    void init();
    void start();
    void shutdown();

    void sendEvent(const std::string & type, const std::string & event);

    template<typename Head>
    void make_message(std::stringstream & ss, Head && head)
    {
        ss << head;
    }

    template<typename Head, typename... Tail>
    void make_message(std::stringstream & ss, Head && head, Tail && ... tail)
    {
        ss << head << " ";
        make_message(ss, std::forward<Tail>(tail)...);
    }

    template<typename... Args>
    void publish(const std::string & channel, Args && ... args)
    {
        std::stringstream ss;
        make_message(ss, std::forward<Args>(args)...);
        sendEvent(channel, ss.str());
    }


};


struct AnalyticsRestEndpoint : public Datacratic::ServiceBase, public Datacratic::RestServiceEndpoint {

    AnalyticsRestEndpoint(std::shared_ptr<Datacratic::ServiceProxies> proxies, const std::string & serviceName);

    void init();
    
    std::pair<std::string, std::string>
    bindTcp();

    std::string addEvent(const std::string & type,
                         const std::string & event);

    void start();

    void shutdown();

    Datacratic::RestRequestRouter router;
};

