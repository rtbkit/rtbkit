/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/
#pragma once

#include <string>

#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"

struct AnalyticsClient : public Datacratic::MessageLoop {

    Datacratic::HttpClient client;

    AnalyticsClient(int port, const std::string & address);

    void sendWin(const std::string & body);

};


struct AnalyticsRestEndpoint : public Datacratic::RestServiceEndpoint {

    AnalyticsRestEndpoint(std::shared_ptr<ServiceProxies> proxies);


};
