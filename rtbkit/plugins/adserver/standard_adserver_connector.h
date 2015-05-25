/* standard_adserver_connector.h
   Wolfgang Sourdeau, March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/options_description.hpp>

#include "soa/service/carbon_connector.h"
#include "soa/service/service_base.h"
#include "soa/service/service_utils.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/types/date.h"

#include "rtbkit/plugins/adserver/http_adserver_connector.h"
#include "rtbkit/common/analytics_publisher.h"

namespace RTBKIT {

using namespace std;

struct StandardAdServerConnector : public HttpAdServerConnector
{
    StandardAdServerConnector(shared_ptr<Datacratic::ServiceProxies> & proxy,
                              const string & serviceName = "StandardAdServer");
    StandardAdServerConnector(std::string const & serviceName, std::shared_ptr<Datacratic::ServiceProxies> const & proxy,
                              Json::Value const & json);

    void init(int winsPort, int eventsPort);
    void start();
    void shutdown();

    /** Handle events received on the win port */
    HttpAdServerResponse handleWinRq(const HttpHeader & header,
                     const Json::Value & json, const string & jsonStr);

    /** Handle events received on the events port */
    HttpAdServerResponse handleDeliveryRq(const HttpHeader & header,
                          const Json::Value & json, const string & jsonStr);

    void publishError(HttpAdServerResponse & resp);

    /** */
    Datacratic::ZmqNamedPublisher publisher_;
    AnalyticsPublisher analytics_;

private :

    void init(int winsPort, int eventsPort, bool verbose, bool analyticsOn = false, int analyticsConnections = 1);
    virtual void initEventType(const Json::Value &json);

    std::map<std::string , std::string> eventType;  
    bool verbose;
};

} // namespace RTBKIT
