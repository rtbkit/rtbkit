/* standard_adserver_connector.h
   Wolfgang Sourdeau, March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#pragma once

#include <memory>
#include <string>

#include "rtbkit/plugins/adserver/http_adserver_connector.h"
#include "rtbkit/common/analytics_publisher.h"

namespace RTBKIT { struct Analytics; }
namespace Datacratic { struct ServiceProxies; }
namespace Json { struct Value; }

namespace RTBKIT {

struct StandardAdServerConnector : public HttpAdServerConnector
{
    StandardAdServerConnector(std::shared_ptr<Datacratic::ServiceProxies> & proxy,
                              const std::string & serviceName = "StandardAdServer");

    StandardAdServerConnector(std::string const & serviceName,
                              std::shared_ptr<Datacratic::ServiceProxies> const & proxy,
                              Json::Value const & json);

    void init(int winsPort, int eventsPort);
    void start();
    void shutdown();

    /** Handle events received on the win port */
    HttpAdServerResponse handleWinRq(const HttpHeader & header,
                                     const Json::Value & json,
                                     const std::string & jsonStr);

    /** Handle events received on the events port */
    HttpAdServerResponse handleDeliveryRq(const HttpHeader & header,
                                          const Json::Value & json,
                                          const std::string & jsonStr);

    void publishError(HttpAdServerResponse & resp);

    /** */
    AnalyticsPublisher analyticsPublisher_;

private :

    void init(int winsPort, int eventsPort, bool verbose,
                    bool analyticsPublisherOn = false, int analyticsPublisherConnections = 1);
    virtual void initEventType(const Json::Value &json);

    std::map<std::string, std::string> eventType;
    bool verbose;
};

} // namespace RTBKIT
