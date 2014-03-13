/** mock_adserver_connector.cc                                 -*- C++ -*-
    Eric Robert, 03 April 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Example of a simple ad server connector.

*/

#include "mock_adserver_connector.h"


using namespace RTBKIT;

MockAdServerConnector::
MockAdServerConnector(std::shared_ptr<ServiceProxies> const & proxies, Json::Value const & json) :
    HttpAdServerConnector(json.get("name", "mock-adserver").asString(), proxies),
    publisher(getServices()->zmqContext) {
}

void MockAdServerConnector::init(int port) {
    auto services = getServices();

    // Initialize our base class
    HttpAdServerConnector::init(services->config);

    // Prepare a simple JSON handler that already parsed the incoming HTTP payload so that it can
    // create the requied post auction object.
    auto handleEvent = [&](const Datacratic::HttpHeader & header,
                           const Json::Value & json,
                           const std::string & text) {
        this->handleEvent(PostAuctionEvent(json));
    };
    registerEndpoint(port, handleEvent);

    // Publish the endpoint now that it exists.
    HttpAdServerConnector::bindTcp();
    
    // And initialize the generic publisher on a predefined range of ports to try avoiding that
    // collision between different kind of service occurs.
    publisher.init(services->config, serviceName() + "/logger");
    publisher.bindTcp(services->ports->getRange("adServer.logger"));
}

void MockAdServerConnector::start() {
    recordLevel(1.0, "up");
    HttpAdServerConnector::start();
    publisher.start();
}


void MockAdServerConnector::shutdown() {
    HttpAdServerConnector::shutdown();
    publisher.shutdown();
}


void MockAdServerConnector::handleEvent(PostAuctionEvent const & event) {
    if(event.type == PAE_WIN) {
        publishWin(event.auctionId,
                   event.adSpotId,
                   event.winPrice,
                   event.timestamp,
                   Json::Value(),
                   event.uids,
                   event.account,
                   Date::now());

        Date now = Date::now();
        publisher.publish("WIN", now.print(3), event.auctionId.toString(), event.winPrice.toString(), "0");
    }

    if (event.type == PAE_CAMPAIGN_EVENT) {
        publishCampaignEvent(event.label,
                             event.auctionId,
                             event.adSpotId,
                             event.timestamp,
                             Json::Value(),
                             event.uids);
    }
}

namespace {

struct AtInit {
    AtInit()
    {
        AdServerConnector::registerFactory("mock", [](std::shared_ptr<ServiceProxies> const & proxies,
                                                      Json::Value const & json) {
            auto server = new MockAdServerConnector(proxies, json);

            int port = json.get("port", "12340").asInt();
            server->init(port);
            return server;
        });
    }
} atInit;

}

