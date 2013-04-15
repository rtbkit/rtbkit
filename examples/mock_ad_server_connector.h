/** ad_server_connector_ex.cc                                 -*- C++ -*-
    Eric Robert, 03 April 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Example of a simple ad server connector.

*/

#include "soa/service/service_utils.h"
#include "soa/service/service_base.h"
#include "soa/service/json_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "rtbkit/plugins/exchange/post_auction_proxy.h"
#include "rtbkit/common/auction_events.h"

namespace RTBKIT {

/******************************************************************************/
/* MOCK AD SERVER CONNECTOR                                                   */
/******************************************************************************/

/** Basic ad server connector that sits between the stream of wins (received from
    the exchange) and rest of the stack. Note that this assumes that the incoming
    message format is from the mock exchange sample.

 */

struct MockAdServerConnector : public Datacratic::ServiceBase
{
    MockAdServerConnector(
            std::shared_ptr<Datacratic::ServiceProxies> proxies,
            const std::string& serviceName) :
        ServiceBase(serviceName, proxies),
        exchange("Exchange"),
        proxy(getServices()->zmqContext),
        publisher(getServices()->zmqContext) {
    }

    MockAdServerConnector(
            Datacratic::ServiceProxyArguments & args,
            const std::string& serviceName) :
        ServiceBase(serviceName, args.makeServiceProxies()),
        exchange("Exchange"),
        proxy(getServices()->zmqContext),
        publisher(getServices()->zmqContext) {
    }


    void init(int port) {
        auto services = getServices();

        // Register this component globaly using the predefined 'adServer' name. Components that want
        // to connect to this can then use the general service discovery mechanism.
        registerServiceProvider(serviceName(), { "adServer" });
        services->config->removePath(serviceName());

        // Prepare a simple JSON handler that already parsed the incoming HTTP payload so that it can
        // create the requied post auction object.
        exchange.handlerFactory = [=]() {
            auto handler = [=](const Datacratic::HttpHeader & header,
                               const Json::Value & json,
                               const std::string & text,
                               Datacratic::AdHocJsonConnectionHandler * connection) {
                this->handleEvent(PostAuctionEvent(json));
            };

            return std::make_shared<Datacratic::AdHocJsonConnectionHandler>(handler);
        };

        exchange.init(port);

        // Now, initialize the proxy to communicate with the post auction loop.
        proxy.init(services->config);

        // And initialize the generic publisher on a predefined range of ports to try avoiding that
        // collision between different kind of service occurs.
        publisher.init(services->config, serviceName() + "/logger");
        publisher.bindTcp(services->ports->getRange("adServer/logger"));
    }


    void start() {
        recordLevel(1.0, "up");
        publisher.start();
    }


    void shutdown() {
        exchange.shutdown();
        proxy.shutdown();
        publisher.shutdown();
    }


    void handleEvent(PostAuctionEvent const & event) {
        if(event.type == PAE_WIN) {
            proxy.injectWin(event.auctionId,
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
    }


    /// Basic HTTP endpoint for receiving incoming wins from the mock exchange. We're using an helper
    /// class from SOA but any communication pipe will do.
    Datacratic::HttpEndpoint exchange;

    /// Specific handling for sending wins to the post auction loop. The post auction proxy makes the
    /// task of sending those events easier.
    PostAuctionProxy proxy;

    /// Generic publishing endpoint to forward wins to anyone registered. Currently, there's only the
    /// router that connects to this.
    Datacratic::ZmqNamedPublisher publisher;
};

} // namepsace RTBKIT

