/** ad_server_connector_ex.cc                                 -*- C++ -*-
    Eric Robert, 03 April 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Example of a simple ad server connector.

*/

#include "soa/service/service_utils.h"
#include "soa/service/service_base.h"
#include "soa/service/json_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "rtbkit/plugins/adserver/http_adserver_connector.h"
#include "rtbkit/common/auction_events.h"

namespace RTBKIT {

/******************************************************************************/
/* MOCK AD SERVER CONNECTOR                                                   */
/******************************************************************************/

/** Basic ad server connector that sits between the stream of wins (received from
    the exchange) and rest of the stack. Note that this assumes that the incoming
    message format is from the mock exchange sample.

 */

struct MockAdServerConnector : public HttpAdServerConnector
{
    MockAdServerConnector(const std::string& serviceName,
                          std::shared_ptr<Datacratic::ServiceProxies> proxies)
        : HttpAdServerConnector(serviceName, proxies),
          publisher(getServices()->zmqContext) {
    }

    MockAdServerConnector(Datacratic::ServiceProxyArguments & args,
                          const std::string& serviceName)
        : HttpAdServerConnector(serviceName, args.makeServiceProxies()),
          publisher(getServices()->zmqContext) {
    }

    void init(int port) {
        auto services = getServices();

        // Prepare a simple JSON handler that already parsed the incoming HTTP payload so that it can
        // create the requied post auction object.
        auto handleEvent = [&](const Datacratic::HttpHeader & header,
                               const Json::Value & json,
                               const std::string & text) {
            this->handleEvent(PostAuctionEvent(json));
        };
        registerEndpoint(port, handleEvent);
        
        // And initialize the generic publisher on a predefined range of ports to try avoiding that
        // collision between different kind of service occurs.
        publisher.init(services->config, serviceName() + "/logger");
        publisher.bindTcp(services->ports->getRange("adServer/logger"));

        HttpAdServerConnector::init(services->config);
    }


    void start() {
        recordLevel(1.0, "up");
        publisher.start();
    }


    void shutdown() {
        publisher.shutdown();
        HttpAdServerConnector::shutdown();
    }


    void handleEvent(PostAuctionEvent const & event) {
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
    }


    /// Generic publishing endpoint to forward wins to anyone registered. Currently, there's only the
    /// router that connects to this.
    Datacratic::ZmqNamedPublisher publisher;
};

} // namepsace RTBKIT

