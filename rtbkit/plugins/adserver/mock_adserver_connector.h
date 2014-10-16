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

    MockAdServerConnector(std::string const & serviceName, std::shared_ptr<ServiceProxies> const & proxies,
                          Json::Value const & json);

    void init(int winPort, int eventPort);
    void start();
    void shutdown();
    HttpAdServerResponse handleEvent(PostAuctionEvent const & event);

    /// Generic publishing endpoint to forward wins to anyone registered. Currently, there's only the
    /// router that connects to this.
    Datacratic::ZmqNamedPublisher publisher;
};

} // namepsace RTBKIT

