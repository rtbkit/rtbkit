/** ad_server_connector_ex.cc                                 -*- C++ -*-
    Eric Robert, 03 April 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Example of a simple ad server connector.

*/

#include "rtbkit/plugins/adserver/http_adserver_connector.h"
#include "rtbkit/common/auction_events.h"
#include "soa/jsoncpp/value.h"

namespace Datacratic { struct ServiceProxies; struct ServiceProxyArguments; }

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
    MockAdServerConnector(const std::string & serviceName,
                          const std::shared_ptr<Datacratic::ServiceProxies> & proxies,
                          const Json::Value & json = Json::Value::null);

    MockAdServerConnector(Datacratic::ServiceProxyArguments & args,
                          const std::string & serviceName);

    ~MockAdServerConnector();

    void init(int winPort, int eventPort);
    void start();
    void shutdown();
    HttpAdServerResponse handleEvent(PostAuctionEvent const & event);
};

} // namepsace RTBKIT

