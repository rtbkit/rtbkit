/* fbx_exchange_connector.h                                    -*- C++ -*-
   Jean-Sebastien Bejeau, 27 June 2013

*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {

/*****************************************************************************/
/* FBX EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Generic exchange connector using the FBX protocol.

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct FBXExchangeConnector : public HttpExchangeConnector {
	FBXExchangeConnector(ServiceBase & owner, const std::string & name);
	FBXExchangeConnector(const std::string & name,
	                     std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "fbx";
    }

    std::string exchangeName() const {
        return exchangeNameString();
    }

    std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

    double getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload) {
    	return 120;
    }

    double getRoundTripTimeMs(HttpAuctionHandler & handler,
                              const HttpHeader & header) {
    	return 5;
    }

    HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const;

};



} // namespace RTBKIT
