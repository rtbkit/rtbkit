/* appnexus_exchange_connector.h                                    -*- C++ -*-
   Eric Robert, 23 July 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once
#include <unordered_set>
#include <boost/any.hpp>
#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {

/*****************************************************************************/
/* APPNEXUS EXCHANGE CONNECTOR                                               */
/*****************************************************************************/

/** Basic AppNexus exchange connector.

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct AppNexusExchangeConnector : public HttpExchangeConnector {
    AppNexusExchangeConnector(ServiceBase & owner, const std::string & name);
    AppNexusExchangeConnector(const std::string & name,
                              std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "appnexus";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload);

    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const;

    virtual HttpResponse
    getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                              const std::string & reason) const;

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const std::string & errorMessage) const;

    /** This is the information that AppNexus needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo
    {   // see https://wiki.appnexus.com/display/adnexusdocumentation/Bid+Response
    	int member_id_ ;                    // *must* have.The ID of the member whose creative is
    	                                    // chosen by the bidder from the "members" array in
    	                                    // the request.
        int creative_id_ ;                  // *must* have either this or:
        std::string creative_code_ ;        // The custom code of the creative passed
                                            //     into the creative service.
        std::string click_url_;             // The click URL to be associated with the creative
        std::string pixel_url_;             // The pixel URL to be associated with the creative
        std::unordered_set<int32_t> attrs_; // Attributes

    };

    virtual bool
    bidRequestCreativeFilter(const BidRequest & request,
                             const AgentConfig & config,
                             const void * info) const;

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

};


} // namespace RTBKIT
