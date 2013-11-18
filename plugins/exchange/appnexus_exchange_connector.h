/* appnexus_exchange_connector.h                                    -*- C++ -*-
   Eric Robert, 23 July 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <unordered_set>
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
                     const Auction & auction,
                     const std::string & errorMessage) const;

    /** This is the information that AppNexus needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo
    {
        uint32_t agency_id_ ;                       ///< Agency Id
        std::string buyer_creative_id_ ;            ///< Buyer Creative Id
        std::string html_snippet_;                  ///< landing Url
        std::string click_through_url_;             ///< Click
        std::unordered_set<int32_t> vendor_type_ ;  ///< Vendor Type
        std::unordered_set<int32_t> category_ ;     ///< Category
        std::unordered_set<int32_t> attribute_;     ///< Attribute
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
