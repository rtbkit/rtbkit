/* gumgum_exchange_connector.h                             
   Anatoly Kochergin, 20 April 2013
   Copyright (c) 2013 GumGum Inc.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* GUMGUM EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for Gumgum.  

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct GumgumExchangeConnector: public HttpExchangeConnector {
    GumgumExchangeConnector(ServiceBase & owner, const std::string & name);
    GumgumExchangeConnector(const std::string & name,
                            std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "gumgum";
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

    struct CampaignInfo {
        Id seat;                ///< ID of the exchange seat
    };

    struct CreativeInfo {
        Id adid;                ///< ID for ad to be service if bid wins 
        std::string adm;        ///< Actual XHTML ad markup
        std::string nurl;       ///< Win notice URL
    };
};



} // namespace RTBKIT
