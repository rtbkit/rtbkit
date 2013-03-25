/* rubicon_exchange_connector.h                                    -*- C++ -*-
   Jeremy Barnes, 12 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/http_exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* RUBICON EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for Rubicon.  This speaks their flavour of the
    OpenRTB 2.1 protocol.

    Configuration options are the same as the HttpExchangeConnector on which
    it is based.
*/

struct RubiconExchangeConnector: public HttpExchangeConnector {
    
    RubiconExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies);

    virtual std::string exchangeName() const
    {
        return "rubicon";
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
                              const Auction & auction,
                              const std::string & reason) const;

    virtual HttpResponse
    getErrorResponse(const HttpAuctionHandler & connection,
                     const Auction & auction,
                     const std::string & errorMessage) const;

    /** This is the information that the Rubicon exchange needs to keep
        for each campaign (agent).
    */
    struct CampaignInfo {
        Id seat;       ///< ID of the Rubicon exchange seat
    };

    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;
    
    /** This is the information that Rubicon needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        std::string adm;                  ///< Ad markup
        std::vector<std::string> adomain; ///< Advertiser domains
        Id crid;                          ///< Creative ID
        OpenRTB::List<OpenRTB::CreativeAttribute> attr; ///< Creative attributes
        std::string ext_creativeapi;      ///< Creative API
    };

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    // Rubicon win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);
};



} // namespace RTBKIT
