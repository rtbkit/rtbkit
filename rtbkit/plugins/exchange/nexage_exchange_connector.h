/* nexage_exchange_connector.h                                    -*- C++ -*-
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/common/creative_configuration.h"

namespace RTBKIT {


/*****************************************************************************/
/* NEXAGE EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for Nexage.  This speaks their flavour of the
    OpenRTB 2.1 protocol.
*/

struct NexageExchangeConnector: public OpenRTBExchangeConnector {
    NexageExchangeConnector(ServiceBase & owner, const std::string & name);
    NexageExchangeConnector(const std::string & name,
                            std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "nexage";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

    void init();

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler & connection,
                    const HttpHeader & header,
                    const std::string & payload);

#if 0
    virtual HttpResponse
    getResponse(const HttpAuctionHandler & connection,
                const HttpHeader & requestHeader,
                const Auction & auction) const;
#endif

    virtual double
    getTimeAvailableMs(HttpAuctionHandler & connection,
                       const HttpHeader & header,
                       const std::string & payload) {
        return 150.0;
    }

    /** This is the information that the Nexage exchange needs to keep
        for each campaign (agent).
    */
    struct CampaignInfo {
        Id seat;          ///< ID of the Nexage exchange seat
    };

    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;

    /** This is the information that Nexage needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        Id crid;                ///< ID Creative Id
        std::string iurl;       ///< Image URL for content checkin
        std::string  nurl;      ///< win notif url (optional)
        std::string  adm;       ///< XHTML markup  (optional)
        std::vector<std::string> adomain;    ///< Advertiser Domain
        OpenRTB::List<OpenRTB::ContentCategory> cat;    ///< Creative category Appendix 6.1
        OpenRTB::List<OpenRTB::CreativeAttribute>  attr;///< Creative attributes Appendix 6.3
    };

    typedef CreativeConfiguration<CreativeInfo> NexageCreativeConfiguration;

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    // Nexage win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);

    virtual bool bidRequestCreativeFilter(const BidRequest & request,
                                          const AgentConfig & config,
                                          const void * info) const;

  private:
    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;

    NexageCreativeConfiguration configuration_;
};



} // namespace RTBKIT
