/* bidswitch_exchange_connector.h                                    -*- C++ -*-
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#pragma once
#include <set>
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"

namespace RTBKIT {


/*****************************************************************************/
/* BIDSWITCH EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for BidSwitch.  This speaks their flavour of the
    OpenRTB 2.1 protocol.
*/

struct BidSwitchExchangeConnector: public OpenRTBExchangeConnector {
    BidSwitchExchangeConnector(ServiceBase & owner, const std::string & name);
    BidSwitchExchangeConnector(const std::string & name,
                               std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "bidswitch";
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
                       const std::string & payload) {
        // TODO: check that is at it seems
        return 200.0;
    }

    /** This is the information that the BidSwitch exchange needs to keep
        for each campaign (agent).
    */
    struct CampaignInfo {
        Id seat;          ///< ID of the BidSwitch exchange seat
        std::string iurl; ///< Image URL for content checking
    };

    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;

    /** This is the information that BidSwitch needs in order to properly
        filter and serve a creative.
    */
    struct CreativeInfo {
        Id adid;                ///< ID for ad to be service if bid wins
        std::string nurl;       ///< Win notice URL
        std::vector<std::string> adomain;    ///< Advertiser Domain
        struct {
            std::vector<int32_t> vendor_type_;
            std::set<int32_t> attribute_ ;
        } Google;
    };

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    virtual bool
    bidRequestCreativeFilter(const BidRequest & request,
                             const AgentConfig & config,
                             const void * info) const;


    // BidSwitch win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);

  private:
    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;

};



} // namespace RTBKIT
