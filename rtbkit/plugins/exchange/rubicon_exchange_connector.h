/* rubicon_exchange_connector.h                                    -*- C++ -*-
   Jeremy Barnes, 12 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/common/creative_configuration.h"

namespace RTBKIT {


/*****************************************************************************/
/* RUBICON EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

/** Exchange connector for Rubicon.  This speaks their flavour of the
    OpenRTB 2.1 protocol.
*/

struct RubiconExchangeConnector: public OpenRTBExchangeConnector {
    RubiconExchangeConnector(ServiceBase & owner, const std::string & name);
    RubiconExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "rubicon";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

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
        std::string adm;                                ///< Ad markup
        std::vector<std::string> adomain;               ///< Advertiser domains
        Id cid;                                         ///< Optional Campaign ID
        Id crid;                                        ///< Creative ID
        OpenRTB::List<OpenRTB::CreativeAttribute> attr; ///< Creative attributes
        std::string ext_creativeapi;                    ///< Creative API
    };

    typedef CreativeConfiguration<CreativeInfo> RubiconCreativeConfiguration;

    void init();

    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    // Rubicon win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);

private:
    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;

    RubiconCreativeConfiguration configuration_;
};



} // namespace RTBKIT
