/* rubicon_exchange_connector.h                                    -*- C++ -*-
   Jeremy Barnes, 12 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "soa/service/logs.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/plugins/exchange/creative_configuration.h"

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
        std::string nurl;                               ///< Nurl the VAST url
        std::vector<std::string> adomain;               ///< Advertiser domains
        Id cid;                                         ///< Optional Campaign ID
        Id crid;                                        ///< Creative ID
        OpenRTB::List<OpenRTB::CreativeAttribute> attr; ///< Creative attributes
        std::string ext_creativeapi;                    ///< Creative API
    };


    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;

    // Rubicon win price decoding function.
    static float decodeWinPrice(const std::string & sharedSecret,
                                const std::string & winPriceStr);

private:
    void init();

    typedef TypedCreativeConfiguration<CreativeInfo> RubiconCreativeConfiguration;
    RubiconCreativeConfiguration configuration_;
    
    virtual void setSeatBid(Auction const & auction,
                            int spotNum,
                            OpenRTB::BidResponse & response) const;

    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;
};



} // namespace RTBKIT
