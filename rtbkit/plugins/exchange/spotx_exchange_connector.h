/* spotx_exchange_connector.h
   Mathieu Stefani, 20 May 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   The SpotX Exchange Connector
*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/common/creative_configuration.h"
#include "soa/service/logs.h"
#include "soa/types/id.h"

namespace RTBKIT {

/*****************************************************************************/
/* SPOTX EXCHANGE CONNECTOR                                                  */
/*****************************************************************************/

struct SpotXExchangeConnector : public OpenRTBExchangeConnector {

    SpotXExchangeConnector(ServiceBase& owner, std::string name);
    SpotXExchangeConnector(std::string name, std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "spotx";
    }

    std::string exchangeName() const {
        return exchangeNameString();
    }

    struct CampaignInfo {
        ///< ID of the bidder seat on whose behalf the bid is made
        Datacratic::Id seat;

        ///< Name of the bidder seat on whose behalf the bid is made
        std::string seatName;

        ///< Bid response ID to assist tracking for bidders
        std::string bidid;
    };

    ExchangeCompatibility
    getCampaignCompatibility(
        const AgentConfig& config,
        bool includeReasons) const;

    ExchangeCompatibility
    getCreativeCompatibility(
        const Creative& creative,
        bool includeReasons) const;

    struct CreativeInfo {
        std::string adm;
        std::vector<std::string> adomain;
        std::string adid;
    };

    static Logging::Category print;
    static Logging::Category warning;

    typedef CreativeConfiguration<CreativeInfo> SpotXCreativeConfiguration;

private:
    void initCreativeConfiguration();

    void setSeatBid(const Auction& auction,
                    int spotNum,
                    OpenRTB::BidResponse& response) const;

    Json::Value getSeatBidExtension(const CampaignInfo* info) const;

    SpotXCreativeConfiguration creativeConfig;

};

} // namespace RTBKIT
