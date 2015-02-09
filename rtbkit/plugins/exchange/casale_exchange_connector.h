/* casale_exchange_connector.h
   Mathieu Stefani, 05 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   Exchange Connector for Casale Media
*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/common/creative_configuration.h"

namespace RTBKIT {

/*****************************************************************************/
/* CASALE EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

struct CasaleExchangeConnector : public OpenRTBExchangeConnector {
    CasaleExchangeConnector(ServiceBase& owner, std::string name);
    CasaleExchangeConnector(std::string name, std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "casale";
    }

    std::string exchangeName() const {
        return exchangeNameString();
    }

    struct CampaignInfo {
        static constexpr uint64_t MaxSeatValue = 16777215;
        ///< ID of the Casale exchange seat if DSP is used by multiple agencies
        uint64_t seat; // [0, 16777215]
    };
                               
    ExchangeCompatibility
    getCampaignCompatibility(
            const AgentConfig& config,
            bool includeReasons) const;

    ExchangeCompatibility
    getCreativeCompatibility(
            const Creative& creative,
            bool includeReasons) const;

    std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler& handler,
                    const HttpHeader& header,
                    const std::string& payload);

    double getTimeAvailableMs(HttpAuctionHandler& handler,
                              const HttpHeader& header,
                              const std::string& payload);

    struct CreativeInfo {
        std::string adm;
        std::vector<std::string> adomain;
    };

    typedef CreativeConfiguration<CreativeInfo> CasaleCreativeConfiguration;

private:
    void initCreativeConfiguration();

    void setSeatBid(const Auction& auction,
                    int spotNum,
                    OpenRTB::BidResponse& response) const;

    CasaleCreativeConfiguration creativeConfig;
};

} // namespace RTBKIT
