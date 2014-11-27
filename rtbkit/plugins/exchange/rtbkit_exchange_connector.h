/* rtbkit_exchange_connector.h                                    -*- C++ -*-
   Mathieu Stefani, 15 May 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.
*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/core/router/filters/generic_creative_filters.h"

namespace RTBKIT {

struct ExternalIdsCreativeExchangeFilter
    : public IterativeCreativeFilter<ExternalIdsCreativeExchangeFilter>
{
    static constexpr const char *name = "ExternalIdsCreativeExchangeFilter";

    bool filterCreative(FilterState &state, const AdSpot &spot,
                        const AgentConfig &config, const Creative &creative) const
    {
        // We're doing this check at the exchange connector level
#if 0
        if (!spot.ext.isMember("external-ids")) {
            return true;
        }
#endif
        const auto &external_ids = spot.ext["external-ids"];
        for (auto it = external_ids.begin(), end = external_ids.end(); it != end; ++it) {
            int id = std::stoi(it.key().asString());
            if (id == config.externalId) return true;
        }

        return false;
    }

};

/*****************************************************************************/
/* RTBKIT EXCHANGE CONNECTOR                                                 */
/*****************************************************************************/

/**
   The RTBKitExchangeConnector is used when connecting two RTBKit stacks. It must
   be used together with the HttpBidderInterface

   This Exchange Connector is based on the OpenRTB Exchange Connector and uses
   the same parsing logic
*/

struct RTBKitExchangeConnector : public OpenRTBExchangeConnector {
    RTBKitExchangeConnector(ServiceBase &owner, const std::string &name);
    RTBKitExchangeConnector(const std::string &name,
                            std::shared_ptr<ServiceProxies> proxies);

    static std::string exchangeNameString() {
        return "rtbkit";
    }

    virtual std::string exchangeName() const {
        return exchangeNameString();
    }

    virtual std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler &connection,
                    const HttpHeader &header,
                    const std::string &payload);

    virtual void
    adjustAuction(std::shared_ptr<Auction>& auction) const;
protected:

    virtual void
    setSeatBid(const Auction &auction,
               int spotNum,
               OpenRTB::BidResponse &response) const;
};

} // namespace RTBKIT
