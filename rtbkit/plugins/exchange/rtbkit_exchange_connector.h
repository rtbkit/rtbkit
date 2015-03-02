/* rtbkit_exchange_connector.h                                    -*- C++ -*-
   Mathieu Stefani, 15 May 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.
*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/core/router/filters/generic_creative_filters.h"

namespace RTBKIT {

struct CreativeIdsExchangeFilter
    : public IterativeCreativeFilter<CreativeIdsExchangeFilter>
{
    static constexpr const char *name = "CreativeIdsExchangeFilter";

    bool filterCreative(FilterState &state, const AdSpot &spot,
                        const AgentConfig &config, const Creative &creative) const
    {

        auto doFilter = [&](const Json::Value& value) -> bool {
            using std::find_if;  using std::begin;  using std::end;
            using std::stoi;
            if (value.isArray()) {
                return find_if(begin(value), end(value), [&](const Json::Value &val) {
                     return val.isIntegral() && val.asInt() == config.externalId;
                }) != end(value);
            }
            else if (value.isObject()) {
                for (auto it = value.begin(), end = value.end(); it != end; ++it) {
                    const auto& key = it.key();
                    try {
                        const int id = stoi(key.asString());
                        if (id == config.externalId) {
                            const auto& crids = *it;
                            return find_if(crids.begin(), crids.end(), [&](const Json::Value& value) {
                                return value.isIntegral() && value.asInt() == creative.id;
                            }) != crids.end();
                        }
                    } catch (const std::invalid_argument&) {
                        return false;
                    }
                }
            }
            else {
                ExcCheck(false, "Invalid code path");
            }
            return false;
        };

        if (spot.ext.isMember("creative-ids")) {
            return doFilter(spot.ext["creative-ids"]);
        } else if (spot.ext.isMember("external-ids")) {
            return doFilter(spot.ext["external-ids"]);
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
