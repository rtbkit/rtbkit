/* rtbkit_exchange_connector.h                                    -*- C++ -*-
   Mathieu Stefani, 15 May 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.
*/

#pragma once

#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/core/router/filters/generic_creative_filters.h"

namespace RTBKIT {

namespace {

    /* Function to compare a plain int (signed or unsigned) to a string

       We use this function to avoid allocating too much in the filter below
       when we compare ids that are in the json

       The function will throw if the @str parameter is not a valid number
       Note that this function is not safe at all as it does not do any
       overflow checking
    */

    bool string_int_equals(const char* str, int value) {
        if (!str) return false;
        if (str[0] == 0) return false;

        /* Fast path, kinda */

        if (value < 0 && str[0] != '-') return false;
        if (value > 0 && str[0] == '-') return false;

        /* Slow path */

        /* Compute the length and check if there is any non-digit character
           at the same time
        */
        const size_t len = [&]() {
            size_t index = 0;

            int c;
            while ((c = str[index]) != 0) {
                if (index == 0 && c != '-' && !isdigit(c))
                    throw std::invalid_argument("Not a valid number");

                ++index;
            }

            return index;
        }();

        if (value < 0)
            value = -value;

        size_t i = len - 1;

        auto to_int = [](char c) {
            return c - '0';
        };

        do {
            const char digitChar = str[i];

            const int currentDigit = value % 10;
            if (to_int(digitChar) != currentDigit) return false;

            value /= 10;
            if (i == 0) break;
            else --i;
        } while (value > 0);

        return value == 0 && i == 0;
    }
}


struct CreativeIdsExchangeFilter
    : public IterativeCreativeFilter<CreativeIdsExchangeFilter>
{
    static constexpr const char *name = "CreativeIdsExchangeFilter";

    bool filterCreative(FilterState &state, const AdSpot &spot,
                        const AgentConfig &config, const Creative &creative) const
    {
        if (spot.ext.isMember("creative-ids")) {
            return filterCreativeIds(config, creative, spot.ext["creative-ids"]);
        } else if (spot.ext.isMember("external-ids")) {
            return filterExternalIds(config, spot.ext["external-ids"]);
        }

        return false;
    }

private:
    bool filterExternalIds(
            const AgentConfig& config, const Json::Value& extIds) const
    {
        if (JML_UNLIKELY(extIds.isNull())) return false;

        ExcAssert(extIds.isArray());

        using std::find_if;
        using std::begin;   using std::end;

        return find_if(
            begin(extIds), end(extIds),
            [&](const Json::Value &extId) {
                 return extId.isIntegral() && extId.asInt() == config.externalId;
            }
        ) != end(extIds);

    }

    bool filterCreativeIds(
            const AgentConfig& config, const Creative& creative,
            const Json::Value& crIds) const
    {
        if (JML_UNLIKELY(crIds.isNull())) return false;

        ExcAssert(crIds.isObject());

        using std::find_if;
        using std::begin;   using std::end;

        for (auto it = crIds.begin(), end = crIds.end(); it != end; ++it) {
            auto key = it.memberNameC();
            try {
                if (string_int_equals(key, config.externalId)) {
                    const auto& crids = *it;
                    return find_if(crids.begin(), crids.end(), [&](const Json::Value& value) {
                        return value.isIntegral() && value.asInt() == creative.id;
                    }) != crids.end();
                }
            } catch (const std::invalid_argument&) {
                return false;
            }
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
