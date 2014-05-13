/* agents_bidder_interface.h
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/bidder_interface.h"
#include "rtbkit/core/router/filters/generic_creative_filters.h"
#include "soa/jsoncpp/json.h"
#include <iostream>

namespace RTBKIT {

struct AllowedIdsCreativeExchangeFilter
    : public IterativeCreativeFilter<AllowedIdsCreativeExchangeFilter>
{
    static constexpr const char *name = "AllowedIdsCreativeExchangeFilter";

    bool filterCreative(FilterState &state, const AdSpot &,
                        const AgentConfig &config, const Creative &creative) const
    {
        const auto &request = state.request;
        ExcCheck(request.imp.size() > 0, "Request has empty impression");
        if (!request.imp[0].ext.isMember("allowed_ids")) {
            return true;
        }

        for (const auto &imp: request.imp) {
            const auto &allowed_ids = imp.ext["allowed_ids"];
            for (auto it = allowed_ids.begin(); it != allowed_ids.end(); ++it) {
                size_t pos;
                int id = std::stoi(it.key().asString(), &pos);
                if (pos != 0) {
                }
                if (id == config.externalId) {
                    const auto &creative_ids = *it;
                    auto crIt = std::find_if(
                        std::begin(creative_ids), std::end(creative_ids),
                        [&](const Json::Value &value) {
                            return value.asInt() == creative.id;
                        });
                    if (crIt != std::end(creative_ids)) {
                        return true;
                    }
                }
            }
        }

        return false;

    }

};

struct AgentsBidderInterface : public BidderInterface
{
    AgentsBidderInterface(std::string const & name = "bidder",
                          std::shared_ptr<ServiceProxies> proxies = std::make_shared<ServiceProxies>(),
                          Json::Value const & config = Json::Value());

    void sendAuctionMessage(std::shared_ptr<Auction> const & auction,
                            double timeLeftMs,
                            std::map<std::string, BidInfo> const & bidders);

    void sendWinLossMessage(MatchedWinLoss const & event);

    void sendLossMessage(std::string const & agent,
                         std::string const & id);

    void sendCampaignEventMessage(std::string const & agent,
                                  MatchedCampaignEvent const & event);

    void sendBidLostMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendBidDroppedMessage(std::string const & agent,
                               std::shared_ptr<Auction> const & auction);

    void sendBidInvalidMessage(std::string const & agent,
                               std::string const & reason,
                               std::shared_ptr<Auction> const & auction);

    void sendNoBudgetMessage(std::string const & agent,
                             std::shared_ptr<Auction> const & auction);

    void sendTooLateMessage(std::string const & agent,
                            std::shared_ptr<Auction> const & auction);

    void sendMessage(std::string const & agent,
                     std::string const & message);

    void sendErrorMessage(std::string const & agent,
                          std::string const & error,
                          std::vector<std::string> const & payload);

    void sendPingMessage(std::string const & agent,
                         int ping);

    void send(std::shared_ptr<PostAuctionEvent> const & event);
};

}

