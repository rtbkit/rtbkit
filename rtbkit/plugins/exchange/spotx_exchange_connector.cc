/* spotx_exchange_connector.cc
   Mathieu Stefani, 20 May 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Implementation of the SpotX Exchange Connector
*/

#include "spotx_exchange_connector.h"
#include "soa/utils/generic_utils.h"

using namespace Datacratic;

namespace RTBKIT {

Logging::Category SpotXExchangeConnector::print("SpotXExchangeConnector");
Logging::Category SpotXExchangeConnector::warning("SpotXExchangeConnector Warning",
                     SpotXExchangeConnector::print);

/*****************************************************************************/
/* SPOTX EXCHANGE CONNECTOR                                                  */
/*****************************************************************************/

SpotXExchangeConnector::SpotXExchangeConnector(
        ServiceBase& owner, std::string name)
    : OpenRTBExchangeConnector(owner, std::move(name))
    , creativeConfig("spotx")
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    initCreativeConfiguration();
}

SpotXExchangeConnector::SpotXExchangeConnector(
        std::string name, std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(std::move(name), std::move(proxies))
    , creativeConfig("spotx")
{
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    initCreativeConfiguration();
}

void
SpotXExchangeConnector::initCreativeConfiguration()
{
    creativeConfig.addField(
        "adm",
        [](const Json::Value& value, CreativeInfo& info) {
            Datacratic::jsonDecode(value, info.adm);
            if (info.adm.empty()) {
                throw std::invalid_argument("adm is required");
            }

            if (info.adm.find("$MBR") == std::string::npos) {
                LOG(warning) << "The adm does not contain the $MBR macro, SpotX might flag the response as invalid" << std::endl;
            }

            return true;
    }).snippet();

    creativeConfig.addField(
        "adomain",
        [](const Json::Value& value, CreativeInfo& info) {
            Datacratic::jsonDecode(value, info.adomain);
            if (info.adomain.empty()) {
                throw std::invalid_argument("adomain is required");
            }

            return true;
    }).required();

    creativeConfig.addField(
        "adid",
        [](const Json::Value& value, CreativeInfo& info) {
            Datacratic::jsonDecode(value, info.adid);
            if (info.adid.empty()) {
                throw std::invalid_argument("adid is required");
            }

            return true;
    }).required();

}

ExchangeConnector::ExchangeCompatibility
SpotXExchangeConnector::getCampaignCompatibility(
        const AgentConfig& config,
        bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    std::string exchange = exchangeName();
    const char* name = exchange.c_str();
    if (!config.providerConfig.isMember(exchange)) {
        result.setIncompatible(
                ML::format("providerConfig.%s is null", name), includeReasons);
        return result;
    }

    const auto& provConf = config.providerConfig[exchange];
    if (!provConf.isMember("seat")) {
        result.setIncompatible(
               ML::format("providerConfig.%s.seat is null", name), includeReasons);
        return result;
    }

    const auto& seat = provConf["seat"];
    if (!seat.isString()) {
        result.setIncompatible(
            ML::format("providerConfig.%s.seat must be a string", name), includeReasons);
        return result;
    }

    if (!provConf.isMember("bidid")) {
        result.setIncompatible(
            ML::format("providerConfig.%s.bidid is null", name), includeReasons);
        return result;
    }

    const auto& bidid = provConf["bidid"];
    if (!bidid.isString()) {
        result.setIncompatible(
                ML::format("providerConfig.%s.bidid must be a string", name), includeReasons);
        return result;
    }

    std::string seatName;
    if (provConf.isMember("seatName")) {
        const auto value = provConf["seatName"];
        if (!value.isString()) {
            result.setIncompatible(
                ML::format("providerConfig.%s.seatName must be a string", name), includeReasons);
        }
        seatName = value.asString();
    }


    auto info = std::make_shared<CampaignInfo>(); 
    auto value = seat.asString();
    auto bididValue = bidid.asString();
    info->seat = Id(value);
    info->seatName = std::move(seatName);
    info->bidid = std::move(bididValue);

    result.info = info;
    return result;
}

ExchangeConnector::ExchangeCompatibility
SpotXExchangeConnector::getCreativeCompatibility(
        const Creative& creative,
        bool includeReasons) const
{
    if (creative.isImage()) {
        const auto& format = creative.format;
        if (format.width != 300 || format.height != 250) {
            ExchangeCompatibility result;
            result.setIncompatible("SpotXchange only supports 300x250", includeReasons);
            return result;
        }
    }

    return creativeConfig.handleCreativeCompatibility(creative, includeReasons);
}

void
SpotXExchangeConnector::setSeatBid(
        const Auction& auction,
        int spotNum,
        OpenRTB::BidResponse& response) const {

    const Auction::Data *current = auction.getCurrentData();

    auto& resp = current->winningResponse(spotNum);

    const AgentConfig* config
        = std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();
    std::string name = exchangeName();

    auto campaignInfo = config->getProviderData<CampaignInfo>(name);
    int creativeIndex = resp.agentCreativeIndex;

    auto& creative = config->creatives[creativeIndex];
    auto creativeInfo = creative.getProviderData<CreativeInfo>(name);

    // Find the index in the seats array
    int seatIndex = indexOf(response.seatbid, &OpenRTB::SeatBid::seat, Id(campaignInfo->seat));

    OpenRTB::SeatBid* seatBid;

    // Create the seat if it does not exist
    if (seatIndex == -1) {
        OpenRTB::SeatBid sbid;
        sbid.seat = Id(campaignInfo->seat);
        sbid.ext = getSeatBidExtension(campaignInfo);

        response.seatbid.push_back(std::move(sbid));

        seatBid = &response.seatbid.back();
    }
    else {
        seatBid = &response.seatbid[seatIndex];
    }

    ExcAssert(seatBid);
    seatBid->bid.emplace_back();
    auto& bid = seatBid->bid.back();

    SpotXCreativeConfiguration::Context context {
        creative,
        resp,
        *auction.request,
        spotNum
    };

    bid.cid = Id(resp.agent);
    bid.crid = Id(resp.creativeId);
    bid.impid = auction.request->imp[spotNum].id;
    bid.id = Id(auction.id, auction.request->imp[0].id);
    bid.price.val = USD_CPM(resp.price.maxPrice);

    bid.adomain = creativeInfo->adomain;
    bid.adid = Id(creativeInfo->adid);
    bid.adm = creativeConfig.expand(creativeInfo->adm, context);

    response.bidid = Id(campaignInfo->bidid);
    response.cur = "USD";
}

Json::Value
SpotXExchangeConnector::getSeatBidExtension(const CampaignInfo* info) const
{
    if (JML_LIKELY(info->seatName.empty())) {
        return { };
    }

    Json::Value value;
    value["seatname"] = info->seatName;
    return value;
}

} // namespace RTBKIT

namespace {

struct Init {
    Init() {
        RTBKIT::ExchangeConnector::registerFactory<SpotXExchangeConnector>();
    }
};

struct Init init;
}

