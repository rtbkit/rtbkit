/* casale_exchange_connector.cc
   Mathieu Stefani, 05 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   Implementation of the Casale Exchange Connector
*/

#include "casale_exchange_connector.h"
#include "soa/utils/generic_utils.h"

using namespace Datacratic;

namespace RTBKIT {

namespace Default {

    // 2.6 Response Times
    static constexpr double MaximumResponseTime = 100;
}

/*****************************************************************************/
/* CASALE EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

CasaleExchangeConnector::CasaleExchangeConnector(
        ServiceBase& owner, std::string name) 
    : OpenRTBExchangeConnector(owner, std::move(name))
    , creativeConfig("casale")
{
    this->auctionResource = "/bidder";
    this->auctionVerb = "POST";
    initCreativeConfiguration();
}

CasaleExchangeConnector::CasaleExchangeConnector(
        std::string name, std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(std::move(name), std::move(proxies))
    , creativeConfig("casale")
{
    this->auctionResource = "/bidder";
    this->auctionVerb = "POST";
    initCreativeConfiguration();
}

void CasaleExchangeConnector::initCreativeConfiguration()
{
    creativeConfig.addField(
        "adm",
        [](const Json::Value& value, CreativeInfo& info) {
            Datacratic::jsonDecode(value, info.adm);
            if (info.adm.empty()) {
                throw std::invalid_argument("adm is required");
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
}

double
CasaleExchangeConnector::getTimeAvailableMs(
        HttpAuctionHandler &handler,
        const HttpHeader& header,
        const std::string& payload) {

    return Default::MaximumResponseTime;
}

ExchangeConnector::ExchangeCompatibility
CasaleExchangeConnector::getCampaignCompatibility(
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
    if (!seat.isIntegral()) {
        result.setIncompatible(
                ML::format("providerConfig.%s.seat is not merdiumint or unsigned", name),
                includeReasons);
        return result;
    }

    uint64_t value = seat.asUInt();
    if (value > CampaignInfo::MaxSeatValue) {
        result.setIncompatible(
                ML::format("providerConfig.%s.seat > %lld", name, CampaignInfo::MaxSeatValue),
                includeReasons);
        return result;
    }

    auto info = std::make_shared<CampaignInfo>(); 
    info->seat = value;

    result.info = info;
    return result;
}

ExchangeConnector::ExchangeCompatibility
CasaleExchangeConnector::getCreativeCompatibility(
        const Creative& creative,
        bool includeReasons) const
{
    return creativeConfig.handleCreativeCompatibility(creative, includeReasons);
}

std::shared_ptr<BidRequest>
CasaleExchangeConnector::parseBidRequest(
        HttpAuctionHandler& handler,
        const HttpHeader& header,
        const std::string& payload)
{
    /* According to the documentation:
     *
     * "x-openrtb-version Yes Indicates the version of OpenRTB. Will always be 2.0"
     *
     * Our OpenRTB parser only supports openrtb 2.1 or 2.2 and will throw otherwise. Since
     * 2.1 should be backward-compatible, we "fake" the version and patch it to 2.1 so that
     * we do not throw
     */
    HttpHeader patchedHeaders(header);
    auto it = patchedHeaders.headers.find("x-openrtb-version");
    if (it != std::end(patchedHeaders.headers) && it->second == "2.0") {
        it->second = "2.1";
    }

    auto request = OpenRTBExchangeConnector::parseBidRequest(handler, patchedHeaders, payload);

    return request;

}

void
CasaleExchangeConnector::setSeatBid(
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

    // Creative the seat if it does not exist
    if (seatIndex == -1) {
        OpenRTB::SeatBid sbid;
        sbid.seat = Id(campaignInfo->seat);

        response.seatbid.push_back(std::move(sbid));

        seatBid = &response.seatbid.back();
    }
    else {
        seatBid = &response.seatbid[seatIndex];
    }

    ExcAssert(seatBid);
    seatBid->bid.emplace_back();
    auto& bid = seatBid->bid.back();

    CasaleCreativeConfiguration::Context context {
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
    /* Prices are in Cents CPM */
    bid.price.val *= 100;

    if (!creativeInfo->adomain.empty()) bid.adomain = creativeInfo->adomain;
    bid.adm = creativeConfig.expand(creativeInfo->adm, context);

}


} // namespace RTBKIT

namespace {

struct Init {
    Init() {
        RTBKIT::ExchangeConnector::registerFactory<CasaleExchangeConnector>();
    }
};

struct Init init;
}

