/* nexage_exchange_connector.cc
   Jan S.
   (C) 2014 Datacratic Inc

   Implementation of the Nexage exchange connector.

   OpenRTB is fully supported with respect to the win notice and ad
   serving mechanisms. In addition, a default win notice can be
   configured on the Nexage RTB Exchange. This default win notice will
   be used if it is not passed via the “nurl” attribute.

   All substitution macros are supported both in ad markup and in win
   notices (whether passed in the bid response or configured as an
   Exchange default). No macro encoding options are currently
   supported.

   Bidders can provide ad markup in their bid response via the “adm”
   attribute in case they win or on the win notice response when they
   win. Only one method should be used in a given auction, but if
   non-empty strings are found in both, the Nexage RTB Exchange will
   use the bid response “adm” attribute and ignore the win notice
   return body.

   On a given auction if the winning bidder fails to return ad markup
   via one of these two methods, then the win will be forfeited and
   the Exchange will select a new winner.
*/

#include "nexage_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

#include "crypto++/blowfish.h"
#include "crypto++/modes.h"
#include "crypto++/filters.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/*****************************************************************************/
/* NEXAGE EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

NexageExchangeConnector::
NexageExchangeConnector(ServiceBase & owner, const std::string & name)
    : OpenRTBExchangeConnector(owner, name),
      configuration_("nexage"){
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    init();
}

NexageExchangeConnector::
NexageExchangeConnector(const std::string & name,
                        std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies),
      configuration_("nexage"){
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    init();
}

ExchangeConnector::ExchangeCompatibility
NexageExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();

    const Json::Value & pconf = config.providerConfig["nexage"];

    try {
        cpinfo->seat = Id(pconf["seat"].asString());
        if (!cpinfo->seat)
            result.setIncompatible("providerConfig.nexage.seat is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
        (string("providerConfig.nexage.seat parsing error: ")
         + exc.what(), includeReasons);
        return result;
    }

    result.info = cpinfo;

    return result;
}

void
NexageExchangeConnector::init()
{

    // Mandatory Attributes
    // Must have adm that includes Nexage macro (at least the price)
    configuration_.addField(
        "adm",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.adm);
            if (data.adm.empty()) {
                throw std::invalid_argument("adm can not be empty");
            }
            return true;
    }).snippet();

    // Must have advertiser names array in mopub.adomain
    configuration_.addField(
        "adomain",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.adomain);
            if (data.adomain.empty()){
                throw std::invalid_argument("adomain can not be empty");
            }
            return true;
    });

    // Must have iurl that represents the creative.
    configuration_.addField(
        "iurl",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.iurl);
            if (data.iurl.empty()){
                throw std::invalid_argument("iurl can not be empty");
            }
            return true;
    });

    // Must have creative ID in mopub.crid
    configuration_.addField(
        "crid",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.crid);
            if (data.crid.toString().empty()){
                throw std::invalid_argument("crid can not be empty");
            }
            return true;
    });


    // Optional Attributes

    configuration_.addField(
        "cat",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.cat);
            return true;
    }).optional();

    configuration_.addField(
        "attr",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.attr);
            return true;
    }).optional();

    configuration_.addField(
        "nurl",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.nurl);
            return true;
    }).optional();

}


ExchangeConnector::ExchangeCompatibility
NexageExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const {
    return configuration_.handleCreativeCompatibility(creative, includeReasons);
}

std::shared_ptr<BidRequest>
NexageExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload) {
    std::shared_ptr<BidRequest> res;
//
    // Check for JSON content-type
    if (header.contentType != "application/json") {
        connection.sendErrorResponse("non-JSON request");
        return res;
    }

    // Parse the bid request
    // Nexage used not to send x-openrtb-version but they're now at 2.2
    // source : http://www.nexage.com/resource-center/openrtb-2-2-technical-reference/
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.2")->parseBidRequest(context, exchangeName(), exchangeName()));

    return res;
}


void
NexageExchangeConnector::
setSeatBid(Auction const & auction,
           int spotNum,
           OpenRTB::BidResponse & response) const {

    const Auction::Data * current = auction.getCurrentData();

    // Get the winning bid
    auto & resp = current->winningResponse(spotNum);

    // Find how the agent is configured.  We need to copy some of the
    // fields into the bid.
    const AgentConfig * config =
        std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();

    std::string en = exchangeName();

    // Get the exchange specific data for this campaign
    auto cpinfo = config->getProviderData<CampaignInfo>(en);

    // Put in the fixed parts from the creative
    int creativeIndex = resp.agentCreativeIndex;

    auto & creative = config->creatives.at(creativeIndex);

    // Get the exchange specific data for this creative
    auto crinfo = creative.getProviderData<CreativeInfo>(en);

    // Find the index in the seats array
    int seatIndex = 0;
    while(response.seatbid.size() != seatIndex) {
        if(response.seatbid[seatIndex].seat == cpinfo->seat) break;
        ++seatIndex;
    }

    // Create if required
    if(seatIndex == response.seatbid.size()) {
        response.seatbid.emplace_back();
        response.seatbid.back().seat = cpinfo->seat;
    }

    // Get the seatBid object
    OpenRTB::SeatBid & seatBid = response.seatbid.at(seatIndex);

    // Add a new bid to the array
    seatBid.bid.emplace_back();
    auto & b = seatBid.bid.back();

    NexageCreativeConfiguration::Context ctx = {
        creative,
        resp,
        *auction.request,
        spotNum
    };

    // Put in the variable parts
    b.cid = Id(resp.agent);
    b.iurl = crinfo->iurl;
    b.impid = auction.request->imp[spotNum].id;
    b.id = Id(auction.id, auction.request->imp[0].id);
    b.price.val = USD_CPM(resp.price.maxPrice);
    b.crid = crinfo->crid;
    b.adomain = crinfo->adomain;
    // optional parts
    if (!crinfo->nurl.empty()) b.nurl = crinfo->nurl;
    if (!crinfo->adm.empty()) b.adm = configuration_.expand(crinfo->adm, ctx);
}

template<typename T>
bool contains(const OpenRTB::List<T> & list, const T & value){
    return std::find(list.cbegin(), list.cend(), value) != list.cend();
}

bool contains(const vector<Utf8String> & list, const Utf8String & value){
    return std::find(list.cbegin(), list.cend(), value) != list.cend();
}


bool NexageExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                                      const AgentConfig & config,
                                      const void * info) const{

    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    for (const auto& cat: crinfo->cat){
        if (contains(request.blockedCategories, cat)) {
            this->recordHit ("blockedCategory");
            return false;
        }
    }

    // 2) now go throught the spots, to check for blocked attrs
    for (const auto& spot : request.imp) {
        for (const auto& battr : spot.banner->battr) {
            if (contains(crinfo->attr, battr)) {
                this->recordHit("blockedAttr");
                return false;
            }
        }
    }

    // Check for blockeds adomains
    for ( const auto& adomain : crinfo->adomain) {
        Utf8String utf_adomain(adomain, false);
        if (contains(request.badv, utf_adomain)) {
            this->recordHit("blockedAdomain");
            return false;
        }
    }

    return true;
}

} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct AtInit {
    AtInit() {
        ExchangeConnector::registerFactory<NexageExchangeConnector>();
    }
} atInit;
}

