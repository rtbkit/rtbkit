/* mopub_exchange_connector.cc
   Jeremy Barnes, 15 March 2013

   Implementation of the MoPub exchange connector.
*/
#include <algorithm>
#include <boost/tokenizer.hpp>

#include "mopub_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "openrtb/openrtb_parsing.h"
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
/* MOPUB EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

MoPubExchangeConnector::
MoPubExchangeConnector(ServiceBase & owner, const std::string & name)
    : OpenRTBExchangeConnector(owner, name) {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

MoPubExchangeConnector::
MoPubExchangeConnector(const std::string & name,
                       std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies) {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
}

ExchangeConnector::ExchangeCompatibility
MoPubExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();

    const Json::Value & pconf = config.providerConfig["mopub"];

    try {
        cpinfo->seat = Id(pconf["seat"].asString());
        if (!cpinfo->seat)
            result.setIncompatible("providerConfig.mopub.seat is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
        (string("providerConfig.mopub.seat parsing error: ")
         + exc.what(), includeReasons);
        return result;
    }

    try {
        cpinfo->iurl = pconf["iurl"].asString();
        if (!cpinfo->iurl.size())
            result.setIncompatible("providerConfig.mopub.iurl is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
        (string("providerConfig.mopub.iurl parsing error: ")
         + exc.what(), includeReasons);
        return result;
    }

    result.info = cpinfo;

    return result;
}

namespace {

using Datacratic::jsonDecode;

/** Given a configuration field, convert it to the appropriate JSON */
template<typename T>
void getAttr(ExchangeConnector::ExchangeCompatibility & result,
             const Json::Value & config,
             const char * fieldName,
             T & field,
             bool includeReasons) {
    try {
        if (!config.isMember(fieldName)) {
            result.setIncompatible
            ("creative[].providerConfig.mopub." + string(fieldName)
             + " must be specified", includeReasons);
            return;
        }

        const Json::Value & val = config[fieldName];

        jsonDecode(val, field);
    } catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.mopub."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}

} // file scope

ExchangeConnector::ExchangeCompatibility
MoPubExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto crinfo = std::make_shared<CreativeInfo>();

    const Json::Value & pconf = creative.providerConfig["mopub"];
    std::string tmp;

    boost::char_separator<char> sep(" ,");

    // 1.  Must have mopub.attr containing creative attributes.  These
    //     turn into MoPubCreativeAttribute filters.
    getAttr(result, pconf, "attr", tmp, includeReasons);
    if (!tmp.empty())  {
        boost::tokenizer<boost::char_separator<char>> tok(tmp, sep);
        auto& ints = crinfo->attr;
        std::transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }
    tmp.clear();


    // 2. Must have mopub.type containing attribute type.
    getAttr(result, pconf, "type", tmp, includeReasons);
    if (!tmp.empty())  {
        boost::tokenizer<boost::char_separator<char>> tok(tmp, sep);
        auto& ints = crinfo->type;
        std::transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }
    tmp.clear();

    // 3. Must have mopub.cat containing attribute type.
    getAttr(result, pconf, "cat", tmp, includeReasons);
    if (!tmp.empty())  {
        boost::tokenizer<boost::char_separator<char>> tok(tmp, sep);
        auto& strs = crinfo->cat;
        copy(tok.begin(),tok.end(),inserter(strs,strs.begin()));
    }
    tmp.clear();

    // 4.  Must have mopub.adm that includes MoPub's macro
    getAttr(result, pconf, "adm", crinfo->adm, includeReasons);
    if (crinfo->adm.find("${AUCTION_PRICE:BF}") == string::npos)
        result.setIncompatible
        ("creative[].providerConfig.mopub.adm ad markup must contain "
         "encrypted win price macro ${AUCTION_PRICE:BF}",
         includeReasons);

    // 5.  Must have creative ID in mopub.crid
    getAttr(result, pconf, "crid", crinfo->crid, includeReasons);
    if (!crinfo->crid)
        result.setIncompatible
        ("creative[].providerConfig.mopub.crid is null",
         includeReasons);

    // 6.  Must have advertiser names array in mopub.adomain
    getAttr(result, pconf, "adomain", crinfo->adomain,  includeReasons);
    if (crinfo->adomain.empty())
        result.setIncompatible
        ("creative[].providerConfig.mopub.adomain is empty",
         includeReasons);

    // Cache the information
    result.info = crinfo;

    return result;
}
std::shared_ptr<BidRequest>
MoPubExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload) {
    std::shared_ptr<BidRequest> res;

    // Check for JSON content-type
    if (header.contentType != "application/json") {
        connection.sendErrorResponse("non-JSON request");
        return res;
    }

    // Parse the bid request
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRtbBidRequestParser::parseBidRequest(context, exchangeName(), exchangeName()));

    // get restrictions enforced by MoPub.
    //1) blocked category
    std::vector<std::string> strv;
    for (const auto& cat: res->blockedCategories)
        strv.push_back(cat.val);
    res->restrictions.addStrings("blockedCategories", strv);

    //2) per slot: blocked type and attribute;
    std::vector<int> intv;
    for (auto& spot: res->imp) {
        for (const auto& t: spot.banner->btype) {
            intv.push_back (t.val);
        }
        spot.restrictions.addInts("blockedTypes", intv);
        intv.clear();
        for (const auto& a: spot.banner->battr) {
            intv.push_back (a.val);
        }
        spot.restrictions.addInts("blockedAttrs", intv);
    }
    return res;
}


void
MoPubExchangeConnector::
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

    // Put in the variable parts
    b.cid = Id(resp.agent);
    b.id = Id(auction.id, auction.request->imp[0].id);
    b.impid = auction.request->imp[spotNum].id;
    b.price.val = USD_CPM(resp.price.maxPrice);
    b.adm = crinfo->adm;
    b.adomain = crinfo->adomain;
    b.crid = crinfo->crid;
    b.iurl = cpinfo->iurl;
}

template <typename T>
bool disjoint (const set<T>& s1, const set<T>& s2) {
    auto i = s1.begin(), j = s2.begin();
    while (i != s1.end() && j != s2.end()) {
        if (*i == *j)
            return false;
        else if (*i < *j)
            ++i;
        else
            ++j;
    }
    return true;
}
bool
MoPubExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const {
    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    // 1) we first check for blocked content categories
    const auto& blocked_categories = request.restrictions.get("blockedCategories");
    for (const auto& cat: crinfo->cat)
        if (blocked_categories.contains(cat)) {
            this->recordHit ("blockedCategory");
            return false;
        }

    // 2) now go throught the spots.
    for (const auto& spot: request.imp) {
        const auto& blocked_types = spot.restrictions.get("blockedTypes");
        for (const auto& t: crinfo->type)
            if (blocked_types.contains(t)) {
                this->recordHit ("blockedType");
                return false;
            }
        const auto& blocked_attr = spot.restrictions.get("blockedAttrs");
        for (const auto& a: crinfo->attr)
            if (blocked_attr.contains(a)) {
                this->recordHit ("blockedAttr");
                return false;
            }
    }

    return true;
}

} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct Init {
    Init() {
        ExchangeConnector::registerFactory<MoPubExchangeConnector>();
    }
} init;
}

