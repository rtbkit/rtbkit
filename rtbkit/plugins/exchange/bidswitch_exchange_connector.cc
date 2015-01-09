/* bidswitch_exchange_connector.cc
   Jeremy Barnes, 15 March 2013

   Implementation of the BidSwitch exchange connector.
*/

#include <iterator> // std::begin

#include "bidswitch_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include "soa/service/logs.h"
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include "jml/utils/file_functions.h"

#include "crypto++/blowfish.h"
#include "crypto++/modes.h"
#include "crypto++/filters.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

Logging::Category bidswitchExchangeConnectorTrace("Bidswitch Exchange Connector");
Logging::Category bidswitchExchangeConnectorError("[ERROR] Bidswitch Exchange Connector", bidswitchExchangeConnectorTrace);

/*****************************************************************************/
/* BIDSWITCH EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

BidSwitchExchangeConnector::
BidSwitchExchangeConnector(ServiceBase & owner, const std::string & name)
    : OpenRTBExchangeConnector(owner, name),
    configuration_("bidswitch") {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    init();
}

BidSwitchExchangeConnector::
BidSwitchExchangeConnector(const std::string & name,
                           std::shared_ptr<ServiceProxies> proxies)
    : OpenRTBExchangeConnector(name, proxies),
      configuration_("bidswitch") {
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
    init();
}

void
BidSwitchExchangeConnector::init() {

    // nurl might contain macros
    configuration_.addField(
        "nurl",
        [](const Json::Value & value, CreativeInfo & data) {
            Datacratic::jsonDecode(value, data.nurl);
            return true;
        }).optional().snippet();
}

namespace {
std::vector<int> stringsToInts(const Json::Value& value) {
    const std::string & data = value.asString();
    std::vector<std::string> strings;
    std::vector<int> ints;
    boost::split(strings, data, boost::is_space(), boost::token_compress_on);
    for (const std::string& e : strings) {
        if (e.empty()) {
            continue;
        }
        ints.push_back(std::stoi(e));
    }

    return ints;
}
}

ExchangeConnector::ExchangeCompatibility
BidSwitchExchangeConnector::
getCampaignCompatibility(const AgentConfig & config,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto cpinfo = std::make_shared<CampaignInfo>();

    const Json::Value & pconf = config.providerConfig["bidswitch"];

    try {
        cpinfo->iurl = pconf["iurl"].asString();
        if (!cpinfo->iurl.size())
            result.setIncompatible("providerConfig.bidswitch.iurl is null",
                                   includeReasons);
    } catch (const std::exception & exc) {
        result.setIncompatible
        (string("providerConfig.bidswitch.iurl parsing error: ")
         + exc.what(), includeReasons);
        return result;
    }

    try {
        cpinfo->seat = Id(pconf["seat"].asString());
    } catch (const std::exception & exc) {
        result.setIncompatible
        (string("providerConfig.bidswitch.seat parsing error: ")
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
            ("creative[].providerConfig.bidswitch." + string(fieldName)
             + " must be specified", includeReasons);
            return;
        }

        const Json::Value & val = config[fieldName];

        jsonDecode(val, field);
    } catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.bidswitch."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}

} // file scope

ExchangeConnector::ExchangeCompatibility
BidSwitchExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const {
    ExchangeCompatibility result;
    result.setCompatible();

    auto crinfo = std::make_shared<CreativeInfo>();

    const Json::Value & pconf = creative.providerConfig["bidswitch"];

    // 1.  Must have bidswitch.nurl that includes BidSwitch's macro
    getAttr(result, pconf, "nurl", crinfo->nurl, includeReasons);
    if (crinfo->nurl.find("${AUCTION_PRICE}") == string::npos)
        result.setIncompatible
        ("creative[].providerConfig.bidswitch.nurl ad markup must contain "
         "encrypted win price macro ${AUCTION_PRICE}",
         includeReasons);

    // 2.  Must have creative ID in bidswitch.crid
    getAttr(result, pconf, "adid", crinfo->adid, includeReasons);
    if (!crinfo->adid)
        result.setIncompatible
        ("creative[].providerConfig.bidswitch.adid is null",
         includeReasons);


    // 3.  Must have AdvertiserDomain in bidswitch.crid
    getAttr(result, pconf, "adomain", crinfo->adomain, includeReasons);
    if (crinfo->adomain.empty())
        result.setIncompatible
        ("creative[].providerConfig.bidswitch.adomain is null",
         includeReasons);
    // Cache the information
    result.info = crinfo;

    // 4. Check if the creative has a Google subsection.
    // if so, try to read "vendor type" and "attributes"
    // we do not enforce anything here. If nothing's configured
    // here, Adx traffic will be filtered out.
    if (pconf.isMember("google"))  {
        const Json::Value & pgconf = pconf["google"];
        if (pgconf.isMember("vendorType")) {
            auto ints = stringsToInts(pgconf["vendorType"]);
            crinfo->Google.vendor_type_ = { std::begin(ints), std::end(ints) };
        }
        if (pgconf.isMember("attribute")) {
            auto ints = stringsToInts(pgconf["attribute"]);
            crinfo->Google.attribute_ = { std::begin(ints), std::end(ints) };
        }
    }
    
    // Don't care about result since it's only an optional macro
    configuration_.handleCreativeCompatibility(creative, includeReasons);

    return result;
}

namespace {

struct GoogleObject {
    std::set<int32_t> allowed_vendor_type;
    std::vector<std::pair<int,double>> detected_vertical;
    std::set<int32_t> excluded_attribute;
};


GoogleObject
parseGoogleObject(const Json::Value& gobj) {
    GoogleObject rc;
    if (gobj.isMember("allowed_vendor_type")) {
        const auto& avt = gobj["allowed_vendor_type"];
        if (avt.isArray()) {
            for (auto ii: avt) {
                rc.allowed_vendor_type.insert (ii.asInt());
            }
        }
    }
    if (gobj.isMember("excluded_attribute")) {
        const auto& avt = gobj["excluded_attribute"];
        if (avt.isArray()) {
            for (auto ii: avt) {
                rc.excluded_attribute.insert (ii.asInt());
            }
        }
    }
    if (gobj.isMember("detected_vertical")) {
        const auto& avt = gobj["detected_vertical"];
        if (avt.isArray()) {
            for (auto ii: avt) {
                rc.detected_vertical.push_back ( {ii["id"].asInt(),ii["weight"].asDouble()});
            }
        }
    }
    return rc;
}

struct AdtruthObject {
    uint64_t tdl_millis;
    std::unordered_map<std::string,std::string> dev_insight_map;
    AdtruthObject() : tdl_millis (0L) {}
    void dump () const {
        LOG(bidswitchExchangeConnectorTrace) << "tdl_millis: " << tdl_millis << endl ;
        LOG(bidswitchExchangeConnectorTrace) << "DevInsight: { ";
        for (auto const ii:  dev_insight_map) {
            LOG(bidswitchExchangeConnectorTrace) << ii.first << ":" << ii.second;
        }
        LOG(bidswitchExchangeConnectorTrace) << " }\n";
    }
};

AdtruthObject
parseAdtruthObject(const Json::Value& adt) {
    AdtruthObject rc;
    for (const auto name: adt.getMemberNames()) {
        if (name == "tdl_millis")
            rc.tdl_millis = adt[name].asInt();
        else
            rc.dev_insight_map[name] = adt[name].asString();
    }
    return rc;
}
}// anonymous

std::shared_ptr<BidRequest>
BidSwitchExchangeConnector::
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
    ML::Parse_Context context("Bid Request", payload.c_str(), payload.size());
    res.reset(OpenRTBBidRequestParser::openRTBBidRequestParserFactory("2.2")->parseBidRequest(context, exchangeName(), exchangeName()));

    return res;
}


void
BidSwitchExchangeConnector::
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
    b.nurl = configuration_.expand(crinfo->nurl, {creative, resp, *auction.request, spotNum});
    b.adid = crinfo->adid;
    b.adomain = crinfo->adomain;
    b.iurl = cpinfo->iurl;
}

namespace {
bool empty_intersection(const set<int32_t>& x, const set<int32_t>& y) {
    auto i = x.begin();
    auto j = y.begin();
    while (i != x.end() && j != y.end()) {
        if (*i == *j)
            return false;
        else if (*i < *j)
            ++i;
        else
            ++j;
    }
    return true;
}
}

bool
BidSwitchExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const {
    // return true for non AdX traffic
    const auto& ext = request.ext;
    if (!ext.isMember("google"))
        return true;

    const auto& gobj = ext["google"];
    auto gobj_parsed = parseGoogleObject (gobj);
    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    // check for attributes
    if (false==empty_intersection(
                crinfo->Google.attribute_,
                gobj_parsed.excluded_attribute)) {
        this->recordHit ("google.attribute_excluded");
        return false ;
    }

    // check for vendors
    for (const auto vendor: crinfo->Google.vendor_type_) {
        if (0==gobj_parsed.allowed_vendor_type.count(vendor)) {
            this->recordHit ("google.now_allowed_vendor");
            return false ;
        }
    }

    return true;
}


} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct AtInit {
    AtInit() {
        ExchangeConnector::registerFactory<BidSwitchExchangeConnector>();
        FilterRegistry::registerFilter<BidSwitchWSeatFilter>();
    }
} atInit;
}

