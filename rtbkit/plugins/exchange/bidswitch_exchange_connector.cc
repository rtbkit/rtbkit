/* bidswitch_exchange_connector.cc
   Jeremy Barnes, 15 March 2013

   Implementation of the BidSwitch exchange connector.
*/

#include "bidswitch_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include "soa/service/json_codec.h"
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

namespace {

    template<typename T>
    Json::Value jsonEncode(const std::vector<T>& vec) {
        Json::Value ret(Json::arrayValue);
        ret.resize(vec.size());
        for (typename std::vector<T>::size_type i = 0; i < vec.size(); ++i) {
            ret[i] = vec[i];
        }

        return ret;
    }

}

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

    #define GENERATE_MACRO_FOR(field) \
        [](const Json::Value& value, CreativeInfo& data) { \
            Datacratic::jsonDecode(value, field); \
            return true; \
        }

    /* General fields */

    // nurl might contain macros
    configuration_.addField(
        "nurl",
        [](const Json::Value& value, CreativeInfo& data) {
            Datacratic::jsonDecode(value, data.nurl);

            if (data.nurl.find("${AUCTION_PRICE}") == std::string::npos) {
                throw std::invalid_argument("The ${AUCTION_PRICE} macro is required");
            }
            return true;
        }
    ).required().snippet();
    configuration_.addField(
        "adomain",
        [](const Json::Value& value, CreativeInfo& data) {
            Datacratic::jsonDecode(value, data.adomain);

            if (data.adomain.empty()) {
                throw std::invalid_argument("adomain is required");
            }
            return true;
        }
    ).required();

    configuration_.addField(
        "adid",
        [](const Json::Value& value, CreativeInfo& data) {
            Datacratic::jsonDecode(value, data.adid);

            if (!data.adid) {
                throw std::invalid_argument("adid is required");
            }
            return true;
        }
    ).required();

    configuration_.addField(
        "adm",
        GENERATE_MACRO_FOR(data.adm)
    ).optional().snippet();

    configuration_.addField(
        "advertiser_name",
        GENERATE_MACRO_FOR(data.ext.advertiserName)
    ).optional();

    configuration_.addField(
        "agency_name",
        GENERATE_MACRO_FOR(data.ext.agencyName)
    ).optional();

    configuration_.addField(
        "lpdomain",
        GENERATE_MACRO_FOR(data.ext.lpDomain)
    ).optional();

    configuration_.addField(
        "language",
        GENERATE_MACRO_FOR(data.ext.language)
    ).optional();

    configuration_.addField(
        "vast_url",
        GENERATE_MACRO_FOR(data.ext.vastUrl)
    ).optional().snippet();

    configuration_.addField(
        "duration",
        GENERATE_MACRO_FOR(data.ext.duration)
    ).optional();

    /* Google SSP-specific fields */

    configuration_.addField(
        "google.vendor_type",
        GENERATE_MACRO_FOR(data.google.vendor_type)
    ).optional();

    configuration_.addField(
        "google.attribute",
        GENERATE_MACRO_FOR(data.google.attribute)
    ).optional();

    /* YieldOne SSP-specific fields */

    configuration_.addField(
        "yieldone.creative_type",
        GENERATE_MACRO_FOR(data.yieldOne.creative_type)
    ).optional();

    configuration_.addField(
        "yieldone.creative_category_id",
        GENERATE_MACRO_FOR(data.yieldOne.creative_category_id)
    ).optional();

    #undef GENERATE_MACRO_FOR
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

    std::string exchange = exchangeName();
    const char* name = exchange.c_str();
    if (!config.providerConfig.isMember(name)){
        result.setIncompatible(
            ML::format("providerConfig.%s is null", name), includeReasons);
        return result;
    }
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

ExchangeConnector::ExchangeCompatibility
BidSwitchExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const {
    return configuration_.handleCreativeCompatibility(creative, includeReasons);
}

namespace {

struct GoogleExtension {
    std::set<int32_t> allowed_vendor_type;
    std::vector<std::pair<int,double>> detected_vertical;
    std::set<int32_t> excluded_attribute;

    static GoogleExtension fromJson(const Json::Value& value) {
        GoogleExtension ge;

        auto parseIntSet = [](std::set<int32_t>& out, const Json::Value& value) {
            if (value.isNull() || !value.isArray()) return;

            for (auto elem: value) {
                if (elem.isIntegral()) {
                    out.insert(elem.asInt());
                }
                else {
                    out.clear();
                    return;
                }
            }

        };

        parseIntSet(ge.allowed_vendor_type, value["allowed_vendor_type"]);
        parseIntSet(ge.excluded_attribute, value["excluded_attribute"]);

        if (value.isMember("detected_vertical")) {
            const auto& avt = value["detected_vertical"];
            if (avt.isArray()) {
                for (auto ii: avt) {
                    ge.detected_vertical.push_back (make_pair(ii["id"].asInt(), ii["weight"].asDouble()));
                }
            }
        }

        return ge;
    }
};

struct YieldOneExtension {
    std::vector<std::string> allowed_creative_types;
    std::vector<int32_t> allowed_creative_category_id;

    static YieldOneExtension fromJson(const Json::Value& value) {
        YieldOneExtension ye;

        if (value.isMember("allowed_creative_types")) {
            auto arr = value["allowed_creative_types"];
            if (arr.isArray()) {
                ye.allowed_creative_types.reserve(arr.size());
                for (auto elem: arr) {
                    ye.allowed_creative_types.push_back(elem.asString());
                }
            }
        }

        if (value.isMember("allowed_creative_category_id")) {
            auto arr = value["allowed_creative_category_id"];
            if (arr.isArray()) {
                ye.allowed_creative_category_id.reserve(arr.size());
                for (auto elem: arr) {
                    ye.allowed_creative_category_id.push_back(elem.asInt());
                }
            }
        }

        return ye;
    }
};

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

    //Parsing "ssp" filed
    if (res!=nullptr){
        std::string exchange;
        if (res->ext.isMember("ssp")) {
            exchange =res->ext["ssp"].asString();
        }
        else {
            exchange = exchangeName();
        }
        res->exchange = std::move(exchange);
    }

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

    BidSwitchCreativeConfiguration::Context context {
        creative,
        resp,
        *auction.request,
        spotNum
    };

    // Put in the variable parts
    b.cid = Id(resp.agent);
    b.id = Id(auction.id, auction.request->imp[0].id);
    b.impid = auction.request->imp[spotNum].id;
    b.price.val = USD_CPM(resp.price.maxPrice);
    b.nurl = configuration_.expand(crinfo->nurl, context);
    b.adid = crinfo->adid;
    b.adomain = crinfo->adomain;
    b.iurl = cpinfo->iurl;

    if (!crinfo->adm.empty()) b.adm = configuration_.expand(crinfo->adm, context);

    auto& ext = b.ext;
    if (!crinfo->ext.advertiserName.empty())
       ext["advertiser_name"] = crinfo->ext.advertiserName;
    if (!crinfo->ext.agencyName.empty())
        ext["agency_name"] = crinfo->ext.agencyName;
    if (!crinfo->ext.lpDomain.empty())
        ext["lpdomain"] = jsonEncode(crinfo->ext.lpDomain);
    if (!crinfo->ext.language.empty())
        ext["language"] = crinfo->ext.language;
    if (!crinfo->ext.vastUrl.empty())
        ext["vast_url"] = configuration_.expand(crinfo->ext.vastUrl, context);
    if (crinfo->ext.duration)
        ext["duration"] = crinfo->ext.duration;

    auto googleExt = toExt(crinfo->google);
    if (!googleExt.isNull()) {
        ext["google"] = std::move(googleExt);
    }

    auto yieldOneExt = toExt(crinfo->yieldOne);
    if (!yieldOneExt.isNull()) {
        ext["yieldone"] = std::move(yieldOneExt);
    }
}

Json::Value
BidSwitchExchangeConnector::
getResponseExt(
    const HttpAuctionHandler& connectio,
    const Auction& auction) const {

    Json::Value ext;
    ext["protocol"] = "4.0";

    return ext;
}

template<typename T>
Json::Value jsonEncode(const std::set<T>& set)
{
    Json::Value result(Json::arrayValue);
    result.resize(set.size());

    size_t i = 0;
    for (const auto &elem: set) {
        result[i++] = Datacratic::jsonEncode(elem);
    }

    return result;
}

Json::Value
BidSwitchExchangeConnector::
toExt(const CreativeInfo::Google& ginfo) const {
    Json::Value ext;
    if (!ginfo.vendor_type.empty()) {
        ext["vendor_type"] = jsonEncode(ginfo.vendor_type);
    }

    if (!ginfo.attribute.empty()) {
        ext["attribute"] = jsonEncode(ginfo.attribute);
    }

    return ext;
}

Json::Value
BidSwitchExchangeConnector::
toExt(const CreativeInfo::YieldOne& yinfo) const {
    Json::Value ext;
    if (!yinfo.creative_type.empty()) {
        ext["creative_type"] = yinfo.creative_type;
    }

    if (yinfo.creative_category_id != -1) {
        ext["creative_category_id"] = yinfo.creative_category_id;
    }

    return ext;
}

namespace {
template<typename T>
bool empty_intersection(const set<T>& x, const set<T>& y) {
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

template<typename T, typename U>
bool isInVector(const std::vector<T> &vec, U elem)
{
    static_assert(std::is_convertible<T, U>::value, "Incompatible types");
    if (JML_UNLIKELY(vec.empty())) return false;

    using std::find;  using std::begin;  using std::end;

    auto it = find(begin(vec), end(vec), elem);
    return it != end(vec);
}

bool
BidSwitchExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const {
    /* Check for SSP-specific filters */
    const auto& ext = request.ext;

    using namespace std;

    if (ext.isMember("google")) {
        const auto& gobj = ext["google"];
        auto gext = GoogleExtension::fromJson(gobj);

        const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

        // check for attributes
        if (false == empty_intersection(
                    crinfo->google.attribute,
                    gext.excluded_attribute)) {
            this->recordHit ("google.attribute_excluded");
            return false ;
        }

        auto allowed_vendor_type = std::move(gext.allowed_vendor_type);
        /* Note that if site.publisher.id or app.publisher.id field value equals to “google_1”
           then the vendors listed in https://storage.googleapis.com/adx-rtb-dictionaries/gdn-vendors.txt
           are also allowed for bidding.
        */
        const Id google_1("google_1");
        if (   (request.site && request.site->publisher && request.site->publisher->id == google_1)
            || (request.app && request.app->publisher && request.app->publisher->id == google_1))
        {
            const int32_t gdn_vendors[] = {
                #define ITEM(id, _) \
                   id,
                #include "gdn-vendors.itm"
                #undef ITEM
            };

            std::copy(std::begin(gdn_vendors), std::end(gdn_vendors), std::inserter(allowed_vendor_type, allowed_vendor_type.begin()));

        }

        // check for vendors
        for (const auto vendor: crinfo->google.vendor_type) {
            if (0 == allowed_vendor_type.count(vendor)) {
                this->recordHit ("google.not_allowed_vendor");
                return false ;
            }
        }
        return true;

    } else if (ext.isMember("yieldone")) {
        const auto& yobj = ext["yieldone"];
        auto yext = YieldOneExtension::fromJson(yobj);

        const auto crInfo = reinterpret_cast<const CreativeInfo*>(info);

        if (!isInVector(yext.allowed_creative_types, crInfo->yieldOne.creative_type)) {
            recordHit("yieldOne.not_allowed_creative_type");
            return false;
        }

        if (!isInVector(yext.allowed_creative_category_id, crInfo->yieldOne.creative_category_id)) {
            recordHit("yieldOne.not_allowed_creative_category_id");
            return false;
        }

        return true;
    }

    return true;
}


} // namespace RTBKIT

namespace {
using namespace RTBKIT;

struct AtInit {
    AtInit() {
        ExchangeConnector::registerFactory<BidSwitchExchangeConnector>();
        FilterBase::registerFactory<BidSwitchWSeatFilter>();
    }
} atInit;
}

