/*
 * adx_exchange_connector.cc
 *
 *  Created on: May 29, 2013
 *      Author: jan sulmont
 *
 *  Implementation of the AdX exchange connector.
 *  see https://developers.google.com/ad-exchange/rtb/getting_started
 *  for details.
 */

#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <string>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string.hpp>
#include "adx_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/plugins/exchange/realtime-bidding.pb.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"

using namespace std;
using namespace Datacratic;
namespace RTBKIT {

/*****************************************************************************/
/* ADX EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

namespace {
struct AtInit {
    AtInit()
    {
	ExchangeConnector::registerFactory<AdXExchangeConnector>();
    }
} atInit;

} // anonymous namespace

AdXExchangeConnector::
AdXExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
    , configuration_("adx")
{
    init();
}

AdXExchangeConnector::
AdXExchangeConnector(const std::string & name,
                     std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
    , configuration_("adx")
{
    // useless?
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";

    init();
}

static std::vector<int> stringsToInts(const Json::Value& value)
{
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

void
AdXExchangeConnector::init()
{
    configuration_.addField(
        "externalId",
        [](const Json::Value & value, CreativeInfo & info) {
            Datacratic::jsonDecode(value, info.buyer_creative_id_);
            return true;
        }
    );

    configuration_.addField(
        "htmlTemplate",
        [](const Json::Value & value, CreativeInfo & info) {
            Datacratic::jsonDecode(value, info.html_snippet_);
            if (info.html_snippet_.find("%%WINNING_PRICE%%") == std::string::npos) {
                throw std::invalid_argument("%%WINNING_PRICE%% price macro expected");
            }
            return true;
        }
    ).snippet();

    configuration_.addField(
        "clickThroughUrl",
        [](const Json::Value & value, CreativeInfo & info) {
            Datacratic::jsonDecode(value, info.click_through_url_);
            return true;
        }
    ).snippet();

    //     according to the .proto file this could also be set
    //     to 1 if nothing has been provided in the providerConfig
    configuration_.addField(
        "agencyId",
        [](const Json::Value & value, CreativeInfo & info)
        {
            Datacratic::jsonDecode(value, info.agency_id_);
            return true;
        }
    ).defaultTo(1);

    configuration_.addField(
        "vendorType",
        [](const Json::Value & value, CreativeInfo & info)
        {
            auto ints = stringsToInts(value);
            info.vendor_type_ = { std::begin(ints), std::end(ints) };
            return true;
        }
    );

    configuration_.addField(
        "attribute",
        [](const Json::Value & value, CreativeInfo & info)
        {
            auto ints = stringsToInts(value);
            info.attribute_ = { std::begin(ints), std::end(ints) };
            return true;
        }
    );

    configuration_.addField(
        "sensitiveCategory",
        [](const Json::Value & value, CreativeInfo & info)
        {
            auto ints = stringsToInts(value);
            info.category_ = { std::begin(ints), std::end(ints) };
            return true;
        }
    );

    /*
          adGroupId is an optional parameter, this must always
          be set if the BidRequest has more than one
          BidRequest.AdSlot.matching_ad_data
    */
    configuration_.addField(
            "adGroupId",
            [](const Json::Value & value, CreativeInfo & info)
            {
                int64_t adGroupId = 0;
                Datacratic::jsonDecode(value, adGroupId);
                info.adgroup_id_ = std::to_string(adGroupId);
                return true;
            }
    ).optional();


    /*
          If you are bidding with ads in restricted categories
          you MUST declare them in restrictedCategories.
    */

    configuration_.addField(
        "restrictedCategories",
        [](const Json::Value & value, CreativeInfo & info)
        {
            auto ints = stringsToInts(value);
            info.restricted_category_ = { std::begin(ints), std::end(ints) };

            if(info.restricted_category_.size() == 1
                && *info.restricted_category_.begin() == 0){
                info.restricted_category_.clear();
            }

            return true;
        }
    );

}

// using GoogleBidRequest = ::BidRequest ;
// using GoogleBidResponse = ::BidResponse ;
typedef ::BidRequest GoogleBidRequest ;
typedef ::BidResponse GoogleBidResponse ;

namespace {

/**
 *    void ParseGbrMobile ()
 *
 *    attempts to convert BidRequest.Mobile sub message.
 *    Creates an OpenRTB::Device and either an OpenRTB::App
 *    or an OpenRTB::Site
 *
 */
void
ParseGbrMobile (const GoogleBidRequest& gbr, BidRequest& br)
{
    if (!br.device)
        br.device.emplace();
    auto& dev = *br.device;
    const auto& mobile = gbr.mobile ();
    if (mobile.has_platform())
        dev.os    = mobile.platform() ;
    if (mobile.has_brand()) dev.make  = mobile.brand() ;
    if (mobile.has_model()) dev.model = mobile.model() ;
    if (mobile.has_os_version())
    {
        dev.osv = "";
        if (mobile.os_version().has_os_version_major())
            dev.osv += to_string(mobile.os_version().os_version_major());
        if (mobile.os_version().has_os_version_minor())
            dev.osv += "." + to_string(mobile.os_version().os_version_minor());
        if (mobile.os_version().has_os_version_micro ())
            dev.osv += "." + to_string(mobile.os_version().os_version_micro ());
    }

    if (mobile.has_carrier_id()) dev.carrier = to_string(mobile.carrier_id()) ;
    if (mobile.is_app())
    {
        if (!br.app) br.app.emplace();
        //
        if (mobile.has_app_id())
        {
            br.app->id = Id(mobile.app_id (), Id::STR);
            br.app->bundle = mobile.app_id();
        }
        else
        {
            // According to the proto file, this case is possible.
            // However, I never saw such bid requests.
            assert (gbr.has_anonymous_id());
            // TODO: check this is valid
            br.app->id = Id(gbr.anonymous_id(),Id::STR);
            br.app->bundle = gbr.anonymous_id();
        }
    }
    else
    {
        //
        if (!br.site) br.site.emplace();
    }

    if (mobile.has_mobile_device_type())
    {
        switch  (mobile.mobile_device_type())
        {
        case BidRequest_Mobile_MobileDeviceType_HIGHEND_PHONE:
        case BidRequest_Mobile_MobileDeviceType_TABLET:
            br.device->devicetype.val = OpenRTB::DeviceType::MOBILE_OR_TABLET ;
            break;
        default:
            break;
        }
    }

    // will be needed later on to populate the ext field on the openrtb device obj.
    auto& ext = br.device->ext ;
    if (mobile.has_screen_height())      ext.atStr("screen_height")      = mobile.screen_height();
    if (mobile.has_screen_width())       ext.atStr("screen_width")       = mobile.screen_width();
    if (mobile.has_screen_orientation()) ext.atStr("screen_orientation") = mobile.screen_orientation();
    if (mobile.has_screen_height()&&mobile.has_screen_width())
        ext.atStr("res") = to_string(mobile.screen_width()) + "x" + to_string(mobile.screen_height());

    // this bid request should have exactly 1 impression.
    if (br.imp.empty()) br.imp.emplace_back();
    if (mobile.has_is_interstitial_request())
        br.imp.back().instl = static_cast<int>(mobile.is_interstitial_request());


    // TODO: verify the mapping between:
    //      https://developers.google.com/adwords/api/docs/appendix/mobileappcategories
    // and IAB Network / 6.1: Content Categories
    //      http://www.iab.net/media/file/OpenRTB-API-Specification-Version-2-1-FINAL.pdf
    if (br.app)
        for (auto i: boost::irange(0, mobile.app_category_ids_size()))
            br.app->cat.emplace_back(to_string(mobile.app_category_ids(i)));

    if (mobile.has_is_mobile_web_optimized())
        ext.atStr("is_mobile_web_optimized") = mobile.is_mobile_web_optimized();
    if (mobile.has_device_pixel_ratio_millis())
        ext.atStr("device_pixel_ratio_millis") = mobile.device_pixel_ratio_millis();

}

/**
 *    void ParseGbrOtherDevice ()
 *    parse non mobile devices.
 *    Will create a OpenRTB::Device and an OpenRTB::Site
 */
void
ParseGbrOtherDevice (const GoogleBidRequest& gbr, BidRequest& br)
{
    if (!br.device)
        br.device.emplace();
    if (!br.site)
        br.site.emplace();
#if 0
    auto& dev = *br.device;
    auto& site = *br.site.get ();
#endif
}

/**
 *   This function attempts to enforce the limits within
 *   which this Exchange Connector works, i.e., put boundaries
 *   around functionalities currently handled.
 *   The bid_requests which don't pass this filter, will cause
 *   the Connector to (safely) ignore the auction.
 *   TODO: handle all possible cases.
 */
bool
GbrIsHandled (const GoogleBidRequest& gbr)
{
    // ping request are of course handled
    if (gbr.has_is_ping() && gbr.is_ping())
        return true ;

    // we currently don't deal with Video.
    if (gbr.has_video())
        return false ;

    // a bid request originated from a device, should
    // either be from an app or a web site
    if (gbr.has_mobile())
    {
        const auto& mob = gbr.mobile();
        if ((mob.has_is_app() && mob.is_app() && !mob.has_app_id()) ||
                !(gbr.has_url() || gbr.has_anonymous_id()))
            return false;
    }
    return true;
}

/**
 *  TODO
 *  Attempt to deal to geo_criteria, if any:
 *    [optional int32 geo_criteria_id    =39;]
 *    [optional string postal_code       =33;]
 *    [optional string postal_code_prefix=34;]
 *  This function (is supposed to) turn:
 *      [optional int32 geo_criteria_id = 39;]
 *  into:
 *      OpenRTB::Geo
 *  object.
 *  This need to be written.
 *  https://commondatastorage.googleapis.com/adx-rtb-dictionaries/geo-table.csv
 *  is actually organized in tree, so that shouldn't be a problem.
 *  For now, we assume that this geo object is some geolocation of the device
 *  itself, and until an ext attribute is available on OpenRTB::Geo (as it should
 *  be as of OpenRTB 2.1),  we stick it on the ext attribute of the OpenRTB::Device
 */
void
ParseGbrGeoCriteria (const GoogleBidRequest& gbr, BidRequest& br)
{
    if (false == gbr.has_geo_criteria_id())
        return ;
    assert (br.device);
    auto& device = *br.device ;
    if (!device.geo) device.geo.emplace();
    auto& ext = br.device->ext ;
    ext.atStr("geo_criteria_id") = gbr.geo_criteria_id() ;

    if (gbr.has_postal_code())
    {
        auto pcode = gbr.has_postal_code_prefix() ?
                     gbr.postal_code_prefix() : "";
        pcode += " " + gbr.postal_code();
        device.geo->zip = pcode;
    }
}

//
void
ParseGbrAdSlot (const std::string currency,
               const CurrencyCode currencyCode,
               const GoogleBidRequest& gbr,
               BidRequest& br)
{
    auto& imp = br.imp;
    imp.clear ();
    for (auto i: boost::irange(0, gbr.adslot_size()))
    {
        const auto& slot = gbr.adslot(i);
        imp.emplace_back();
        auto& spot = imp.back();
        spot.id = Id(slot.id());
        spot.banner.emplace();
        for (auto i: boost::irange(0,slot.width_size()))
        {
            spot.banner->w.push_back (slot.width(i));
            spot.banner->h.push_back (slot.height(i));
            spot.formats.push_back(Format(slot.width(i),slot.height(i)));
        }
        spot.banner->id = Id(slot.id());

        /**
         *   This is what google says about the ad_block_key:
         *   "The BidRequest.AdSlot.ad_block_key field in the RTB protocol contains
         *   a 64-bit integer that provides a stable identifier for the (web_property,
         *   slot, page) combinations. This new field allows you to track the performance
         *    of specific adblock-ad combinations to make better bidding decisions."
         */
        if (slot.has_ad_block_key())
            spot.tagid = to_string(slot.ad_block_key());

        // Parse restrictions ;
        {
            vector<int> tmp ;
            for (auto i: boost::irange(0,slot.allowed_vendor_type_size()))
                tmp.push_back(slot.allowed_vendor_type(i));
            spot.restrictions.addInts("allowed_vendor_type", tmp);

            tmp.clear();
            for (auto i: boost::irange(0,slot.excluded_attribute_size()))
                tmp.push_back(slot.excluded_attribute(i));
            spot.restrictions.addInts("excluded_attribute", tmp);

            tmp.clear();
            for (auto i: boost::irange(0,slot.excluded_sensitive_category_size()))
                tmp.push_back(slot.excluded_sensitive_category(i));
            spot.restrictions.addInts("excluded_sensitive_category", tmp);

            vector<std::string> adg_ids;
            for (auto i: boost::irange(0,slot.matching_ad_data_size())){
                if(slot.matching_ad_data(i).has_adgroup_id()){
                    adg_ids.push_back(
                        boost::lexical_cast<std::string>(
                            slot.matching_ad_data(i).adgroup_id()));
                }
            }
            spot.restrictions.addStrings("allowed_adgroup", adg_ids);

            tmp.clear();
            for (auto i: boost::irange(0,slot.allowed_restricted_category_size()))
                tmp.push_back(slot.allowed_restricted_category(i));
            spot.restrictions.addInts("allowed_restricted_category", tmp);
        }

        if (slot.has_slot_visibility())
        {
            switch (slot.slot_visibility())
            {
            case BidRequest_AdSlot_SlotVisibility_ABOVE_THE_FOLD:
                spot.banner->pos.val = OpenRTB::AdPosition::ABOVE ;
                break;
            case BidRequest_AdSlot_SlotVisibility_BELOW_THE_FOLD:
                spot.banner->pos.val = OpenRTB::AdPosition::BELOW ;
                break;
            default:
                break;
            }
        }

        {
            spot.pmp.emplace();

            for (auto const & matchingAdData : slot.matching_ad_data()) {

                if (matchingAdData.has_adgroup_id())
                {
                    spot.pmp->ext["adgroup_id"] =
                        std::to_string(matchingAdData.adgroup_id());
                }

                for (auto const & directDeal : matchingAdData.direct_deal()) {

                    double amountInCpm =
                        getAmountIn<CPM>(RTBKIT::createAmount<MicroCPM>(
                            directDeal.fixed_cpm_micros(), currencyCode));

                    static const int SECOND_PRICE_AUCTION = 2;
                    OpenRTB::Deal deal
                    {
                        Id(directDeal.direct_deal_id()),
                        amountInCpm,
                        currency,
                        List<std::string>{}, // Assumes empty wseat
                        List<std::string>{}, // Assumes empty wadomain
                        SECOND_PRICE_AUCTION,
                        Json::Value::null
                    };

                    spot.pmp->deals.emplace_back(std::move(deal));
                }
            }
        }
    }
}

} // anonymous NS

std::shared_ptr<BidRequest>
AdXExchangeConnector::
parseBidRequest(HttpAuctionHandler & connection,
                const HttpHeader & header,
                const std::string & payload)
{

    // protocol buffer
    if (header.contentType != "application/octet-stream")
    {
        connection.sendErrorResponse("contentType not octet-stream");
        return std::shared_ptr<BidRequest> ();
    }

    // Try and parse the protocol buffer payload
    GoogleBidRequest gbr;
    if (!gbr.ParseFromString (payload))
    {
        connection.sendErrorResponse("couldn't decode BidRequest message");
        return std::shared_ptr<BidRequest> ();
    }

    // check if this BidRequest is handled by us;
    // or it's a ping.
    if (!GbrIsHandled(gbr) || gbr.is_ping())
    {
        auto msg = gbr.is_ping() ? "pingRequest" : "requestNotHandled" ;
        connection.dropAuction (msg);
        this->recordHit(msg);
        return std::shared_ptr<BidRequest> ();
    }

    std::shared_ptr<BidRequest> res (new BidRequest);
    res->timeAvailableMs = deadline_ms () ;

    auto& br = *res ;

    // TODO: check reuse
    auto binary_to_hexstr = [] (const std::string& str)
    {
        std::ostringstream os;
        os << std::hex << std::setfill('0');
        const unsigned char* pc = reinterpret_cast<const unsigned char*>(str.c_str()) ;
        for (auto i=0 ; i<str.size(); i++,pc++) os << std::setw(2) << int(*pc) ;
        return os.str() ;
    };

    // TODO couldn't get Id() to represent correctly [required bytes id = 2;]
    br.auctionId = Id (binary_to_hexstr(gbr.id()));
    // AdX is a second price auction type.

    br.timestamp = Date::now();
    // TODO: verify the 3 lines are correct
    br.protocolVersion = "Google Protocol Buffer";
    br.exchange = "adx";
    br.provider = "Google";

    // deal with BidRequest.Mobile
    if (gbr.has_mobile())
    {
        ParseGbrMobile (gbr, br);
    }
    else
    {
        ParseGbrOtherDevice (gbr, br);
    }

    assert (br.app || br.site);

    auto& device  = *br.device;

    // deal with things on BidRequest
    // BidRequest.ip
    if (gbr.has_ip())
    {

        if (3==gbr.ip().length())
        {
            struct Ip
            {
                unsigned char a;
                unsigned char b;
                unsigned char c;

                operator std::string() const
                {
                    return std::to_string(uint32_t(a)) + "."
                           + std::to_string(uint32_t(b)) + "."
                           + std::to_string(uint32_t(c)) + ".0";
                }
            } ip = *reinterpret_cast<const Ip*>(gbr.ip().data());
            device.ip = ip;
        }
        else if (6==gbr.ip().size())
        {
            // TODO: proto says that 3 bytes will be transmitted
            // for IPv4 and 6 bytes for IPv6...
            struct Ip
            {
                unsigned char a;
                unsigned char b;
                unsigned char c;
                unsigned char d;
                unsigned char e;
                unsigned char f;

                operator std::string() const
                {
                    return std::to_string(uint32_t(a)) + ":"
                           + std::to_string(uint32_t(b)) + ":"
                           + std::to_string(uint32_t(c)) + ":"
                           + std::to_string(uint32_t(d)) + ":"
                           + std::to_string(uint32_t(e)) + ":"
                           + std::to_string(uint32_t(f));
                }
            } ip = *reinterpret_cast<const Ip*>(gbr.ip().data());
            device.ip = ip;
        }
    }

    if (!br.user)
    {
        // Always create an OpenRTB::User object, since it's recommended
        // in the spec. All top level recommended objects are always created.
    	br.user.emplace();
    }

    bool has_google_user_id = gbr.has_google_user_id();
    bool has_user_agent = gbr.has_user_agent();

    if (has_google_user_id)
    {
    	// google_user_id
        // TODO: fix Id() so that it can parse 27 char Google ID into GOOG128
        // for now, fall back on STR type
    	br.user->id = Id (gbr.google_user_id());
        br.userIds.add(br.user->id, ID_EXCHANGE);
    }

    if (gbr.has_hosted_match_data())
    {
        br.user->buyeruid = Id(binary_to_hexstr(gbr.hosted_match_data()));
        // Provider ID is needed to map different bid requests to the same user
        br.userIds.add(br.user->buyeruid, ID_PROVIDER);
    }
    else
    {
        if(has_google_user_id) {
            // We'll use the google user id for the provider ID
            br.userIds.add(br.user->id, ID_PROVIDER);
        }
        else if (gbr.has_ip() && has_user_agent){
            // Use a hashing function of IP + User Agent concatenation
            br.userAgentIPHash = Id(CityHash64((gbr.ip() + gbr.user_agent()).c_str(),(gbr.ip() + gbr.user_agent()).length()));
            br.userIds.add(br.userAgentIPHash, ID_PROVIDER);
        }
        else {
            // Set provider ID to 0
            br.userIds.add(Id(0), ID_PROVIDER);
        }
    }

    if (gbr.has_cookie_age_seconds())
    {
        br.user->ext.atStr("cookie_age_seconds") = gbr.cookie_age_seconds();
    }

    // TODO: BidRequest.cookie_version
    if (has_user_agent)
    {

#if 0
    	// we need to strip funny spurious strings inserted
    	//   ... ,gzip(gfe)[,gzip(gfe)]
        static sregex oe = sregex::compile("(,gzip\\(gfe\\))+$") ;
        device.ua = regex_replace (gbr.user_agent(), oe, string());
#else
        device.ua = Datacratic::UnicodeString(gbr.user_agent());
#endif
        br.userAgent = Datacratic::UnicodeString(gbr.user_agent());
    }

    // See function comment.
    ParseGbrGeoCriteria (gbr,br);

    // timezone:
    // currently missing from OpenRTB
    // (see: http://code.google.com/p/openrtb/issues/detail?id=44)
    if (gbr.has_timezone_offset())
        device.ext.atStr("timezone_offset") = gbr.timezone_offset();

    if (br.site)
    {
        assert (gbr.has_url()||gbr.has_anonymous_id());
        br.url = Url(gbr.has_url() ? gbr.url() : gbr.anonymous_id());
        br.site->page = br.url;
        if (gbr.has_seller_network_id())
        {
            if (!br.site->publisher) br.site->publisher.emplace();
            br.site->publisher->id = Id(gbr.seller_network_id());
            br.site->id = Id(gbr.seller_network_id());
        }
    }

    // parse Impression array
    ParseGbrAdSlot(getCurrencyAsString(), getCurrency(), gbr, br);

    if (gbr.detected_language_size())
    {   // TODO when gbr.detected_language_size()>1
        device.language = gbr.detected_language(0);
    }

    // detected verticals:
    if (gbr.detected_vertical_size() > 0)
    {
        vector<pair<int,float>> segs;
        for (auto i: boost::irange(0,gbr.detected_vertical_size()))
            segs.emplace_back(make_pair(gbr.detected_vertical(i).id(),
                                        gbr.detected_vertical(i).weight()));
        br.segments.addWeightedInts("AdxDetectedVerticals", segs);
    }
    // auto str = res->toJsonStr();
    // cerr << "RTBKIT::BidRequest: " << str << endl ;
    return res ;
}

/**
 *  prepare a Google BidResponse.
 *  Will handle
 */
HttpResponse
AdXExchangeConnector::
getResponse(const HttpAuctionHandler & connection,
            const HttpHeader & requestHeader,
            const Auction & auction) const
{
    const Auction::Data * current = auction.getCurrentData();

    if (current->hasError())
        return getErrorResponse(connection,current->error + ": " + current->details);

    GoogleBidResponse gresp ;
    gresp.set_processing_time_ms(static_cast<uint32_t>(auction.timeUsed()*1000));

    auto en = exchangeName();

    // Create a spot for each of the bid responses
    for (auto spotNum: boost::irange(0UL, current->responses.size()))
    {

        if (!current->hasValidResponse(spotNum))
            continue ;
        // Get the winning bid
        auto & resp = current->winningResponse(spotNum);

        // Find how the agent is configured.  We need to copy some of the
        // fields into the bid.
        const AgentConfig * config  =
            std::static_pointer_cast<const AgentConfig>(resp.agentConfig).get();

        // Put in the fixed parts from the creative
        int creativeIndex = resp.agentCreativeIndex;

        auto & creative = config->creatives.at(creativeIndex);

        // Get the exchange specific data for this creative
        const auto & crinfo = creative.getProviderData<CreativeInfo>(en);

        auto ad = gresp.add_ad();
        auto adslot = ad->add_adslot() ;


        ad->set_buyer_creative_id(crinfo->buyer_creative_id_);
        ad->set_width(creative.format.width);
        ad->set_height(creative.format.height);

        const BidRequest & br = *auction.request;

        // handle macros.
        AdxCreativeConfiguration::Context ctx { creative, resp, br, static_cast<int>(spotNum) };

        // populate, substituting whenever necessary
        ad->set_html_snippet(
                configuration_.expand(crinfo->html_snippet_, ctx));
        ad->add_click_through_url(
                configuration_.expand(crinfo->click_through_url_, ctx));

        for(auto vt : crinfo->vendor_type_)
            ad->add_vendor_type(vt);
        adslot->set_max_cpm_micros(getAmountIn<MicroCPM>(resp.price.maxPrice));
        adslot->set_id(auction.request->imp[spotNum].id.toInt());

        { // handle direct deals
            auto imp = br.imp[spotNum];
            if (imp.pmp) {
                auto const & pmp = *imp.pmp;
                if (pmp.privateAuction.val != 0) {

                    Json::Value meta;
                    Json::Reader reader;
                    if (!reader.parse(resp.meta.rawString(), meta)) {
                        return getErrorResponse(
                                connection,
                                "Cannot decode meta information");
                    }

                    if (meta.isMember("deal_id")) {
                        Id dealId = Id(meta["deal_id"].asString());
                        adslot->set_deal_id(dealId.toInt());
                    } else {
                        adslot->set_deal_id(1); // open auction
                    }

                    if (meta.isMember("adgroup_id")) {
                        int64_t adgroup_id = stoll(meta["adgroup_id"].asString());
                        adslot->set_adgroup_id(adgroup_id);
                    }
                }
            }
        }

        if(!crinfo->adgroup_id_.empty()) {
            adslot->set_adgroup_id(
                boost::lexical_cast<uint64_t>(crinfo->adgroup_id_));
        }
        for(const auto& cat : crinfo->restricted_category_)
            ad->add_restricted_category(cat);
    }
    return HttpResponse(200, "application/octet-stream", gresp.SerializeAsString());
}

HttpResponse
AdXExchangeConnector::
getDroppedAuctionResponse(const HttpAuctionHandler & connection,
                          const std::string & reason) const
{
    //
    GoogleBidResponse resp ;
    // AdX requires us to set the processing time on an empty BidResponse
    // however we do not have this time here, for there is no auction available.
    // Arbitrary chose to set the processing time to 0 millisecond
    resp.set_processing_time_ms(0);
    return HttpResponse(200, "application/octet-stream", resp.SerializeAsString());
}

HttpResponse
AdXExchangeConnector::
getErrorResponse(const HttpAuctionHandler & connection,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(500, response);
}

ExchangeConnector::ExchangeCompatibility
AdXExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    return configuration_.handleCreativeCompatibility(creative, includeReasons);
}

bool
AdXExchangeConnector::
bidRequestCreativeFilter(const BidRequest & request,
                         const AgentConfig & config,
                         const void * info) const
{
    const auto crinfo = reinterpret_cast<const CreativeInfo*>(info);

    // This function is called once per BidRequest.
    // However a bid request can return multiple AdSlot.
    // The creative restrictions do apply per AdSlot.
    // We then check that *all* the AdSlot present in this BidRequest
    // do pass the filter.
    // TODO: verify performances of the implementation.
    for (const auto& spot: request.imp)
    {

        const auto& excluded_attribute_seg = spot.restrictions.get("excluded_attribute");
        for (auto atr: crinfo->attribute_)
            if (excluded_attribute_seg.contains(atr))
            {
                this->recordHit ("attribute_excluded");
                return false ;
            }

        const auto& excluded_sensitive_category_seg =
            spot.restrictions.get("excluded_sensitive_category");
        for (auto atr: crinfo->category_)
            if (excluded_sensitive_category_seg.contains(atr))
            {
                this->recordHit ("sensitive_category_excluded");
                return false ;
            }

        const auto& allowed_vendor_type_seg =
            spot.restrictions.get("allowed_vendor_type");
	if (!allowed_vendor_type_seg.empty())
        {
            for (auto atr: crinfo->vendor_type_)
                if (!allowed_vendor_type_seg.contains(atr))
                {
                    this->recordHit ("vendor_type_not_allowed");
                    return false ;
                }
	}

        const auto& allowed_adgroup_seg =
            spot.restrictions.get("allowed_adgroup");
        if (!allowed_adgroup_seg.contains(crinfo->adgroup_id_) 
                 && !allowed_adgroup_seg.empty()
                 && !crinfo->adgroup_id_.empty())
        {
            this->recordHit ("adgroup_not_allowed");
            return false ;
        }

        const auto& allowed_restricted_category_seg =
            spot.restrictions.get("allowed_restricted_category");
        for (auto atr: crinfo->restricted_category_)
            if (    (!allowed_restricted_category_seg.contains(atr) 
                  && !allowed_restricted_category_seg.empty())
                ||
                    (allowed_restricted_category_seg.empty()
                  && !crinfo->restricted_category_.empty())
                )
            {
                this->recordHit ("restricted_category_not_allowed");
                return false ;
            }
    }
    return true;
}

} // namespace RTBKIT

