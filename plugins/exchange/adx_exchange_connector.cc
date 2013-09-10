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
#include <iterator>  // back_inserter
#include <algorithm> // transform
#include <boost/any.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/range/irange.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <boost/tokenizer.hpp>
#include "adx_exchange_connector.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/plugins/exchange/realtime-bidding.pb.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"

using namespace std;
using namespace Datacratic;
namespace RTBKIT {

/*****************************************************************************/
/* ADX EXCHANGE CONNECTOR                                                */
/*****************************************************************************/

namespace {

__attribute__((constructor))
void
registerConnector() {
    auto factory = [] (ServiceBase * owner, string const & name) {
        return new AdXExchangeConnector(*owner, name);
    };
    ExchangeConnector::registerFactory("adx", factory);
}

} // anonymous namespace

AdXExchangeConnector::
AdXExchangeConnector(ServiceBase & owner, const std::string & name)
    : HttpExchangeConnector(name, owner)
{
}

AdXExchangeConnector::
AdXExchangeConnector(const std::string & name,
                     std::shared_ptr<ServiceProxies> proxies)
    : HttpExchangeConnector(name, proxies)
{
    // useless?
    this->auctionResource = "/auctions";
    this->auctionVerb = "POST";
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
ParseGbrAdSlot (const GoogleBidRequest& gbr, BidRequest& br)
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
    }
}

} // anonymous NS

using namespace boost;
using namespace boost::xpressive;
using xpressive::smatch;
using xpressive::sregex;
using xpressive::sregex_iterator;
using xpressive::regex_replace;

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

    if (gbr.has_google_user_id())
    {
    	// google_user_id
        // TODO: fix Id() so that it can parse 27 char Google ID into GOOG128
        // for now, fall back on STR type
    	br.user->id = Id (gbr.google_user_id());
    }

    if (gbr.has_hosted_match_data())
    {
        br.user->buyeruid = Id(binary_to_hexstr(gbr.hosted_match_data()));
    }

    if (gbr.has_cookie_age_seconds())
    {
        br.user->ext.atStr("cookie_age_seconds") = gbr.cookie_age_seconds();
    }

    // TODO: BidRequest.cookie_version
    if (gbr.has_user_agent())
    {

#if 0
    	// we need to strip funny spurious strings inserted
    	//   ... ,gzip(gfe)[,gzip(gfe)]
        static sregex oe = sregex::compile("(,gzip\\(gfe\\))+$") ;
        device.ua = regex_replace (gbr.user_agent(), oe, string());
#else
        device.ua = gbr.user_agent();
#endif
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
    ParseGbrAdSlot(gbr, br);

    if (gbr.has_detected_language())
        device.language = gbr.detected_language();

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

namespace {
/**
 *   with dict = {{"K1","V1"}, {"K2","V2"}, ...}
 *   return a copy of the input string, in which all the
 *   occurrences of the string "${Ki}" are replaced
 *   by the string "Vi"
 */
string
myFormat (const string& in, unordered_map<string,string>& dict)
{
    auto format_dict = [&](smatch const& what) {
        return dict[what[1].str()];
    };
    static sregex envar = "${" >> (s1 = +_w) >> "}";
    auto rc = regex_replace(in, envar, format_dict);
    return rc;
}
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
        return getErrorResponse(connection, auction, current->error + ": " + current->details);

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

        // handle macros.

        // 1. take care of the agent defined macros,
        // passed along with every bid response, as a
        // stringified JSON object (a la Python)
        unordered_map<string,string> dict;
        auto 	vals = Json::parse (resp.meta);
        for (auto name: vals.getMemberNames())
            dict[name] = vals.atStr(name).asString();

        // 2. enrich the above dictionary, with
        // ExchangeConnector specific variables
        dict["AUCTION_ID"] = auction.id.toString();

        // 3. populate, substituting whenever necessary
        ad->set_buyer_creative_id(crinfo->buyer_creative_id_);
        ad->set_html_snippet(myFormat(crinfo->html_snippet_,dict));
        ad->add_click_through_url(myFormat(crinfo->click_through_url_,dict)) ;
        adslot->set_max_cpm_micros(MicroUSD_CPM(resp.price.maxPrice));
        adslot->set_id(auction.request->imp[spotNum].id.toInt());
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
                 const Auction & auction,
                 const std::string & errorMessage) const
{
    Json::Value response;
    response["error"] = errorMessage;
    return HttpResponse(500, response);
}

namespace {

using Datacratic::jsonDecode;

/** Given a configuration field, convert it to the appropriate JSON */
template<typename T>
void getAttr(ExchangeConnector::ExchangeCompatibility & result,
             const Json::Value & config,
             const char * fieldName,
             T & field,
             bool includeReasons)
{
    try {
        if (!config.isMember(fieldName)) {
            result.setIncompatible
            ("creative[].providerConfig.adx." + string(fieldName)
             + " must be specified", includeReasons);
            return;
        }

        const Json::Value & val = config[fieldName];

        jsonDecode(val, field);
    }
    catch (const std::exception & exc) {
        result.setIncompatible("creative[].providerConfig.adx."
                               + string(fieldName) + ": error parsing field: "
                               + exc.what(), includeReasons);
        return;
    }
}
} // file scope

ExchangeConnector::ExchangeCompatibility
AdXExchangeConnector::
getCreativeCompatibility(const Creative & creative,
                         bool includeReasons) const
{
    ExchangeCompatibility result;
    result.setCompatible();

    auto crinfo = std::make_shared<CreativeInfo>();

    const Json::Value & pconf = creative.providerConfig["adx"];

    // 1.  Must have adx.externalId containing creative attributes.
    getAttr(result, pconf, "externalId", crinfo->buyer_creative_id_, includeReasons);

    // 2.  Must have adx.htmlTemplate that includes AdX's macro
    getAttr(result, pconf, "htmlTemplate", crinfo->html_snippet_, includeReasons);
    if (crinfo->html_snippet_.find("%%WINNING_PRICE%%") == string::npos)
        result.setIncompatible
        ("creative[].providerConfig.adx.html_snippet must contain "
         "encrypted win price macro %%WINNING_PRICE%%",
         includeReasons);

    // 3.  Must have adx.clickThroughUrl
    getAttr(result, pconf, "clickThroughUrl", crinfo->click_through_url_, includeReasons);

    // 4.  Must have adx.agencyId
    //     according to the .proto file this could also be set
    //     to 1 if nothing has been provided in the providerConfig
    getAttr(result, pconf, "agencyId", crinfo->agency_id_, includeReasons);
    if (!crinfo->agency_id_) crinfo->agency_id_ = 1;

    string tmp;
    const auto to_int = [] (const string& str) {
        return atoi(str.c_str());
    };

    // 5.  Must have vendorType
    getAttr(result, pconf, "vendorType", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->vendor_type_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    tmp.clear();
    // 6.  Must have attribute
    getAttr(result, pconf, "attribute", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->attribute_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    tmp.clear();
    // 7.  Must have sensitiveCategory
    getAttr(result, pconf, "sensitiveCategory", tmp, includeReasons);
    if (!tmp.empty())
    {
        tokenizer<> tok(tmp);
        auto& ints = crinfo->category_;
        transform(tok.begin(), tok.end(),
        std::inserter(ints, ints.begin()),[&](const std::string& s) {
            return atoi(s.data());
        });
    }

    // Cache the information
    result.info = crinfo;

    return result;
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
        for (auto atr: crinfo->vendor_type_)
            if (!allowed_vendor_type_seg.contains(atr))
            {
                this->recordHit ("vendor_type_not_allowed");
                return false ;
            }
    }
    return true;
}

} // namespace RTBKIT

