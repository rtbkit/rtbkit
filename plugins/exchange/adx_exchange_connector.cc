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

/*
 *  For the sake of implementing the Google's specific filtering
 *  we have to store, one per AdSlot (OpenRTB::Imp)
 */
struct AdXBidRequest: RTBKIT::BidRequest {
    /**
     * The disallowed attribute ids for the ads that can show in this slot.
     * See publisher-excluded-creative-attributes.txt file in the technical
     * documentation for a list of ids.
     * Indexed by BidRequest.AdSlot.id
     */
    std::unordered_map<int32_t,std::unordered_set<int32_t>> excluded_attribute_;

    /**
     *The allowed vendor types.  See the vendors.txt file in the technical
     * documentation for a list of ids.  When the seller_network is GDN, the
     * vendor ids listed in gdn-vendors.txt in the supporting technical
     * documentation are also allowed.
     * Indexed by BidRequest.AdSlot.id
     */
    std::unordered_map<int32_t,std::unordered_set<int32_t>> allowed_vendor_type_;

    /**
     * The disallowed sensitive ad categories.  See the
     * ad-sensitive-categories.txt file in the technical documentation for a
     * list of ids.  You should enforce these exclusions if you have the ability
     * to classify ads into the listed categories.
     * Indexed by BidRequest.AdSlot.id
     */
    std::unordered_map<int32_t,std::unordered_set<int32_t>> excluded_sensitive_category_;
};


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
        br.device.reset (new OpenRTB::Device);
    auto& dev = *br.device.get();
    const auto& mobile = gbr.mobile ();
    if (mobile.has_platform()) 	dev.os    = mobile.platform() ;
    if (mobile.has_brand()) 	dev.make  = mobile.brand() ;
    if (mobile.has_model()) 	dev.model = mobile.model() ;
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
        if (!br.app) br.app.reset (new OpenRTB::App);
        //
        if (mobile.has_app_id())
        {
            br.app.get()->id = Id(mobile.app_id (), Id::STR);
            br.app.get()->bundle = mobile.app_id();
        }
        else
        {
            // According to the proto file, this case is possible.
            // However, I never saw such bid requests.
            assert (gbr.has_anonymous_id());
            // TODO: check this is valid
            br.app.get()->id = Id(gbr.anonymous_id(),Id::STR);
            br.app.get()->bundle = gbr.anonymous_id();
        }
    }
    else
    {
        //
        if (!br.site) br.site.reset (new OpenRTB::Site);
    }

    if (mobile.has_mobile_device_type())
    {
        switch  (mobile.mobile_device_type())
        {
        case BidRequest_Mobile_MobileDeviceType_HIGHEND_PHONE:
        case BidRequest_Mobile_MobileDeviceType_TABLET:
            br.device.get()->devicetype.val = OpenRTB::DeviceType::MOBILE_OR_TABLET ;
            break;
        default:
            br.device.get()->devicetype.val = OpenRTB::DeviceType::UNSPECIFIED ;
        }
    }

    // will be needed later on to populate the ext field on the openrtb device obj.
    auto& ext = br.device.get()->ext ;
    if (mobile.has_screen_height())      ext.atStr("screen_height")      = mobile.screen_height();
    if (mobile.has_screen_width())       ext.atStr("screen_width")       = mobile.screen_width();
    if (mobile.has_screen_orientation()) ext.atStr("screen_orientation") = mobile.screen_orientation();

    // this bid request should have exactly 1 impression.
    if (br.imp.empty()) br.imp.emplace_back();
    if (mobile.has_is_interstitial_request())
        br.imp.back().instl.val = static_cast<int>(mobile.is_interstitial_request());


    // TODO: verify the mapping between:
    //      https://developers.google.com/adwords/api/docs/appendix/mobileappcategories
    // and IAB Network / Content Categories:
    //      http://www.iab.net/media/file/NE-QA-Guidelines-Final-Release-0610.pdf
    if (br.app)
        for (auto i=0; i < mobile.app_category_ids_size(); i++)
        {
            OpenRTB::ContentCategory cc (to_string(mobile.app_category_ids(i)));
            br.app.get()->cat.push_back(cc);
        }

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
        br.device.reset (new OpenRTB::Device);
    if (!br.site)
        br.site.reset (new OpenRTB::Site);
#if 0
    auto& dev = *br.device.get();
    auto& site = *br.site.get ();
#endif
}

/**
 *   This function attempts to enforce the limits within
 *   which this Exchange Connector works, i.e., put boundaries
 *   around functionalities currently handled.
 *   The bid_requests which don't pass this filter, will cause
 *   the Connector to drop the auction.
 *	TODO: handle all possible cases.
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

    // only deal with traffic from mobile devices;
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
    auto& device = *br.device.get() ;
    if (!device.geo) device.geo.reset (new OpenRTB::Geo);
    auto& ext = br.device.get()->ext ;
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
ParseGbrAdSlot (const GoogleBidRequest& gbr, AdXBidRequest& br)
{
    auto& imp = br.imp;
    imp.clear ();
    for (auto i: boost::irange(0, gbr.adslot_size()))
    {
        const auto& slot = gbr.adslot(i);
        imp.emplace_back();
        auto& spot = imp.back();
        spot.id = Id(slot.id());
        spot.banner.reset (new OpenRTB::Banner) ;
        for (auto i: boost::irange(0,slot.width_size()))
        {
            spot.banner->w.push_back (slot.width(i));
            spot.banner->h.push_back (slot.height(i));
            spot.formats.push_back(Format(slot.width(i),slot.height(i)));
        }
        spot.banner->id = Id(slot.id());


        auto& allowed_vendor_type = br.allowed_vendor_type_[slot.id()];
        for (auto i: boost::irange(0,slot.allowed_vendor_type_size()))
            allowed_vendor_type.insert(slot.allowed_vendor_type(i));

        auto& excluded_attribute  = br.excluded_attribute_[slot.id()];
        for (auto i: boost::irange(0,slot.excluded_attribute_size()))
            excluded_attribute.insert(slot.excluded_attribute(i));

        auto& exluded_sensitive_category = br.excluded_sensitive_category_[slot.id()];
        for (auto i: boost::irange(0,slot.excluded_sensitive_category_size()))
            exluded_sensitive_category.insert(slot.excluded_sensitive_category(i));

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
                spot.banner->pos.val = OpenRTB::AdPosition::UNKNOWN ;
                break;
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

    std::shared_ptr<BidRequest> res (new AdXBidRequest);
    res->timeAvailableMs = deadline_ms () ;

    auto& br = *reinterpret_cast<AdXBidRequest*>(res.get()) ;

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

    auto& device  = *br.device.get();


    // deal with things on BidRequest
    // BidRequest.ip
    if (gbr.has_ip())
    {
        if (3==gbr.ip().length())
            device.ip = binary_to_hexstr(gbr.ip());
        else
            device.ipv6 = binary_to_hexstr(gbr.ip());
    }

    // google_user_id
    if (gbr.has_google_user_id())
    {
        if (!br.user) br.user.reset (new OpenRTB::User) ;
        // TODO: fix Id() so that it can parse 27 char Google ID into GOOG128
        // for now, will fall back to STR type
        br.user->id = Id (gbr.google_user_id());
    }

    // TODO: BidRequest.cookie_version
    if (gbr.has_user_agent()) device.ua = gbr.user_agent();

    // See function comment.
    ParseGbrGeoCriteria (gbr,br);

    // timezone:
    // currently missing from OpenRTB
    // (see: http://code.google.com/p/openrtb/issues/detail?id=44)

    if (br.site)
    {
        assert (gbr.has_url()||gbr.has_anonymous_id());
        br.url = Url(gbr.has_url() ? gbr.url() : gbr.anonymous_id());
        br.site->page = br.url;
    }

    // parse Impression array
    ParseGbrAdSlot(gbr, br);

    if (gbr.has_detected_language())
        device.language = gbr.detected_language();

    // detected verticals:
    // we only deal with verticals in case the bid request contains
    // an OpenRTB::Site object
    if (br.site && gbr.detected_vertical_size() > 0)
    {
#if 0
        // Jeremy mentioned they should be set as segments
        if (!br.user)
        {
            br.user.reset (new OpenRTB::User);
        }
        // push back a Data object ;
        br.user->data.push_back (OpenRTB::Data());
        auto& data = br.user->data.back();
        for (auto i=0; i < gbr.detected_vertical_size(); i++)
        {

        }
#endif //
        // but we rather stick them on the ext attribute of the site
        // Note that we could also stick them on our extended AdXBidRequest
        // TODO: useful?
        auto value = Json::Value ();
        for (auto i: boost::irange(0,gbr.detected_vertical_size()))
        {
            value.atStr(to_string(gbr.detected_vertical(i).id())) =
                gbr.detected_vertical(i).weight();
        }
        auto& ext = br.device.get()->ext ;
        ext.atStr("detected_vertical") = value;
    }
    auto str = res->toJsonStr();
    cerr << "RTBKIT::BidRequest: " << str << endl ;
    return res ;
}

namespace {
using namespace boost;
using namespace xpressive;
using xpressive::smatch;
using xpressive::sregex;
using xpressive::sregex_iterator;
using xpressive::regex_replace;

/**
 *   with dict { {"K1","V1"}, {"K2","V2"}, ...}
 *   return a copy of the input string, in which all the
 *   occurrences of the string "$$Ki$$" are replaced
 *   by the string "Vi"
 */
string
myFormat (const string& in, unordered_map<string,string>& dict)
{
    auto format_dict = [&](smatch const& what) {
        return dict[what[1].str()];
    };
    static sregex envar = "$$" >> (s1 = +_w) >> "$$";
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

    assert (current->responses.size() >= 1) ;
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
        static sregex oe = sregex::compile("\"([^\"]+)\":\"([^\"]+)\"");
        sregex_iterator cur(resp.meta.begin(), resp.meta.end(), oe), end;
        unordered_map<string,string> dict;
        for ( ; cur != end; ++cur)
            dict[(*cur)[1].str()] = (*cur)[2].str();

        // 2. enrich the above dictionary, with
        // ExchangeConnector specific variables
        dict["AUCTION_ID"] = auction.id.toString();

        // 3. populate, substituting whenever necessary
        ad->set_buyer_creative_id(crinfo->creative_id_);
        ad->set_html_snippet(myFormat(crinfo->html_snippet_,dict));
        ad->add_click_through_url(myFormat(crinfo->click_through_url_,dict)) ;
        adslot->set_max_cpm_micros(resp.price.maxPrice.value);
        // TODO: check what that could be
        adslot->set_id(auction.request->imp[0].id.toInt());

#if 0
        cerr << "b.cid =" << Id(resp.agent) << endl;
        cerr << "b.id =" << Id(auction.id, auction.request->imp[0].id) << endl ;
        cerr << "b. =auction.request->imp[0].id =" << auction.request->imp[0].id << endl ;
        cerr << "b.impid = " <<  auction.request->imp[spotNum].id << endl ;
        cerr << "b.price.val = " << USD_CPM(resp.price.maxPrice) << endl ;
        cerr << "b.adm = " << crinfo->adm << endl ;
        cerr << "b.adomain = " << crinfo->adomain.size() << endl ;
        cerr << "b.crid = " << crinfo->crid << endl ;
#endif
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
    return HttpResponse(400, response);
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
    getAttr(result, pconf, "externalId", crinfo->creative_id_, includeReasons);

    // 2.  Must have adx.htmlTemplate that includes AdX's macro
    getAttr(result, pconf, "htmlTemplate", crinfo->html_snippet_, includeReasons);
    if (crinfo->html_snippet_.find("%%WINNING_PRICE%%") == string::npos)
        result.setIncompatible
        ("creative[].providerConfig.adx.html_snippet must contain "
         "encrypted win price macro %%WINNING_PRICE%%",
         includeReasons);

    // 3.  Must have adx.clickThroughUrl
    getAttr(result, pconf, "clickThroughUrl", crinfo->click_through_url_, includeReasons);

    // 3.  Must have adx.agencyId
    getAttr(result, pconf, "agencyId", crinfo->agency_id_, includeReasons);

    string tmp;
    const auto to_int = [] (const string& str) {
        return atoi(str.c_str());
    };

    // 4.  Must have vendorType
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
    // 5.  Must have attribute
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
    // 5.  Must have sensitiveCategory
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
    const auto& adxbr = reinterpret_cast<const AdXBidRequest&>(request);

    // This function is called once per BidRequest.
    // However a bid request can return multiple AdSlot.
    // The creative restrictions do apply per AdSlot.
    // We then check that *all* the AdSlot present in this BidRequest
    // do pass the filter.
    for (const auto& spot: request.imp)
    {
        // the following cast will always work, as the
        // google says: int32 BidRequest.AdSlot.id
        const auto id = static_cast<int32_t>(spot.id.toInt());

        // check excluded attribute
        auto it = adxbr.excluded_attribute_.find (id);
        if (it != adxbr.excluded_attribute_.end())
    for (auto atr: crinfo->attribute_)
                if ( (*it).second.find(atr) != (*it).second.end())
                    return false ;

        // check excluded sensitivity
        it = adxbr.excluded_sensitive_category_.find (id);
        if (it != adxbr.excluded_sensitive_category_.end())
    for (auto atr: crinfo->category_)
                if ( (*it).second.find(atr) != (*it).second.end())
                    return false ;

        // finally check vendor type;
        it = adxbr.allowed_vendor_type_.find (id) ;
        if (it != adxbr.allowed_vendor_type_.end())
    for (auto atr: crinfo->vendor_type_)
                if ( (*it).second.find(atr) == (*it).second.end())
                    return false ;
    }
    return true;
}


} // namespace RTBKIT

