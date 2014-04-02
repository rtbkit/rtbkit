/* appnexus_bid_request.cc
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
 */


#include <chrono>

#include "appnexus_parsing.h"
#include "appnexus_bid_request.h"
#include "soa/jsoncpp/value.h"
#include "jml/utils/exc_assert.h"


using namespace std;
using namespace Datacratic;


namespace
{
OpenRTB::AdPosition
convertAdPosition(AppNexus::AdPosition pos)
{
    OpenRTB::AdPosition ret;

    if (pos.value() == AppNexus::AdPosition::UNKNOWN) {
        ret.val = OpenRTB::AdPosition::UNKNOWN;
    }
    else if (pos.value() == AppNexus::AdPosition::ABOVE) {
        ret.val = OpenRTB::AdPosition::ABOVE;
    }
    else if (pos.value() == AppNexus::AdPosition::BELOW) {
        ret.val = OpenRTB::AdPosition::BELOW;
    }
    else { // AN only supports the above three AdPos types.
        // ORTB supports others but AN does not.
        ret.val = OpenRTB::AdPosition::UNSPECIFIED;
    }

    return ret;
}
}  // anonym

namespace RTBKIT {

/*****************************************************************************/
/* APPNEXUS BID REQUEST PARSER                                                */
/*****************************************************************************/
shared_ptr<BidRequest>
fromAppNexus(const AppNexus::BidRequest & req_b,
             const std::string & provider,
             const std::string & exchange)
{
    shared_ptr<BidRequest> rv (new BidRequest);
    auto const& req = req_b.bidRequest;

    rv->timestamp = std::move(Datacratic::Date::parse(req.timestamp.c_str(),
    		                  "%Y-%m-%d %H:%M:%S"));

    ExcAssertEqual (req.tags.size(), 1);
    const auto & reqTag = req.tags.front();

    // OpenRTB::User
    rv->user.reset(new OpenRTB::User);
    rv->user->id = Id(req.bidInfo.userId64.val);
    rv->user->gender = req.bidInfo.gender;
    static struct current_year_st_ {
    	current_year_st_ () {
    		using namespace std;
    		using namespace std::chrono;
    		auto tt = system_clock::to_time_t(system_clock::now());
    		auto utc_tm = *gmtime(&tt);
    		val_ = utc_tm.tm_year + 1900;
    	}
    	int val_;
    } current_year ;
    rv->user->yob.val = current_year.val_ - (req.bidInfo.age.val<=0?0:req.bidInfo.age.val);
    rv->device.reset (new OpenRTB::Device);
    rv->device->geo.reset(new OpenRTB::Geo);
    rv->device->ua = req.bidInfo.userAgent.utf8String();
    int osCode = req.bidInfo.operatingSystem.val;
    rv->device->os = req.bidInfo.getANDeviceOsStringForCode(osCode);
    rv->device->osv = "N/A";
    rv->device->language = req.bidInfo.acceptedLanguages;
    rv->device->flashver = req.bidInfo.noFlash.val == -1 ? "Flash not available"
    		: "Flash available - version unknown";
    rv->device->ip = req.bidInfo.ipAddress;
    rv->device->carrier = to_string(req.bidInfo.carrier.val);
    rv->device->make = to_string(req.bidInfo.make.val);
    rv->device->model = to_string(req.bidInfo.model.val);

    // TODO VALIDATION convert to ISO 3166-1 Alpha 3
    rv->device->geo->country = req.bidInfo.country.utf8String();
    rv->device->geo->region = req.bidInfo.region.utf8String();
    rv->device->geo->city = req.bidInfo.city; // copy ctor, Utf8Strings
    rv->device->geo->zip = req.bidInfo.postalCode;
    rv->device->geo->dma = to_string(req.bidInfo.dma.val);

    auto ix = req.bidInfo.loc.find(',');
    if (string::npos!=ix)
    {
    	rv->device->geo->lat.val = boost::lexical_cast<float>(req.bidInfo.loc.substr(0, ix));
    	rv->device->geo->lon.val = boost::lexical_cast<float>(req.bidInfo.loc.substr(ix + 1));
    }

    if (!req.bidInfo.url.empty())
    	rv->url = std::move(Url(req.bidInfo.url));

    // Impression
    rv->imp.emplace_back (AdSpot());
    auto& impression = rv->imp[0];

    impression.banner.reset(new OpenRTB::Banner);

    // TODO CONFIRM THIS ASSUMPTION
    // NOTE: Assume for now that AN price units are in full currency units per CPM, e.g. if currency is USD
    // then the reserve_price == '1.00 USD' then this is a price of $1 CPM. i.e. - price not in microdollars etc.
    // Note that OpenRTB mapped field is Impression::bidfloorcur, and its unit is again full units per CPM, e.g. $1 CPM
    // So the code in appnexus_parsing.cc for now simply assumes the AN price equals the OpenRTB price
    // Also, for now, only support USD.
    impression.bidfloor.val = reqTag.reservePrice.val;

    /*******************************************
    (still) TODO - come back to this. What we need is:
    - access to AN docs
    - enums for AN for Tag::allowedMediaTypes
    - identify the AN mediaTypes that are for "banner" and for "video"
    - have conditional code that sets OpenRTB Impression::Video.topframe or Impression::Banner.topframe depending
    auto iframePosn = FramePosition::TOPFRAME;
    if (! req.bidInfo.withinIframe) {
        iframePosn = FramePosition::IFRAME;
    }
    if (reqTag.allowedMediaTypes == "banner") {
        impression->banner.topframe = iframePosn;
    }
    else if (reqTag.allowedMediaTypes == "video") {
        impression->video.topframe = iframePosn;
    }
    *******************************************/

    impression.id = Id(reqTag.auctionId64.val);

    for (const string& s : reqTag.sizes)
    {
        ix = s.find('x');
        if (string::npos!=ix)
        {
            impression.banner->w.push_back(boost::lexical_cast<int>(s.substr(0, ix)));
            impression.banner->h.push_back(boost::lexical_cast<int>(s.substr(ix + 1)));
        }
    }

    OpenRTB::AdPosition position = convertAdPosition(reqTag.position);
    impression.banner->pos.val = position.val;

    // OpenRTB::Site
    rv->site.reset (new OpenRTB::Site);
    rv->site->publisher.reset(new OpenRTB::Publisher);
    rv->site->id = reqTag.inventorySourceId;

    // OpenRTB::App
    rv->app.reset (new OpenRTB::App);
    rv->app->publisher.reset(new OpenRTB::Publisher);

    // BUSINESS RULE - This is a weak test for "is this an app or a site"
    //  Can't see a better way to do this in AN, so we see if the Bid has an appId
    if (req.bidInfo.appId == "")  // It's a 'site' and not an 'app'
        rv->site->publisher->id = Id(req.bidInfo.publisherId.val);
    else  // It's an 'app' and not a 'site'
        rv->app->publisher->id = Id(req.bidInfo.publisherId.val);
    // But always just statelessy assign appId. If it's empty, no harm
    rv->app->id = Id(req.bidInfo.appId);

    rv->timeAvailableMs = req.bidderTimeoutMs.val;
    rv->auctionId = move(Id(reqTag.auctionId64.val));
    rv->auctionType = AuctionType::SECOND_PRICE;
    rv->timeAvailableMs = req.bidderTimeoutMs.val;
    rv->isTest = req.test.val ? true : false ;
    rv->unparseable = std::move(req.unparseable);
    rv->provider = provider;
    rv->exchange = (exchange.empty() ? provider : exchange);

    // deal with the members. needed in BidResponse
    vector<int> tmp ;
    for (const auto& m: req.members)
       tmp.push_back (m.id.toInt());
    if (!tmp.empty())
       rv->restrictions.addInts("members", tmp);
    // deal with excluded attributes. needed in BidResponse
    tmp.clear();
    for (const auto& m: req.excludedAttributes)
       tmp.push_back (m.val);
    if (!tmp.empty())
    	rv->restrictions.addInts("excluded_attributes", tmp);
    return rv;
}

shared_ptr<BidRequest>
AppNexusBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange)
{
    StructuredJsonParsingContext jsonContext(jsonValue);
    AppNexus::BidRequest req;
    Datacratic::DefaultDescription<AppNexus::BidRequest> desc;
    desc.parseJson(&req, jsonContext);
    if (!(req.unparseable.isNull() && req.bidRequest.unparseable.isNull()))
    {
        auto str = req.unparseable.isNull() ?
                   req.bidRequest.unparseable.toString() :
                   req.unparseable.toString();
        cerr << "\n\n\n/*** WARNING!!! coudln't parse the following element: "
             << str
             << " INPUT IGNORED!!! ***/\n\n\n";
        return shared_ptr<BidRequest>();
    }
    return fromAppNexus(req, provider, exchange);
}

shared_ptr<BidRequest>
AppNexusBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    StreamingJsonParsingContext jsonContext(context);
    AppNexus::BidRequest req;
    Datacratic::DefaultDescription<AppNexus::BidRequest> desc;
    desc.parseJson(&req, jsonContext);
    if (!(req.unparseable.isNull() && req.bidRequest.unparseable.isNull()))
    {
        auto str = req.unparseable.isNull() ?
                   req.bidRequest.unparseable.toString() :
                   req.unparseable.toString();
        cerr << "\n\n\n/*** WARNING!!! coudln't parse the following element: "
             << str
             << " INPUT IGNORED!!! ***/\n\n\n";
        return shared_ptr<BidRequest>();
    }
    return fromAppNexus(req, provider, exchange);
}

} // namespace RTBKIT

