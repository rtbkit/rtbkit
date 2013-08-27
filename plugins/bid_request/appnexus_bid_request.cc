/* appnexus_bid_request.cc
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
*/


#include "appnexus_openrtb_mapping.h"
#include "appnexus_parsing.h"
#include "appnexus_bid_request.h"
#include <set>

using namespace std;


namespace RTBKIT {

/*****************************************************************************/
/* APPNEXUS BID REQUEST PARSER                                                */
/*****************************************************************************/

BidRequest *
fromAppNexus(const AppNexus::BidRequest & req,
            const std::string & provider,
            const std::string & exchange)
{
    // To save calling vector member function repeatedly
    // TODO validate there is at least one Tag in tags. vector data member will only
    //  be initialized to be typed vector, not to hold any actual objects
    const AppNexus::Tag & reqTag = req.tags.front();

    // OpenRTB::User
    std::unique_ptr<OpenRTB::User> user(new OpenRTB::User);
    user->id = Id(req.bidInfo.userId64.val);
    user->gender = req.bidInfo.gender;
    user->yob.val = req.bidInfo.age.val;

    // OpenRTB::Device
    std::unique_ptr<OpenRTB::Device> device(new OpenRTB::Device);
    // OpenRTB::Geo
    std::unique_ptr<OpenRTB::Geo> geo(new OpenRTB::Geo);
    device->geo.reset(geo.release());
    //
    device->ua = req.bidInfo.userAgent.rawString();
    // AN codes are located in their wiki documentation:
    // https://wiki.appnexus.com/display/adnexusdocumentation/Operating+System+Service 
    // Helper function here converts AN OS code to a string, using the documentation from this URL retrieved as of Jun 2013
    int osCode = req.bidInfo.operatingSystem.val;
    device->os = req.bidInfo.getANDeviceOsStringForCode(osCode);
    device->osv = req.bidInfo.getANDeviceOsVersionStringForCode(osCode);
    // TODO VALIDATION against ISO-639-1
    device->language = req.bidInfo.acceptedLanguages;
    // BUSINESS RULE:
    // - AN only provides a boolean flag for Flash, indicating if it present or absent
    // - OpenRTB only has a 'flashver' field which wants the version of Flash if it is present
    device->flashver = "Flash available - version unknown";
    if (req.bidInfo.noFlash.val || req.bidInfo.noFlash.val == -1) {
        device->flashver = "Flash not available";
    }
    device->ip = req.bidInfo.ipAddress;
    // BUSINESS RULE: AN only supports one IP address field,
    //  so assign to both ORTB 'ip' and 'ipv6' fields
    device->ipv6 = req.bidInfo.ipAddress;
    // TODO Need lookup of AN int code values to strings, from AN docs
    device->carrier = to_string(req.bidInfo.carrier.val);
    // TODO Need lookup of AN int code values to strings, from AN docs
    device->make = to_string(req.bidInfo.make.val);
    // TODO Need lookup of AN int code values to strings, from AN docs
    device->model = to_string(req.bidInfo.model.val);
    // TODO VALIDATION convert to ISO 3166-1 Alpha 3
    device->geo->country = req.bidInfo.country.rawString();
    device->geo->region = req.bidInfo.region.rawString();
    device->geo->city = req.bidInfo.city; // copy ctor, Utf8Strings
    device->geo->zip = req.bidInfo.postalCode;
    device->geo->dma = to_string(req.bidInfo.dma.val);
    // Spec: "Expressed in the format 'snnn.ddddddddddddd,snnn.ddddddddddddd',
    //  south and west are negative, up to 13 decimal places of precision."
    // Example: "38.7875232696533,-77.2614831924438"
    unsigned int splitIdx = req.bidInfo.loc.find(',');
    string lat = req.bidInfo.loc.substr(0, splitIdx);
    string lon = req.bidInfo.loc.substr(splitIdx + 1);
    device->geo->lat.val = boost::lexical_cast<float>(lat);
    device->geo->lon.val = boost::lexical_cast<float>(lon);

    // OpenRTB::Content
    // std::unique_ptr<OpenRTB::Content> content(new OpenRTB::Content);
    // content->url = Url(req.bidInfo.url); // Datacratic::Url from Datacatic Utf8String

    // OpenRTB::Impression
    OpenRTB::Impression impression;
    std::unique_ptr<OpenRTB::Banner> banner(new OpenRTB::Banner);
    impression.banner.reset(banner.release());
    // TODO CONFIRM THIS ASSUMPTION
    // NOTE: Assume for now that AN price units are in full currency units per CPM, e.g. if currency is USD
    // then the reserve_price == '1.00 USD' then this is a price of $1 CPM. i.e. - price not in microdollars etc.
    // Note that OpenRTB mapped field is Impression::bidfloorcur, and its unit is again full units per CPM, e.g. $1 CPM
    // So the code in appnexus_parsing.cc for now simply assumes the AN price equals the OpenRTB price
    // Also, for now, only support USD. 
    impression.bidfloor.val = reqTag.reservePrice.val;

    // TODO Add code in ./plugins/exchange/appnexus_exchange_connector.cc to call its inherited configure() method
    //   and set a key for bid_currency there, and then check it here and set it to the OpenRTB bidfloorcur field
    // string bidfloorcur;
    // req.getParam(AppNexusExchangeConnector::getParameters, bidfloorcur ["bid_currency"]);
    // impression->bidfloorcur = atof(bidfloorcur.str());

    /*
    TODO - come back to this. What we need is:
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
    */
    impression.id = Id(reqTag.auctionId64.val);
    // BUSINESS RULE: AN provides both a 'size' field and a 'sizes' field.
    // OpenRTB provides two fields, ordered lists, for 'w' and 'h'. So values at each index must match
    // and provide a 'WxH' pair. Also, because AN provides both, we first put the values from 'size' into
    // the OpenRTB fields, then, in addition put all the values in from 'sizes'. Also, deduped.
    set<string> dedupedSizes;
    dedupedSizes.insert(reqTag.size);
    dedupedSizes.insert(reqTag.sizes.begin(), reqTag.sizes.end());
    for (string adSizePair : dedupedSizes) {
        splitIdx = adSizePair.find('x');
        int w = boost::lexical_cast<int>(adSizePair.substr(0, splitIdx));
        int h = boost::lexical_cast<int>(adSizePair.substr(splitIdx + 1));
        impression.banner->w.push_back(w);
        impression.banner->h.push_back(h);
    }
    OpenRTB::AdPosition position = convertAdPosition(reqTag.position);
    impression.banner->pos.val = position.val;

    // OpenRTB::Publisher
    std::unique_ptr<OpenRTB::Publisher> publisher1(new OpenRTB::Publisher);
    std::unique_ptr<OpenRTB::Publisher> publisher2(new OpenRTB::Publisher);

    // OpenRTB::Site
    std::unique_ptr<OpenRTB::Site> site(new OpenRTB::Site);
    site->publisher.reset(publisher1.release());
    site->id = reqTag.inventorySourceId;

    // OpenRTB::App
    std::unique_ptr<OpenRTB::App> app(new OpenRTB::App);
    app->publisher.reset(publisher2.release());

    // BUSINESS RULE - This is a weak test for "is this an app or a site"
    //  Can't see a better way to do this in AN, so we see if the Bid has an appId
    if (req.bidInfo.appId == "") { // It's a 'site' and not an 'app'
        site->publisher->id = Id(req.bidInfo.publisherId.val);
    }
    else { // It's an 'app' and not a 'site'
        app->publisher->id = Id(req.bidInfo.publisherId.val);
    }
    // But always just statelessy assign appId. If it's empty, no harm
    app->id = Id(req.bidInfo.appId);

    // BidRequest
    std::unique_ptr<BidRequest> bidRequest(new BidRequest);
    bidRequest->timeAvailableMs = req.bidderTimeoutMs.val;
    bidRequest->device.reset(device.release());
    bidRequest->user.reset(user.release());
    // bidRequest->content.reset(content.release());
    bidRequest->imp.emplace_back(std::move(impression));
    bidRequest->app.reset(app.release());
    bidRequest->site.reset(site.release());

    /*
    if (req.at.value() != OpenRTB::AuctionType::SECOND_PRICE)
        throw ML::Exception("TODO: support 1st price auctions in OpenRTB");

    result->auctionId = std::move(req.id);
    result->auctionType = AuctionType::SECOND_PRICE;
    result->timeAvailableMs = req.tmax.value();
    result->timestamp = Date::now();
    result->isTest = false;
    result->unparseable = std::move(req.unparseable);

    result->provider = provider;
    result->exchange = (exchange.empty() ? provider : exchange);

    auto onImpression = [&] (OpenRTB::Impression && imp)
        {
            AdSpot spot(std::move(imp));

            // Copy the ad formats in for the moment
            if (spot.banner) {
                for (unsigned i = 0;  i < spot.banner->w.size();  ++i) {
                    spot.formats.push_back(Format(spot.banner->w[i],
                                                 spot.banner->h[i]));
                }
            }

            // Now create tags

#if 0


            spot.id = std::move(imp.id);
            if (imp.banner) {
                auto & b = *imp.banner;

                if (b.w.size() != b.h.size())
                    throw ML::Exception("widths and heights must match");

                for (unsigned i = 0;  i < b.w.size();  ++i) {
                    int w = b.w[i];
                    int h = b.h[i];

                    Format format(w, h);
                    spot.formats.push_back(format);
                }

                if (!bexpdir.empty()) {
                    spot.tagFilter.mustInclude.add("expandableTargetingNotSupported");
                }
                if (!bapi.empty()) {
                    spot.tagFilter.mustInclude.add("apiFrameworksNotSupported");
                }
                if (!bbtype.empty()) {
                    spot.tagFilter.mustInclude.add("creativeTypeBlockingNotSupported");
                }
                if (!bbattr.empty()) {
                    spot.tagFilter.mustInclude.add("creativeTypeB");
                    // Blocked creative attributes
                }
                if (!bmimes.empty()) {
                    // We must have specified a MIME type and it must be
                    // supported by the exchange.

                }
            }

            if (!imp.displaymanager.empty()) {
                tags.add("displayManager", imp.displaymanager);
            }
            if (!imp.displaymanagerver.empty()) {
                tags.add("displayManagerVersion", imp.displaymanagerver);
            }
            if (!imp.instl.unspecified()) {
                tags.add("interstitial", imp.instl.value());
            }
            if (!imp.tagid.empty()) {
                tags.add("tagid", imp.tagid.value());
            }
            if (imp.bidfloor.value() != 0.0) {
                if (!imp.bidfloorcur.empty())
                    spot.reservePrice = Amount(imp.bidfloorcur,
                                               imp.bidfloor.value() * 0.001);
                else
                    spot.reservePrice = USD_CPM(imp.bidfloor.value());
            }
            for (b: imp.iframebuster) {
                spot.tags.add("iframebuster", b);
            }
#endif

            result->spots.emplace_back(std::move(spot));


        };

    result->spots.reserve(req.imp.size());

    for (auto & i: req.imp)
        onImpression(std::move(i));

    if (req.site && req.app)
        throw ML::Exception("can't have site and app");

    if (req.site) {
        result->site.reset(req.site.release());
        if (!result->site->page.empty())
            result->url = result->site->page;
        else if (result->site->id)
            result->url = Url("http://" + result->site->id.toString() + ".siteid/");
    }
    else if (req.app) {
        result->app.reset(req.app.release());

        if (!result->app->bundle.empty())
            result->url = Url(result->app->bundle);
        else if (result->app->id)
            result->url = Url("http://" + result->app->id.toString() + ".appid/");
    }

    if (req.device) {
        result->device.reset(req.device.release());
        result->language = result->device->language;
        result->userAgent = result->device->ua;
        if (!result->device->ip.empty())
            result->ipAddress = result->device->ip;
        else if (!result->device->ipv6.empty())
            result->ipAddress = result->device->ipv6;

        if (result->device->geo) {
            const auto & g = *result->device->geo;
            auto & l = result->location;
            l.countryCode = g.country;
            if (!g.region.empty())
                l.regionCode = g.region;
            else l.regionCode = g.regionfips104;
            l.cityName = g.city;
            l.postalCode = g.zip;

            // TODO: DMA
        }
    }

    if (req.user) {
        result->user.reset(req.user.release());
        for (auto & d: result->user->data) {
            string key;
            if (d.id)
                key = d.id.toString();
            else key = d.name;

            vector<string> values;
            for (auto & v: d.segment) {
                if (v.id)
                    values.push_back(v.id.toString());
                else if (!v.name.empty())
                    values.push_back(v.name);
            }

            result->segments.addStrings(key, values);
        }

        if (result->user->tz.val != -1)
            result->location.timezoneOffsetMinutes = result->user->tz.val;

        if (result->user->id)
            result->userIds.add(result->user->id, ID_EXCHANGE);
        if (result->user->buyeruid)
            result->userIds.add(result->user->buyeruid, ID_PROVIDER);
    }

    if (!req.cur.empty()) {
        for (unsigned i = 0;  i < req.cur.size();  ++i) {
            result->bidCurrency.push_back(Amount::parseCurrency(req.cur[i]));
        }
    }
    else {
        result->bidCurrency.push_back(CurrencyCode::CC_USD);
    }
*/

    return bidRequest.release();
}

/*
namespace {

    static Datacratic::DefaultDescription<AppNexus::BidRequest> desc;

} // file scope
*/


BidRequest *
AppNexusBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange)
{
    return fromAppNexus(RTBKIT::fromJson<AppNexus::BidRequest>(jsonValue), provider, exchange);
}

BidRequest *
AppNexusBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    return fromAppNexus(RTBKIT::fromJson<AppNexus::BidRequest>(context), provider, exchange);
}

} // namespace RTBKIT

