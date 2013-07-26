/* openrtb_bid_request.cc
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
*/

#include "openrtb_bid_request.h"
#include "jml/utils/json_parsing.h"
#include "openrtb/openrtb.h"
#include "openrtb/openrtb_parsing.h"

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* OPENRTB BID REQUEST PARSER                                                */
/*****************************************************************************/

BidRequest *
fromOpenRtb(OpenRTB::BidRequest && req,
            const std::string & provider,
            const std::string & exchange)
{
    std::unique_ptr<BidRequest> result(new BidRequest());

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
            
            result->imp.emplace_back(std::move(spot));

            
        };

    result->imp.reserve(req.imp.size());

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

        if (result->user->ext.isMember("tz"))
            result->location.timezoneOffsetMinutes = result->user->ext["tz"].asInt();

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

    result->ext = std::move(req.ext);
    
    return result.release();
}

namespace {

static DefaultDescription<OpenRTB::BidRequest> desc;

} // file scope

BidRequest *
OpenRtbBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange)
{
    StructuredJsonParsingContext jsonContext(jsonValue);

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);

    return fromOpenRtb(std::move(req), provider, exchange);
}

BidRequest *
OpenRtbBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    StreamingJsonParsingContext jsonContext(context);

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);

    return fromOpenRtb(std::move(req), provider, exchange);
}

} // namespace RTBKIT
