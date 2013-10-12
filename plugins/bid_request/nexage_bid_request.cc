/* nexage_bid_request.cc
   
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
*/

//#include "nexage_parsing.h"

#include "nexage_bid_request.h"
#include <set>

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* Nexage BID REQUEST PARSER                                                */
/*****************************************************************************/

BidRequest *
fromNexage(Nexage::BidRequest && req,
            const std::string & provider,
            const std::string & exchange)
{
    std::unique_ptr<BidRequest> result(new BidRequest());
	
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
            
       
        }
    }

    if (req.user) {
	    
		//Todo: Start working from here by changing the type of the string "Location type" to int "Location type"
		//Do Lexical cast or convert it by making another function
		std::unique_ptr<OpenRTB::User> Users(new OpenRTB::User);
		
		Users->id			=	req.user->id;
		Users->buyeruid		=	req.user->buyeruid;
		Users->yob			=	req.user->yob;
		Users->gender		=	req.user->gender;
		Users->keywords		=	req.user->keywords;
		Users->customdata	=	req.user->customdata;
		 
		Users->geo.country =	req.user->geo.country;
		Users->geo.region	=	req.user->geo.region;
		Users->geo.city		=	req.user->geo.city; // copy ctor, Utf8Strings
		Users->geo.metro	=	req.user->geo.metro;
		Users->geo.zip		=	req.user->geo.zip;
		Users->geo.dma		=	req.user->geo.dma;
		Users->geo.type.val=	boost::lexical_cast<int>(req.user->geo.type.val);

		Users->data			=	req.user->data;
		Users->ext			=	req.user->ext;
		//cerr  << endl<< endl<< endl<< endl << "Ext----> " <<Users->geo.type.val << endl<< endl<< endl<< endl<< endl;
	  //throw ML::Exception("<<<<<<<<Bid Request Data Provider:>>>> " + provider);
		result->user.reset(Users.release());
		
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

    result->ext = std::move(req.ext);
    
    return result.release();
}

namespace {

static DefaultDescription<Nexage::BidRequest> desc;

} // file scope

BidRequest *
NexageBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange)
{
    StructuredJsonParsingContext jsonContext(jsonValue);
	
    Nexage::BidRequest req;
    desc.parseJson(&req, jsonContext);
		//throw ML::Exception("<<<<<<<<First Bid Request Data:>>>> ");
    return fromNexage(std::move(req), provider, exchange);
}

BidRequest *
NexageBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    StreamingJsonParsingContext jsonContext(context);
	
    
	Nexage::BidRequest req;
   
	desc.parseJson(&req, jsonContext);
	
    return fromNexage(std::move(req), provider, exchange);
}

} // namespace RTBKIT
