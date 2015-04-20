/* openrtb_bid_request.cc
   Jeremy Barnes, 19 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
*/

#include "openrtb_bid_request.h"
#include "jml/utils/json_parsing.h"
#include "rtbkit/openrtb/openrtb.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/service/logs.h"

using namespace std;

namespace RTBKIT {

namespace {

    Logging::Category openrtbBidRequestTrace("OpenRTB Bid Request");
    Logging::Category openrtbBidRequestError("[ERROR] OpenRTB Bid Request error", openrtbBidRequestTrace);
}

/*****************************************************************************/
/* OPENRTB BID REQUEST PARSER                                                */
/*****************************************************************************/

void parseSDKs(const AdSpot & spot, BidRequest & result){
    std::vector<std::string> sdks;
    for ( OpenRTB::ApiFramework sdk : spot.banner->api){
        switch (sdk.value()) {
            case OpenRTB::ApiFramework::MRAID:
                sdks.push_back("MRAID");
                break;
            case OpenRTB::ApiFramework::ORMMA:
                sdks.push_back("ORMMA");
                break;
            case OpenRTB::ApiFramework::VPAID_1:
                sdks.push_back("VPAID 1.0");
                break;
            case OpenRTB::ApiFramework::VPAID_2:
                sdks.push_back("VPAID 2.0");
                break;
            default:
                break;
        }
    }
    if (!sdks.empty()) result.segments.addStrings("supportedSDKs", sdks);
}

BidRequest *
fromOpenRtb(OpenRTB::BidRequest && req,
            const std::string & provider,
            const std::string & exchange,
            const std::string & version)
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
		parseSDKs(spot, *result.get());
                spot.position = spot.banner->pos;
            
            } else if (spot.video) {
                
                // Unique ptr doesn't overload operators.. great.
                auto & v = *spot.video;

                if(v.mimes.empty()) {
                    // We need at least one MIME type supported by the exchange
                    //LOG(openrtbBidRequestError) << "Video::mimes needs to be populated." << endl;
                    v.mimes.push_back(OpenRTB::MimeType("application/octet-stream"));
                }
            
                if(version == "2.1") {
                    /** 
                    * Refers to table 6.6 of OpenRTB 2.1
                    * Linearity is initialized to -1 if we don't get it in the bid request
                    * so if it's under 0, it means it wasn't given and it can only be 1 or 2
                    */
                    if(v.linearity.value() < -0 ||v.linearity.value() > 2) {
                        //LOG(openrtbBidRequestError) <<"Video::linearity must be specified and match a value in OpenRTB 2.1 Table 6.6." << endl;
                        //LOG(openrtbBidRequestError) <<"Video::linearity has been set to UNSPECIFIED." << endl;
                        v.linearity.val = -1;
                    }
                
                    /** 
                    * Refers to table 6.7 of OpenRTB 2.1
                    * Linearity is initialized to -1 if we don't get it in the bid request
                    * so if it's under -1, it means it wasn't given and it can only be 1 to 6
                    */
                
                    if(v.protocol.value() < 0 || v.protocol.value() > 6) {
                        //LOG(openrtbBidRequestError) << "Video::protocol must be specified and match a value in OpenRTB 2.1 Table 6.7." << endl;
                        //LOG(openrtbBidRequestError) <<"Video::protocol has been set to UNSPECIFIED." << endl;
                        v.protocol.val = -1;
                    }
                }

                if(v.minduration.val < 0) {
                    THROW(openrtbBidRequestError) << "Video::minduration must be specified and positive." << endl;
                }

                if(v.maxduration.val < 0) {
                    THROW(openrtbBidRequestError) << "Video::maxduration must be specified and positive." << endl;
                }
                else if(v.maxduration.val < v.minduration.val) {
                    // Illogical
                    THROW(openrtbBidRequestError) << "Video::maxduration can't be smaller than Video::minduration." << endl;
                }

                spot.position = spot.video->pos;

                Format format(v.w.value(), v.h.value());
                spot.formats.push_back(format);
            }

#if 0
            if (imp.banner) {
                auto & b = *imp.banner;
                
                if (b.w.size() != b.h.size())
                    THROW(openrtbBidRequestError) <<("widths and heights must match");
                
                for (unsigned i = 0;  i < b.w.size();  ++i) {
                    int w = b.w[i];
                    int h = b.h[i];

                    Format format(w, h);
                    spot.formats.push_back(format);
                }

                if (!b.expdir.empty()) {
                    spot.tagFilter.mustInclude.add("expandableTargetingNotSupported");
                }
                if (!b.api.empty()) {
                    spot.tagFilter.mustInclude.add("apiFrameworksNotSupported");
                }
                if (!b.btype.empty()) {
                    spot.tagFilter.mustInclude.add("creativeTypeBlockingNotSupported");
                }
                if (!b.battr.empty()) {
                    spot.tagFilter.mustInclude.add("creativeTypeB");
                    // Blocked creative attributes
                }
                if (!b.mimes.empty()) {
                    // We must have specified a MIME type and it must be
                    // supported by the exchange.
                    
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
            result->imp.emplace_back(std::move(spot));
        };

    result->imp.reserve(req.imp.size());

    for (auto & i: req.imp)
        onImpression(std::move(i));

    if (req.site && req.app)
        THROW(openrtbBidRequestError) << "can't have site and app" << endl;

    if (req.site) {
        result->site.reset(req.site.release());
        if (!result->site->page.empty())
            result->url = result->site->page;
        else if (result->site->id)
            result->url = Url("http://" + result->site->id.toString() + ".siteid/");

        // Adding IAB categories to segments
        for(auto& v : result->site->cat) {
            result->segments.add("iab-categories", v.val);
        }
    }
    else if (req.app) {
        result->app.reset(req.app.release());

        if (!result->app->bundle.empty())
            result->url = Url(result->app->bundle);
        else if (result->app->id)
            result->url = Url("http://" + result->app->id.toString() + ".appid/");
        
        // Adding IAB categories to segments
        for(auto& v : result->app->cat) {
            result->segments.add("iab-categories", v.val);
        }
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
            if(!g.metro.empty())
                l.metro = boost::lexical_cast<int> (g.metro);            
// TODO DMA
        }
    }

    if (req.user) {
        result->user.reset(req.user.release());
        for (auto & d: result->user->data) {
            string key;
            if (d.id)
                key = d.id.toString();
            else key = d.name.extractAscii();

            vector<string> values;
            for (auto & v: d.segment) {
                if (v.id)
                    values.push_back(v.id.toString());
                else if (!v.name.empty())
                    values.push_back(v.name.extractAscii());
            }

            result->segments.addStrings(key, values);
        }

        if (result->user->tz.val != -1)
            result->location.timezoneOffsetMinutes = result->user->tz.val;

        if (result->user->id)
            result->userIds.add(result->user->id, ID_EXCHANGE);
        if (result->user->buyeruid)
            result->userIds.add(result->user->buyeruid, ID_PROVIDER);
        else if (result->user->id)
            result->userIds.add(result->user->id, ID_PROVIDER);
        else if(result->device && !result->device->ip.empty() && !result->device->ua.empty()) {
            const std::string &strToHash = (result->device->ip + result->device->ua.extractAscii());
            result->userAgentIPHash = Id(CityHash64(strToHash.c_str(), strToHash.length()));
            result->userIds.add(result->userAgentIPHash, ID_PROVIDER);
        }
        
        else
            result->userIds.add(Id(0), ID_PROVIDER);

        if (result->user->geo) {
            const auto & ug = *result->user->geo;
            auto & l = result->location;
            if(l.countryCode.empty() && !ug.country.empty())
                l.countryCode = ug.country;
            if(l.regionCode.empty() && !ug.region.empty())
                l.regionCode = ug.region;
            if(l.cityName.empty() && !ug.city.empty())
                l.cityName = ug.city;
            if(l.postalCode.empty() && !ug.zip.empty())
                l.postalCode = ug.zip;
        }

    }
    else
    {
        // We don't receive a user object, we need at least to set provider_ID in order to identify
        // the user

        if(result->device && !result->device->ip.empty() && !result->device->ua.empty()) {
            const std::string &strToHash = (result->device->ip + result->device->ua.extractAscii());
            result->userAgentIPHash = Id(CityHash64(strToHash.c_str(), strToHash.length()));
            result->userIds.add(result->userAgentIPHash, ID_PROVIDER);
        }
        else {
            result->userIds.add(Id(0), ID_PROVIDER);
        }
    }

    if (!req.cur.empty()) {
        for (unsigned i = 0;  i < req.cur.size();  ++i) {
            result->bidCurrency.push_back(parseCurrencyCode(req.cur[i]));
        }
    }
    else {
        result->bidCurrency.push_back(CurrencyCode::CC_USD);
    }

    result->blockedCategories = std::move(req.bcat);

    // blocked tld advertisers
    result->badv = std::move(req.badv);
    vector<string> badv ;
    for (auto s: req.badv)
    	badv.push_back (s.extractAscii());
    result->restrictions.addStrings("badv", badv);


    result->ext = std::move(req.ext);

    result->segments.addStrings("openrtb-wseat", req.wseat);
    
    return result.release();
}

OpenRTB::BidRequest toOpenRtb(const BidRequest &req)
{
    OpenRTB::BidRequest result;

    result.id = req.auctionId; 
    result.at = req.auctionType;
    result.tmax = req.timeAvailableMs;
    result.unparseable = req.unparseable;

    auto onAdSpot = [&](const AdSpot &spot) {
        OpenRTB::Impression imp(spot);

        result.imp.push_back(std::move(imp));
    };

    result.imp.reserve(req.imp.size());
    for (const auto &spot: req.imp) {
        onAdSpot(spot);
    }

    if (req.site && req.app)
        THROW(openrtbBidRequestError) << "can't have site and app" << endl;

    if (req.site) {
        result.site.reset(new OpenRTB::Site(*req.site));
    }
    else if (req.app) {
        result.app.reset(new OpenRTB::App(*req.app));
    }

    if (req.user) {
        result.user.reset(new OpenRTB::User(*req.user));
    }

    if (req.device) {
        result.device.reset(new OpenRTB::Device(*req.device));
    }

    result.bcat = req.blockedCategories;
    result.cur.reserve(req.bidCurrency.size());

    for (const auto &cur: req.bidCurrency) {

        result.cur.push_back(toString(cur));
    }

    result.badv = req.badv;

    result.ext = req.ext;
    const auto &wseatSegments = req.segments.get("openrtb-wseat");
    std::vector<std::string> wseat;
    wseatSegments.forEach([&](int, const std::string &str, float) {
        wseat.push_back(str);
    });

    result.wseat = std::move(wseat);

    return result;
}

namespace {

static DefaultDescription<OpenRTB::BidRequest> desc;

} // file scope


OpenRTB::BidRequest
OpenRtbBidRequestParser::
parseBidRequest(const std::string & jsonValue)
{
    const char * strStart = jsonValue.c_str();
    StreamingJsonParsingContext jsonContext(jsonValue, strStart,
                                            strStart + jsonValue.size());

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);
    return std::move(req);
}


BidRequest *
OpenRtbBidRequestParser::
parseBidRequest(const std::string & jsonValue,
                const std::string & provider,
                const std::string & exchange,
                const std::string & version)
{
    return fromOpenRtb(parseBidRequest(jsonValue), provider, exchange, version);
}

OpenRTB::BidRequest
OpenRtbBidRequestParser::
parseBidRequest(ML::Parse_Context & context)
{
    StreamingJsonParsingContext jsonContext(context);

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);
    return std::move(req);
}

BidRequest *
OpenRtbBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange,
                const std::string & version)
{
    return fromOpenRtb(parseBidRequest(context), provider, exchange, version);
}

} // namespace RTBKIT
