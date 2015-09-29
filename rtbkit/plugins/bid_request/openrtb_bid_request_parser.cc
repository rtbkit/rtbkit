/* openrtb_bid_request_parser.cc
   Jean-Michel Bouchard, 21 August 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   Bid request parser for OpenRTB.
*/

#include "openrtb_bid_request_parser.h"
#include "jml/utils/json_parsing.h"
#include "rtbkit/openrtb/openrtb.h"
#include "rtbkit/openrtb/openrtb_parsing.h"

using namespace std;

namespace RTBKIT {

    Logging::Category OpenRTBBidRequestLogs::trace("OpenRTB Bid Request Parser");
    Logging::Category OpenRTBBidRequestLogs::error("[ERROR] OpenRTB Bid Request Parser error", OpenRTBBidRequestLogs::trace);
    Logging::Category OpenRTBBidRequestLogs::trace22("OpenRTB Bid Request 2.2 Parser ");
    Logging::Category OpenRTBBidRequestLogs::error22("[ERROR] OpenRTB Bid Request Parser 2.2 error", OpenRTBBidRequestLogs::trace22);

    static DefaultDescription<OpenRTB::BidRequest> desc;

namespace { const char* DefaultVersion = "2.2"; }

std::unique_ptr<OpenRTBBidRequestParser>
OpenRTBBidRequestParser::
openRTBBidRequestParserFactory(const std::string & version) 
{

    if(version == "2.0" || version == "2.1") {
        return std::unique_ptr<OpenRTBBidRequestParser2point1>(new OpenRTBBidRequestParser2point1());
    } else if(version == "2.2") {
        return std::unique_ptr<OpenRTBBidRequestParser2point2>(new OpenRTBBidRequestParser2point2());
    }

    THROW(OpenRTBBidRequestLogs::error) << "Version : " << version << " not supported in RTBkit." << endl;
}

OpenRTB::BidRequest
OpenRTBBidRequestParser::
toBidRequest(const RTBKIT::BidRequest & br) {

    OpenRTB::BidRequest result;

    result.id = br.auctionId;
    result.at = br.auctionType;
    result.tmax = br.timeAvailableMs;
    result.unparseable = br.unparseable;

    result.imp.reserve(br.imp.size());

    for(const auto & spot : br.imp) {
        OpenRTB::Impression imp(spot);

        // Since it's openrtb 2.1, make sure none of the 2.n fields are added.
        imp.pmp.reset();

        if (imp.banner) {
            auto& banner = *imp.banner;
            ExcAssertEqual(banner.h.size(), banner.w.size());

            // openrtb only supports a single value in the h and w fields so any
            // extra values are relocated to the ext field.
            if (banner.h.size() > 1) {

                for (const auto& h : banner.h) banner.ext["h"].append(h);
                for (const auto& w : banner.w) banner.ext["w"].append(w);

                banner.h.resize(1);
                banner.w.resize(1);
            }
        }

        result.imp.push_back(std::move(imp));
    }

    if(br.site && br.app)
        THROW(OpenRTBBidRequestLogs::error) << "OpenRTB::BidRequest cannot have site and app." << endl;

    if(br.site)
        result.site.reset(new OpenRTB::Site(*br.site));
    else if(br.app)
        result.app.reset(new OpenRTB::App(*br.app));

    if(br.user)
        result.user.reset(new OpenRTB::User(*br.user));

    if(br.device)
        result.device.reset(new OpenRTB::Device(*br.device));

    result.bcat = br.blockedCategories;
    result.cur.reserve(br.bidCurrency.size());

    for(const auto & cur : br.bidCurrency) {
        result.cur.push_back(toString(cur));
    }

    result.badv = br.badv;

    result.ext = br.ext;

    const auto & wseatSegments = br.segments.get("openrtb-wseat");

    std::vector<std::string> wseat;

    wseatSegments.forEach([&](int, const std::string & str, float) {
        wseat.push_back(str);        
    });

    return result;
}

OpenRTB::BidRequest
OpenRTBBidRequestParser::
parseBidRequest(const std::string & jsonValue)
{
    const char * strStart = jsonValue.c_str();
    StreamingJsonParsingContext jsonContext(jsonValue, strStart,
                                            strStart + jsonValue.size());

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);
    return std::move(req);
}

OpenRTB::BidRequest
OpenRTBBidRequestParser::
parseBidRequest(ML::Parse_Context & context)
{
    StreamingJsonParsingContext jsonContext(context);

    OpenRTB::BidRequest req;
    desc.parseJson(&req, jsonContext);
    return std::move(req);
}

RTBKIT::BidRequest *
OpenRTBBidRequestParser::
createBidRequestHelper(OpenRTB::BidRequest & br,
                       const std::string & provider,
                       const std::string & exchange) {

    // Create context
    ctx.br = std::unique_ptr<BidRequest>(new BidRequest());
    
    ctx.br->timestamp = Date::now();
    ctx.br->isTest = false;
    // Assign provider and exchange if available
    ctx.br->provider = provider;
    ctx.br->exchange = (exchange.empty() ? provider : exchange);

    this->onBidRequest(br);

    // Transfer app, site, device, user to br
    ctx.br->app.reset(br.app.release());
    ctx.br->site.reset(br.site.release());
    ctx.br->device.reset(br.device.release());
    ctx.br->user.reset(br.user.release());

    // Release control upon exit
    return ctx.br.release();

}

RTBKIT::BidRequest *
OpenRTBBidRequestParser::
parseBidRequest(ML::Parse_Context & context,
                const std::string & provider,
                const std::string & exchange)
{
    // Parse using Parse_Context
    auto br = parseBidRequest(context);
    return createBidRequestHelper(br,
                                  provider,
                                  exchange);
}

RTBKIT::BidRequest *
OpenRTBBidRequestParser::
parseBidRequest(const std::string & json,
                const std::string & provider,
                const std::string & exchange)
{
    // Parse using json string
    auto br = parseBidRequest(json);
    return createBidRequestHelper(br,
                                  provider,
                                  exchange);
}

void 
OpenRTBBidRequestParser::
onBidRequest(OpenRTB::BidRequest & br) {

    ctx.br->auctionId = br.id;

    // Check for at to be 1 or 2
    if(br.at.val == 1 || br.at.val == 2)
        ctx.br->auctionType = AuctionType(br.at);
    else {
        ctx.br->auctionType = AuctionType::SECOND_PRICE;
    }

    ctx.br->timeAvailableMs = br.tmax.value();
    ctx.br->unparseable = std::move(br.unparseable);

    // Reserve enough space for imps
    ctx.br->imp.reserve(br.imp.size());

    // Bid object is composed of impressions
    for (auto & i : br.imp)
        this->onImpression(i);

    // Should only contain site or app
    if (br.site && br.app)
        THROW(OpenRTBBidRequestLogs::error) << " can't have site and app in one openrtb bid request" << endl;
    
    if (br.site) {
        // Contains one site object
        this->onSite(*br.site);
    } else if(br.app) {
        // Contains one app object
        this->onApp(*br.app);
    }

    // User object
    if(br.user)
        this->onUser(*br.user);

    // Device
    // Note : It's important that onUser() is called before onDevice()
    // since the br.userIds field may be populated with device information
    // if no user information is available
    if(br.device) {
        this->onDevice(*br.device);
    }

    // Do we have an ID_PROVIDER (stored as "prov" key) in ctx.br->userIds? 
    if(ctx.br->userIds.count("prov") == 0)
        // 4) Add Id(0) since it's required.
        ctx.br->userIds.add(Id(0), ID_PROVIDER);

    // wseat // allowable buyer seats
    // Put them into a segment called openrtb-wseat which we can use for filtering after
    ctx.br->segments.addStrings("openrtb-wseat", br.wseat);

    // Currencies allowed to bid
    if (!br.cur.empty()) {
        for(auto curr : br.cur)
            ctx.br->bidCurrency.push_back(parseCurrencyCode(curr));
    } else {
        // Assume USD
        ctx.br->bidCurrency.push_back(CurrencyCode::CC_USD);
    }

    // Blocked cats if any, put into restriction segment
    std::vector<string> bcats;
    for(auto b : br.bcat)
        bcats.push_back(b.val);

    ctx.br->restrictions.addStrings("bcat", bcats);
    ctx.br->blockedCategories = std::move(br.bcat);

    // Blocked advertisers, put into restrictions segment
    std::vector<string> badvs;
    for(auto b : br.badv)
        badvs.push_back(b.utf8String());

    ctx.br->restrictions.addStrings("badv", badvs);
    ctx.br->badv = std::move(br.badv);

    // Ext
    ctx.br->ext = std::move(br.ext);

}

void 
OpenRTBBidRequestParser::
onImpression(OpenRTB::Impression & impression) {

    ctx.spot = AdSpot(std::move(impression));
/*
    if(!ctx.spot->banner && !ctx.spot->video)
        LOG(openrtbBidRequestError) << "br.imp must included either a video or a banner object." << endl;
*/  
    // Possible to have a video and a banner object.
    if(ctx.spot.banner) {
        this->onBanner(*ctx.spot.banner);
    }

    if(ctx.spot.video) {
        if(ctx.spot.banner && ctx.spot.banner->id == Id("0"))
            LOG(OpenRTBBidRequestLogs::trace) << "It's recommended to include br.imp.banner.id when subordinate to video object." << endl;
        this->onVideo(*ctx.spot.video);
    }

    // TODO Support tagFilters / mime filers

    ctx.br->imp.emplace_back(std::move(ctx.spot));
}

void
OpenRTBBidRequestParser::
onBanner(OpenRTB::Banner & banner) {

    // RTBKit allows multiple banner sizes restrictions
    if((banner.w.size() == 0) and (banner.h.size() == 1)) {
        banner.w.push_back(0);
    }
    if((banner.w.size() == 1) and (banner.h.size() == 0)) {
        banner.h.push_back(0);
    }

    if(banner.w.size() != banner.h.size())
        LOG(OpenRTBBidRequestLogs::error) << "Mismatch between number of width and heights illegal." << endl;

    for(unsigned int i = 0; i < banner.w.size(); ++i) {
        ctx.spot.formats.push_back(Format(banner.w[i], banner.h[i]));
    }

    // Add api to the segments in order to filter on it
    for(auto & api : banner.api) {
        auto framework = apiFrameworks.find(api.val);
        if (framework != apiFrameworks.end())
            ctx.br->segments.add("api-banner", framework->second , 1.0);
    }
    ctx.spot.position = banner.pos;
}

void 
OpenRTBBidRequestParser::
onVideo(OpenRTB::Video & video) {
    
    if(video.mimes.empty()) {
        //LOG(OpenRTBBidRequestLogs::error) << "br.imp.video.mimes needs to be populated." << endl;
        video.mimes.push_back(OpenRTB::MimeType("application/octet-stream"));
    }

    if(video.linearity.value() < 0 || video.linearity.value() > 2) {
        //LOG(OpenRTBBidRequestLogs::error) <<"Video::linearity must be specified and match a value in OpenRTB 2.1 Table 6.6." << endl;
        //LOG(OpenRTBBidRequestLogs::error) <<"Video::linearity has been set to UNSPECIFIED." << endl;
        video.linearity.val = -1;
    }

    if(video.protocol.value() < 0 || video.protocol.value() > 6) {
        //LOG(OpenRTBBidRequestLogs::error) << "br.imp.video.protocol must be specified and match a value in OpenRTB 2.1 Table 6.7." << endl;
        //LOG(OpenRTBBidRequestLogs::error) <<"Video::protocol has been set to UNSPECIFIED." << endl;
        video.protocol.val = -1;
    }

    if(video.minduration.val < 0) {
        THROW(OpenRTBBidRequestLogs::error) << "br.imp.video.minduration must be specified and positive." << endl;
    }

    if(video.maxduration.val < 0) {
        THROW(OpenRTBBidRequestLogs::error) << "br.imp.video.maxduration must be specified and positive." << endl;
    } else if (video.maxduration.val < video.minduration.val) {
        // Illogical
        THROW(OpenRTBBidRequestLogs::error) << "br.imp.video.maxduration can't be smaller than br.imp.video.minduration." << endl;
    } 

    ctx.spot.position = video.pos;

    // Add api to the segments in order to filter on it
    for(auto & api : video.api) {
        auto framework = apiFrameworks.find(api.val);
        if (framework != apiFrameworks.end())
            ctx.br->segments.add("api-video", framework->second, 1.0);
    }
    ctx.spot.formats.push_back(Format(video.w.value(), video.h.value()));
}

void
OpenRTBBidRequestParser::
onSite(OpenRTB::Site & site) {

    this->onContext(site);

    // Try to define url since we use it in RTBKIT::BidRequest
    if(!site.page.empty())
        ctx.br->url = site.page;
    else if (site.id)
        ctx.br->url = Url("http://" + site.id.toString() + ".siteid/");
}

void
OpenRTBBidRequestParser::
onApp(OpenRTB::App & app) {

    this->onContext(app);
    
    // Try to define url since we use it in RTBKIT::BidRequest
    if(!app.bundle.empty())
        ctx.br->url = Url(app.bundle);
    else if (app.id)
        ctx.br->url = Url("http://" + app.id.toString() + ".appid/");
}

// Helper method for App / Site
void
OpenRTBBidRequestParser::
onContext(OpenRTB::Context & context) {
    
    // Adding IAB categories to segments for filtering
    for(auto & v : context.cat) {
        ctx.br->segments.add("iab-categories", v.val);
    }

    // Adding sub section IAB categories to segments for filtering
    for(auto & v : context.sectioncat) {
        ctx.br->segments.add("subsection-iab-categories", v.val);
    }

    // Adding page IAB categories to segments for filtering
    for(auto & v : context.pagecat) {
        ctx.br->segments.add("page-iab-categories", v.val);
    }

    // Publisher object
    if(context.publisher)
        this->onPublisher(*context.publisher);

    // Content object
    if(context.content)
        this->onContent(*context.content);

}

void 
OpenRTBBidRequestParser::
onContent(OpenRTB::Content & content) {
    // Nothing for now
}

void
OpenRTBBidRequestParser::
onProducer(OpenRTB::Producer & producer) {
    // Nothing for now
}

void 
OpenRTBBidRequestParser::
onPublisher(OpenRTB::Publisher & publisher) {
    // Nothing for now
}

void
OpenRTBBidRequestParser::
onDevice(OpenRTB::Device & device) {

    ctx.br->language = device.language;
    ctx.br->userAgent = device.ua;
    
    if(!device.ip.empty())
        ctx.br->ipAddress = device.ip;
    else if(!device.ipv6.empty())
        ctx.br->ipAddress = device.ipv6;
    // Assign ctx.br->userAgentIPHash
    if(!device.ua.empty()) {
        const std::string &strToHash = (device.ip + device.ua.extractAscii());
        ctx.br->userAgentIPHash = Id(CityHash64(strToHash.c_str(), strToHash.length()));
        // Do we have a user id provider (set as "prov" key) (was in set in onUser()) ?
        // If not 3) add user agent + ip hash
        if(ctx.br->userIds.count("prov") == 0)
            ctx.br->userIds.add(ctx.br->userAgentIPHash, ID_PROVIDER);
    }

    if(device.geo) {
        this->onGeo(*device.geo);
    }
}

void
OpenRTBBidRequestParser::
onGeo(OpenRTB::Geo & geo) {

    auto & loc = ctx.br->location;

    // Validation that lat is -90 to 90
    if(geo.lat.val > 90.0 || geo.lat.val < -90.0)
        LOG(OpenRTBBidRequestLogs::trace) << " br.device.geo.lat : " << geo.lat.val << 
                                             " is invalid and should be within -90 to 90." << " ReqID: " << ctx.br->auctionId << endl; 


    // Validation that lat is -180 to 180
    if(geo.lon.val > 180.0 || geo.lon.val < -180.0)
        LOG(OpenRTBBidRequestLogs::trace) << " br.device.geo.lon : " << geo.lon.val << 
                                             " is invalid and should be within -180 to 180." << " ReqID: " << ctx.br->auctionId << endl; 

    // Validate ISO-3166 Alpha 3 for country
    if(!geo.country.empty()) {/*
        if(geo.country.size() != 3) 
            LOG(OpenRTBBidRequest::trace) << " br.device.geo.country : " << geo.country <<  
                                             " is invalid and doesn't respect ISO-3166 1 Alpha-3" << endl;
*/      
        // TODO
        // Maybe create a map / flat file with the valid codes and test against it 
        if(loc.countryCode.empty())
            loc.countryCode = geo.country;
    }

    if(!geo.region.empty()) {
        // TODO
        // Maybe create a map / flat file with the valid ISO-3166 2 codes and test against it.
        // If we decide to do it, do not throw, warn, since a lot of exchanges do not respect this.
        if(loc.regionCode.empty())
            loc.regionCode = geo.region;
    }

    if(!geo.regionfips104.empty()) {
        // TODO
        // Maybe create a map / flat file with the valid fips 10- codes and test against it.
        if (loc.regionCode.empty())
            loc.regionCode = geo.regionfips104;
    }

    if(!geo.city.empty()) {
        // TODO Use UN Code for trade and transport location.. so extractAscii
        // http://www.unece.org/cefact/locode/service/location.html
        if(loc.cityName.empty())
            loc.cityName = geo.city;
    }

    // Zip code into location.
    loc.postalCode = geo.zip;

    if(!geo.metro.empty()) {
        // TODO Metro code
        // https://developers.google.com/adwords/api/docs/appendix/cities-DMAregions?csw=1
        if(loc.metro != 0)
            loc.metro = boost::lexical_cast<int> (geo.metro);
    }

}

void
OpenRTBBidRequestParser::
onUser(OpenRTB::User & user) {

    // User ID is the id on the exchange
    if(user.id)
        ctx.br->userIds.add(user.id, ID_EXCHANGE);

    // We need provider ID at the least
    // 1) Use buyer uid
    if(user.buyeruid)
        ctx.br->userIds.add(user.buyeruid, ID_PROVIDER);
    // 2) Use the exchange id if the provider doesn't use an id
    else if(user.id)
        ctx.br->userIds.add(user.id, ID_PROVIDER);
    // 3) At the BR level, we will validate we have a user ID.
    //    If not, we will use ip/ua hash
    //    Done onDevice()
    
    /*if(user.yob.val != -1) {
        // Nothing for now
    }*/

    if(!user.gender.empty()){
        if(user.gender.size() == 1) {

            // TODO Validate if this valid behaviour
            // If we receive m, f or o, toUpper()
            user.gender = std::toupper(user.gender[0]);

            if(user.gender.compare("M") == 0) {
            
            } else if(user.gender.compare("F") == 0) {
            } else if(user.gender.compare("O") == 0) {
            } else {
                LOG(OpenRTBBidRequestLogs::trace) << " br.user.gender : " << user.gender <<  
                                                     "is invalid. It should be either 'M' 'F' 'O' or null/empty" << endl;
            }
        } else {
            // TODO Validate if we accept "unknown" or "UNKNOWN" as gender.
            // According to the spec, unknown should be null (or empty).
            if(!(user.gender.compare("unknown") || user.gender.compare("UNKNOWN")))
                // Invalid gender
                LOG(OpenRTBBidRequestLogs::trace) << " br.user.gender : " << user.gender << 
                                                     "is invalid. It should be either 'M' 'F' 'O' or null/empty" << endl;
        }

    } else {
        // Gender unknown.. According to the spec leave it null.
    }

    // Data object
    if(!user.data.empty())
        for(auto & d : user.data) {
            this->onData(d);
        }

    // Rubicon related information .. Why is this in OpenRTB?
    // TODO Create a parser / converter for rubicon that overrides onUser
    // which probably means also creating an object definition that extends openrtb.h
    if(user.tz.val != -1)
        ctx.br->location.timezoneOffsetMinutes = user.tz.val;
}

void
OpenRTBBidRequestParser::
onData(OpenRTB::Data & data) {

    std::string key;
    if(data.id)
        key = data.id.toString();
    else
        key = data.name.extractAscii();

    std::vector<std::string> values;

    for (auto & s : data.segment) {
        if(s.id)
            values.push_back(s.id.toString());
        else if(!s.name.empty())
            values.push_back(s.name.extractAscii());
        else
            // Values with no name
            values.push_back("unknown_segment");
    }
    
    ctx.br->segments.addStrings(key, values);
}

void 
OpenRTBBidRequestParser::
onSegment(OpenRTB::Segment & segment) {
    LOG(OpenRTBBidRequestLogs::error) << "onSegment : Not used / Not implemented." << endl;
}

void
OpenRTBBidRequestParser2point1::
onDevice(OpenRTB::Device& device) {
    if(device.devicetype.val > 3) {
        LOG(OpenRTBBidRequestLogs::error) << "Device Type : " << device.devicetype.val << " not supported in OpenRTB 2.1." << endl;
    }

    // Call base version
    OpenRTBBidRequestParser::onDevice(device);
}


void
OpenRTBBidRequestParser2point2::
onBidRequest(OpenRTB::BidRequest & br) {

    // Call V1
    OpenRTBBidRequestParser::onBidRequest(br);

    if(ctx.br->regs)
        this->onRegulations(*br.regs);
}

void
OpenRTBBidRequestParser2point2::
onImpression(OpenRTB::Impression & imp) {

    // Deal with secure, business logic
    
    // Deal with PMP
    if(imp.pmp) {
        this->onPMP(*imp.pmp);
    }

    // Call V1
    OpenRTBBidRequestParser::onImpression(imp);
    
}

void
OpenRTBBidRequestParser2point2::
onBanner(OpenRTB::Banner & banner) {

    // Business logic around w/h min/max
    
    // Call V1
    OpenRTBBidRequestParser::onBanner(banner);
}

void
OpenRTBBidRequestParser2point2::
onVideo(OpenRTB::Video & video) {

    if(video.mimes.empty()) {
        THROW(OpenRTBBidRequestLogs::error22) << "br.imp.video.mimes needs to be populated." << endl;
    }

    // -1 being the default value
    if(video.protocol.value() != -1 && (video.protocol.value() < 0 || video.protocol.value() > 6)) {
        LOG(OpenRTBBidRequestLogs::error22) << video.protocol.value() << endl;
        THROW(OpenRTBBidRequestLogs::error22) << "br.imp.video.protocol if specified must match a value in OpenRTB 2.2 Table 6.7." << endl;
    }

    if(video.minduration.val < 0 ) {
        THROW(OpenRTBBidRequestLogs::error22) << "br.imp.video.minduration must be specified and positive." << endl;
    }

    if(video.maxduration.val < 0 ) {
        THROW(OpenRTBBidRequestLogs::error22) << "br.imp.video.maxduration must be specified and positive." << endl;
    } else if (video.maxduration.val < video.minduration.val) {
        // Illogical
        THROW(OpenRTBBidRequestLogs::error22) << "br.imp.video.maxduration can't be smaller than br.imp.video.minduration." << endl;
    } 

    ctx.spot.position = video.pos;

    // Add api to the segments in order to filter on it
    for(auto & api : video.api) {
        auto framework = apiFrameworks.find(api.val);
        if (framework != apiFrameworks.end())
            ctx.br->segments.add("api-video", framework->second, 1.0);
    }
    ctx.spot.formats.push_back(Format(video.w.value(), video.h.value()));

}

void
OpenRTBBidRequestParser2point2::
onDevice(OpenRTB::Device & device) {

    if(device.devicetype.val > 7)
        LOG(OpenRTBBidRequestLogs::error22) << "Device Type : " << device.devicetype.val << " not supported in OpenRTB 2.2." << endl;

    // Call base version
    OpenRTBBidRequestParser::onDevice(device);
}

void
OpenRTBBidRequestParser2point2::
onRegulations(OpenRTB::Regulations & regs) {

}

void
OpenRTBBidRequestParser2point2::
onDeal(OpenRTB::Deal & deal) {

}

void
OpenRTBBidRequestParser2point2::
onPMP(OpenRTB::PMP & pmp) {
}

namespace {

struct AtInit {
    AtInit()
    {
        auto parser = [](const std::string& request) {
            auto parser = OpenRTBBidRequestParser::openRTBBidRequestParserFactory(DefaultVersion);
            return parser->parseBidRequest(request, "", "");
        };
        PluginInterface<BidRequest>::registerPlugin("openrtb", parser);
    }
} atInit;

}

} // namespace RTBKIT
