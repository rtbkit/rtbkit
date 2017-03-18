/* openrtb_parsing.cc
   Jeremy Barnes, 22 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Structure descriptions for OpenRTB.
*/

#include "openrtb_parsing.h"
#include "soa/types/json_parsing.h"

using namespace OpenRTB;
//using namespace RTBKIT;
using namespace std;

namespace Datacratic {

DefaultDescription<BidRequest>::
DefaultDescription()
{
    onUnknownField = [=] (BidRequest * br, JsonParsingContext & context)
        {
            //cerr << "got unknown field " << context.printPath() << endl;

            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].fieldName()]);
            };

            getEntry(0, br->unparseable)
                = context.expectJson();
        };

    addField("id", &BidRequest::id, "Bid Request ID",
             new StringIdDescription());
    addField("imp", &BidRequest::imp, "Impressions");
    //addField("context", &BidRequest::context, "Context of bid request");
    addField("site", &BidRequest::site, "Information about site the request is on");
    addField("app", &BidRequest::app, "Information about the app the request is being shown in");
    addField("device", &BidRequest::device, "Information about the device on which the request was made");
    addField("user", &BidRequest::user, "Information about the user who is making the request");
    addField("test", &BidRequest::test, "Flag for testing");
    addField("at", &BidRequest::at, "Type of auction: 1(st) or 2(nd)");
    addField("tmax", &BidRequest::tmax, "Maximum response time (ms)");
    addField("wseat", &BidRequest::wseat, "Allowable seats");
    addField("bseat", &BidRequest::bseat, "Banned seats");
    addField("allimps", &BidRequest::allimps, "Set to 1 if all impressions on this page are in the bid request");
    addField("cur", &BidRequest::cur, "List of acceptable currencies to bid in");
    addField("wlang", &BidRequest::wlang, "White list of languages for creatives.");
    addField("bcat", &BidRequest::bcat, "Blocked advertiser content categories");
    addField("badv", &BidRequest::badv, "Blocked adversiser domains");
    addField("bapp", &BidRequest::bapp, "Block list of applications by their platform-specific exchange- independent application identifiers.");
    addField("source", &BidRequest::source, "Data about inventory source.");
    addField("regs", &BidRequest::regs, "Legal regulations");
    addField("ext", &BidRequest::ext, "Extended fields outside of protocol");
    addField("unparseable", &BidRequest::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<Impression>::
DefaultDescription()
{
    addField("id", &Impression::id, "Impression ID within bid request",
             new StringIdDescription());
    addField("metric", &Impression::metric, "An array of metric objects");
    addField("audio", &Impression::audio, "Audio information if an audio ad");
    addField("native", &Impression::native, "Native information if a native ad");
    addField("banner", &Impression::banner, "Banner information if a banner ad");
    addField("video", &Impression::video, "Video information if a video ad");
    addField("displaymanager", &Impression::displaymanager, "Display manager that renders the ad");
    addField("displaymanagerver", &Impression::displaymanagerver, "Version of the display manager");
    addField("instl", &Impression::instl, "Is the ad interstitial");
    addField("tagid", &Impression::tagid, "Add tag ID for auction");
    addField("bidfloor", &Impression::bidfloor, "Bid floor in CPM of currency");
    addField("bidfloorcur", &Impression::bidfloorcur, "Currency for bid floor");
    addField("clickbrowser", &Impression::clickbrowser, "Indicates the type of browser opened upon clicking the creative in an app, where 0 = embedded, 1 = native.");
    addField("secure", &Impression::secure, "Does the impression require https");
    addField("iframebuster", &Impression::iframebuster, "Supported iframe busters");
    addField("pmp", &Impression::pmp, "Contains any deals eligible for the impression");
    addField("ext", &Impression::ext, "Extended impression attributes");
}

DefaultDescription<OpenRTB::Content>::
DefaultDescription()
{
    addField("id", &Content::id, "Unique identifier representing the content",
             new StringIdDescription());
    addField("episode", &Content::episode, "Unique identifier representing the episode");
    addField("title", &Content::title, "Title of the content");
    addField("series", &Content::series, "Series to which the content belongs");
    addField("season", &Content::season, "Season to which the content belongs");
    addField("artist", &Content::artist, "Artist credited with the content.");
    addField("genre", &Content::genre, "Genre that best describes the content (e.g., rock, pop, etc).");
    addField("album", &Content::album, "Album to which the content belongs; typically for audio.");
    addField("isrc", &Content::isrc, "International Standard Recording Code conforming to ISO- 3901.");
    addField("url", &Content::url, "URL of the content's original location");
    addField("cat", &Content::cat, "IAB content categories of the content");
    addField("prodq", &Content::prodq, "Production quality.");
    addField("videoquality", &Content::videoquality, "Quality of the video");
    ValueDescriptionT<CSList> * kwdesc = new Utf8CommaSeparatedListDescription();
    addField("keywords", &Content::keywords, "Keywords describing the keywords", kwdesc);
    addField("contentrating", &Content::contentrating, "Rating of the content");
    addField("userrating", &Content::userrating, "User-provided rating of the content");
    addField("context", &Content::context, "Rating context");
    addField("livestream", &Content::livestream, "Is this being live streamed?");
    addField("sourcerelationship", &Content::sourcerelationship, "Relationship with content source");
    addField("producer", &Content::producer, "Content producer");
    addField("len", &Content::len, "Content length in seconds");
    addField("qagmediarating", &Content::qagmediarating, "Media rating per QAG guidelines");
    addField("embeddable", &Content::embeddable, "1 if embeddable, 0 otherwise");
    addField("data", &Content::data, "Additional content data.");
    addField("language", &Content::language, "ISO 639-1 Content language");
    addField("ext", &Content::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Native>::
DefaultDescription()
{
    addField("request", &Native::request, "Request payload complying with the Native Ad Specification.");
    addField("ver", &Native::ver, "Version of the Native Ad Specification to which request complies.");
    addField("api", &Native::api, "List of supported API frameworks for this impression.");
    addField("battr", &Native::battr, "Blocked creative attributes.");
    addField("ext", &Native::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Banner>::
DefaultDescription()
{
    addField<List<int>>("w", &Banner::w, "Width of ad in pixels",
             new FormatListDescription());
    addField<List<int>>("h", &Banner::h, "Height of ad in pixels",
             new FormatListDescription());
    addField("hmin", &Banner::hmin, "Ad minimum height");
    addField("hmax", &Banner::hmax, "Ad maximum height");
    addField("wmin", &Banner::wmin, "Ad minimum width");
    addField("wmax", &Banner::wmax, "Ad maximum width");
    addField("id", &Banner::id, "Ad ID", new StringIdDescription());
    addField("pos", &Banner::pos, "Ad position");
    addField("btype", &Banner::btype, "Blocked creative types");
    addField("battr", &Banner::battr, "Blocked creative attributes");
    addField("mimes", &Banner::mimes, "Whitelist of content MIME types (none = all)");
    addField("topframe", &Banner::topframe, "Is it in the top frame or an iframe?");
    addField("expdir", &Banner::expdir, "Expandable ad directions");
    addField("api", &Banner::api, "Supported APIs");
    addField("vcm", &Banner::vcm, "Relevant only for Banner objects used with a Video object.");
    addField("ext", &Banner::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Video>::
DefaultDescription()
{
    addField("mimes", &Video::mimes, "Content MIME types supported");
    addField("linearity", &Video::linearity, "Ad linearity");
    addField("minduration", &Video::minduration, "Minimum duration in seconds");
    addField("maxduration", &Video::maxduration, "Maximum duration in seconds");
    addField("protocol", &Video::protocol, "Bid response supported protocol");
    addField("protocols", &Video::protocols, "Bid response supported protocols");
    addField("w", &Video::w, "Width of player in pixels");
    addField("h", &Video::h, "Height of player in pixels");
    addField("startdelay", &Video::startdelay, "Starting delay in seconds of video");
    addField("placement", &Video::placement, "Placement type for the impression.");
    addField("sequence", &Video::sequence, "Which ad number in the video");
    addField("battr", &Video::battr, "Which creative attributes are blocked");
    addField("maxextended", &Video::maxextended, "Maximum extended video ad duration");
    addField("minbitrate", &Video::minbitrate, "Minimum bitrate for ad in kbps");
    addField("maxbitrate", &Video::maxbitrate, "Maximum bitrate for ad in kbps");
    addField("boxingallowed", &Video::boxingallowed, "Is letterboxing allowed?");
    addField("playbackmethod", &Video::playbackmethod, "Available playback methods");
    addField("playbackend", &Video::playbackend, "The event that causes playback to end");
    addField("delivery", &Video::delivery, "Available delivery methods");
    addField("pos", &Video::pos, "Ad position");
    addField("companionad", &Video::companionad, "List of companion banners available");
    addField("api", &Video::api, "List of supported API frameworks");
    addField("companiontype", &Video::companiontype, "List of VAST companion types");
    addField("ext", &Video::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Publisher>::
DefaultDescription()
{
    addField("id", &Publisher::id, "Unique ID representing the publisher/producer",
            new StringIdDescription());
    addField("name", &Publisher::name, "Publisher/producer name");
    addField("cat", &Publisher::cat, "Content categories");
    addField("domain", &Publisher::domain, "Domain name of publisher");
    addField("ext", &Publisher::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Context>::
DefaultDescription()
{
    addField("id", &Context::id, "Site or app ID on the exchange",
            new StringIdDescription());
    addField("name", &Context::name, "Site or app name");
    addField("domain", &Context::domain, "Site or app domain");
    addField("cat", &Context::cat, "IAB content categories for the site/app");
    addField("sectioncat", &Context::sectioncat, "IAB content categories for site/app section");
    addField("pagecat", &Context::pagecat, "IAB content categories for site/app page");
    addField("privacypolicy", &Context::privacypolicy, "Site or app has a privacy policy");
    addField("publisher", &Context::publisher, "Publisher of site or app");
    addField("content", &Context::content, "Content of site or app");
    ValueDescriptionT<CSList> * kwdesc = new Utf8CommaSeparatedListDescription();
    addField("keywords", &Context::keywords, "Keywords describing siter or app", kwdesc);
    addField("ext", &Context::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Site>::
DefaultDescription()
{
    addParent<OpenRTB::Context>();

    //addField("id",   &Context::id,   "Site ID");
    addField("page",   &SiteInfo::page,   "URL of the page");
    addField("ref",    &SiteInfo::ref,    "Referrer URL to the page");
    addField("search", &SiteInfo::search, "Search string to page");
}

DefaultDescription<OpenRTB::App>::
DefaultDescription()
{
    addParent<OpenRTB::Context>();

    addField("ver",    &AppInfo::ver,     "Application version");
    addField("bundle", &AppInfo::bundle,  "Application bundle name");
    addField("paid",   &AppInfo::paid,    "Is a paid version of the app");
    addField("storeurl", &AppInfo::storeurl, "App store url");
}

DefaultDescription<OpenRTB::Geo>::
DefaultDescription()
{
    addField("lat", &Geo::lat, "Latiture of user in degrees from equator");
    addField("lon", &Geo::lon, "Longtitude of user in degrees (-180 to 180)");
    addField("type", &Geo::type, "Source of location data");
    addField("accuracy", &Geo::accuracy, "Estimated location accuracy in meters.");
    addField("lastfix", &Geo::lastfix, "Number of seconds since this geolocation fix was established.");
    addField("ipservice", &Geo::ipservice, "Service or provider used to determine geolocation from IP address if applicable.");
    addField("country", &Geo::country, "ISO 3166-1 country code");
    addField("region", &Geo::region, "ISO 3166-2 Region code");
    addField("regionfips104", &Geo::regionfips104, "FIPS 10-4 region code");
    addField("metro", &Geo::metro, "Metropolitan region (Google Metro code");
    addField("city", &Geo::city, "City name (UN Code for Trade and Transport)");
    addField("zip", &Geo::zip, "Zip or postal code");
    addField("utcoffset", &Geo::utcoffset, "UTC offset");
    addField("ext", &Geo::ext, "Extensions to the protocol go here");
    /// Datacratic extension
    addField("dma", &Geo::dma, "DMA code");
    /// Rubicon extension
    addField("latlonconsent", &Geo::latlonconsent, "User has given consent for lat/lon information to be used");
}

DefaultDescription<OpenRTB::Device>::
DefaultDescription()
{
    addField("dnt", &Device::dnt, "Is do not track set");
    addField("ua", &Device::ua, "User agent of device");
    addField("ip", &Device::ip, "IP address of device");
    addField("lmt", &Device::lmt, "limit");
    addField("geo", &Device::geo, "Geographic location of device");
    addField("didsha1", &Device::didsha1, "SHA-1 Device ID");
    addField("didmd5", &Device::didmd5, "MD5 Device ID");
    addField("dpidsha1", &Device::dpidsha1, "SHA-1 Device Platform ID");
    addField("dpidmd5", &Device::dpidmd5, "MD5 Device Platform ID");
    addField("macsha1", &Device::macsha1, "SHA-1 Mac Address");
    addField("macmd5", &Device::macmd5, "MD5 Mac Address");
    addField("ipv6", &Device::ipv6, "Device IPv6 address");
    addField("carrier", &Device::carrier, "Carrier or ISP derived from IP address");
    addField("mccmnc", &Device::mccmnc, "Mobile carrier as the concatenated MCC-MNC code.");
    addField("language", &Device::language, "Browser language");
    addField("make", &Device::make, "Device make");
    addField("model", &Device::model, "Device model");
    addField("os", &Device::os, "Device OS");
    addField("osv", &Device::osv, "Device OS version");
    addField("hwv", &Device::hwv, "Hardware version of the device");
    addField("h", &Device::h, "Physical height of the screen in pixels.");
    addField("w", &Device::w, "Physical width of the screen in pixels.");
    addField("ppi", &Device::ppi, "Screen size as pixels per linear inch.");
    addField("pxratio", &Device::pxratio, "The ratio of physical pixels to device independent pixels");
    addField("js", &Device::js, "Javascript is supported");
    addField("geofetch", &Device::geofetch, "Indicates if the geolocation API will be available to JavaScript\n"
            "code running in the banner, where 0 = no, 1 = yes.");
    addField("connectiontype", &Device::connectiontype, "Device connection type");
    addField("devicetype", &Device::devicetype, "Device type");
    addField("flashver", &Device::flashver, "Flash version on device");
    addField("ifa", &Device::ifa, "Native identifier for advertisers");
    addField("ext", &Device::ext, "Extensions to device field go here");
}

DefaultDescription<OpenRTB::Segment>::
DefaultDescription()
{
    addField("id", &Segment::id, "Segment ID", new StringIdDescription());
    addField("name", &Segment::name, "Segment name");
    addField("value", &Segment::value, "Segment value");
    addField("ext", &Segment::ext, "Extensions to the protocol go here");
    /// Datacratic extension
    addField("segmentusecost", &Segment::segmentusecost, "Segment use cost in CPM");
}

DefaultDescription<OpenRTB::Data>::
DefaultDescription()
{
    addField("id", &Data::id, "Segment ID", new StringIdDescription());
    addField("name", &Data::name, "Segment name");
    addField("segment", &Data::segment, "Data segment");
    addField("ext", &Data::ext, "Extensions to the protocol go here");
    /// Datacratic extension
    addField("datausecost", &Data::datausecost, "Cost of using data in CPM");
    addField("usecostcurrency", &Data::usecostcurrency, "Currency for use cost");
}

DefaultDescription<OpenRTB::User>::
DefaultDescription()
{
    addField("id", &User::id, "Exchange specific user ID", new StringIdDescription());
    addField("buyeruid", &User::buyeruid, "Seat specific user ID",
            new StringIdDescription());
    addField("yob", &User::yob, "Year of birth");
    addField("gender", &User::gender, "Gender");
    ValueDescriptionT<CSList> * kwdesc = new Utf8CommaSeparatedListDescription();
    addField("keywords", &User::keywords, "Keywords about user", kwdesc);
    addField("customdata", &User::customdata, "Exchange-specific custom data");
    addField("geo", &User::geo, "Geolocation of user at registration");
    addField("data", &User::data, "User segment data");
    addField("ext", &User::ext, "Extensions to the protocol go here");
    /// Rubicon extension
    addField("tz", &User::tz, "Timezone offset of user in seconds wrt GMT");
    addField("sessiondepth", &User::sessiondepth, "Session depth of user in visits");
}

DefaultDescription<OpenRTB::Bid>::
DefaultDescription()
{
    addField("id", &Bid::id, "Bidder's ID to identify the bid",
             new StringIdDescription());
    addField("impid", &Bid::impid, "ID of impression",
             new StringIdDescription());
    addField("price", &Bid::price, "CPM price to bid for the impression");
    addField("adid", &Bid::adid, "ID of ad to be served if bid is won",
             new StringIdDescription());
    addField("nurl", &Bid::nurl, "Win notice/ad markup URL");
    addField("burl", &Bid::burl, "Billing URL");
    addField("lurl", &Bid::lurl, "Loss URL");
    addField("adm", &Bid::adm, "Ad markup");
    addField("adomain", &Bid::adomain, "Advertiser domain(s)");
    addField("bundle", &Bid::bundle, "Bundle");
    addField("iurl", &Bid::iurl, "Image URL for content checking");
    addField("cid", &Bid::cid, "Campaign ID",
             new StringIdDescription());
    addField("crid", &Bid::crid, "Creative ID",
             new StringIdDescription());
    addField("cat", &Bid::cat, "category");
    addField("attr", &Bid::attr, "Creative attributes");
    addField("api", &Bid::api, "API required by the markup if applicable.");
    addField("protocol", &Bid::protocol, "Video response protocol of the markup if applicable.");
    addField("qagmediarating", &Bid::qagmediarating, "Creative media rating per IQG guidelines");
    addField("language", &Bid::language, "Language of the creative");
    addField("dealid", &Bid::dealid, "Deal Id for PMP Auction");
    addField("w", &Bid::w, "width of ad in pixels");
    addField("h", &Bid::h, "height of ad in pixels");
    addField("wratio", &Bid::wratio, "Relative width of the creative when expressing size as a ratio.");
    addField("hratio", &Bid::hratio, "Relative height of the creative when expressing size as a ratio");
    addField("exp", &Bid::exp, "Advisory as to the number of seconds the bidder is willing to wait between the auction and the actual impression.");
    addField("ext", &Bid::ext, "Extensions");
}

DefaultDescription<OpenRTB::SeatBid>::
DefaultDescription()
{
    addField("bid", &SeatBid::bid, "Bids made for this seat");
    addField("seat", &SeatBid::seat, "Seat name who is bidding",
             new StringIdDescription());
    addField("group", &SeatBid::group, "Do we require all bids to be won in a group?");
    addField("ext", &SeatBid::ext, "Extensions");
}

DefaultDescription<OpenRTB::BidResponse>::
DefaultDescription()
{
    addField("id", &BidResponse::id, "ID of auction",
             new StringIdDescription());
    addField("seatbid", &BidResponse::seatbid, "Array of bids for each seat");
    addField("bidid", &BidResponse::bidid, "Bidder's internal ID for this bid",
             new StringIdDescription());
    addField("cur", &BidResponse::cur, "Currency in which we're bidding");
    addField("customData", &BidResponse::customData, "Custom data to be stored for user");
    addField("ext", &BidResponse::ext, "Extensions");
}

DefaultDescription<OpenRTB::Deal>::
DefaultDescription()
{
    addField("id", &Deal::id, "Id of the deal", new StringIdDescription);
    addField("bidfloor", &Deal::bidfloor, "bid floor");
    addField("bidfloorcur", &Deal::bidfloorcur, "Currency of the deal");
    addField("wseat", &Deal::wseat, "List of buyer seats allowed");
    addField("wadomain", &Deal::wadomain, "List of advertiser domains allowed");
    addField("at", &Deal::at, "Auction type");
    addField("ext", &Deal::ext, "Extensions");
}

DefaultDescription<OpenRTB::PMP>::
DefaultDescription()
{
    addField("private_auction", &PMP::privateAuction, "is a private auction");
    addField("deals", &PMP::deals, "Deals");
    addField("ext", &PMP::ext, "Extensions");
}

DefaultDescription<OpenRTB::Regulations>::
DefaultDescription()
{
    addField("coppa", &Regulations::coppa, "is coppa regulated traffic");
    addField("ext", &Regulations::ext, "Extensions");
}

DefaultDescription<OpenRTB::Format>::
DefaultDescription()
{
    addField("w", &Format::w, "Width");
    addField("h", &Format::h, "Height");
    addField("wratio", &Format::wratio, "Relative width when expressing size as a ratio.");
    addField("hratio", &Format::hratio, "Relative height when expressing size as a ratio.");
    addField("wmin", &Format::wmin, "The minimum width in device independent pixels.");
    addField("ext", &Format::ext, "Extensions");
}

DefaultDescription<OpenRTB::Audio>::
DefaultDescription()
{
    addField("mimes", &Audio::mimes, "Content MIME types supported.");
    addField("minduration", &Audio::minduration, "Minimum audio ad duration in seconds.");
    addField("maxduration", &Audio::maxduration, "Maximum audio ad duration in seconds.");
    addField("protocols", &Audio::protocols, "Array of supported audio protocols.");
    addField("startdelay", &Audio::startdelay, "Indicates the start delay in seconds for pre-roll, mid-roll, or post-roll ad placements.");
    addField("sequence", &Audio::sequence, "sequence number will allow for the coordinated delivery of multiple creatives.");
    addField("battr", &Audio::battr, "Blocked creative attributes.");
    addField("maxextended", &Audio::maxextended, "Maximum extended ad duration if extension is allowed.");
    addField("minbitrate", &Audio::minbitrate, "Minimum bit rate in Kbps.");
    addField("maxbitrate", &Audio::maxbitrate, "Maximum bit rate in Kbps.");
    addField("delivery", &Audio::delivery, "upported delivery methods.");
    addField("companionad", &Audio::companionad, "Array of Banner objects.");
    addField("api", &Audio::api, "List of supported API frameworks for this impression.");
    addField("companiontype", &Audio::companiontype, "Supported DAAST companion ad types.");
    addField("maxseq", &Audio::maxseq, "The maximum number of ads that can be played in an ad pod.");
    addField("feed", &Audio::feed, "Type of audio feed.");
    addField("stitched", &Audio::stitched, "Indicates if the audio is stitched, where 0 = no, 1 = yes.");
    addField("nvol", &Audio::nvol, "Volume normalization mode.");
    addField("dl", &Audio::dl, "Indicates if the audio file can be downloaded by the user.");
    addField("ext", &Audio::ext, "Extensions");
}

DefaultDescription<OpenRTB::Source>::
DefaultDescription()
{
    addField("fd", &Source::fd, "Entity responsible for the final impression sale decision.");
    addField("tid", &Source::tid, "Transaction ID.");
    addField("pchain", &Source::pchain, "Payment ID chain string.");
    addField("ext", &Source::ext, "Extensions");
}

DefaultDescription<OpenRTB::Metric>::
DefaultDescription()
{
    addField("type", &Metric::type, "Type of metric being presented.");
    addField("value", &Metric::value, "Number representing the value of the metric. 0.0-1.0");
    addField("vendor", &Metric::vendor, "Source of the value using exchange curated string names.");
    addField("ext", &Metric::ext, "Extensions");
}

} // namespace Datacratic
