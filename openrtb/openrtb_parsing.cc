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

    addField("id", &BidRequest::id, "Bid Request ID");
    addField("imp", &BidRequest::imp, "Impressions");
    //addField("context", &BidRequest::context, "Context of bid request");
    addField("site", &BidRequest::site, "Information about site the request is on");
    addField("app", &BidRequest::app, "Information about the app the request is being shown in");
    addField("device", &BidRequest::device, "Information about the device on which the request was made");
    addField("user", &BidRequest::user, "Information about the user who is making the request");
    addField("at", &BidRequest::at, "Type of auction: 1(st) or 2(nd)");
    addField("tmax", &BidRequest::tmax, "Maximum response time (ms)");
    addField("wseat", &BidRequest::wseat, "Allowable seats");
    addField("allimps", &BidRequest::allimps, "Set to 1 if all impressions on this page are in the bid request");
    addField("cur", &BidRequest::cur, "List of acceptable currencies to bid in");
    addField("bcat", &BidRequest::bcat, "Blocked advertiser content categories");
    addField("badv", &BidRequest::badv, "Blocked adversiser domains");
    addField("ext", &BidRequest::ext, "Extended fields outside of protocol");
    addField("unparseable", &BidRequest::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<Impression>::
DefaultDescription()
{
    addField("id", &Impression::id, "Impression ID within bid request");
    addField("banner", &Impression::banner, "Banner information if a banner ad");
    addField("video", &Impression::video, "Video information if a video ad");
    addField("displaymanager", &Impression::displaymanager, "Display manager that renders the ad");
    addField("displaymanagerver", &Impression::displaymanagerver, "Version of the display manager");
    addField("instl", &Impression::instl, "Is the ad interstitial");
    addField("tagid", &Impression::tagid, "Add tag ID for auction");
    addField("bidfloor", &Impression::bidfloor, "Bid floor in CPM of currency");
    addField("bidfloorcur", &Impression::bidfloorcur, "Currency for bid floor");
    addField("iframebuster", &Impression::iframebuster, "Supported iframe busters");
    addField("ext", &Impression::ext, "Extended impression attributes");
}

DefaultDescription<OpenRTB::Content>::
DefaultDescription()
{
    addField("id", &Content::id, "Unique identifier representing the content");
    addField("episode", &Content::episode, "Unique identifier representing the episode");
    addField("title", &Content::title, "Title of the content");
    addField("series", &Content::series, "Series to which the content belongs");
    addField("season", &Content::season, "Season to which the content belongs");
    addField("url", &Content::url, "URL of the content's original location");
    addField("cat", &Content::cat, "IAB content categories of the content");
    addField("videoquality", &Content::videoquality, "Quality of the video");
    ValueDescriptionT<std::string> * kwdesc = new CommaSeparatedListDescription();
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
    addField("language", &Content::language, "ISO 639-1 Content language");
    addField("ext", &Content::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Banner>::
DefaultDescription()
{
    addField<List<int>>("w", &Banner::w, "Width of ad in pixels",
             new FormatListDescription());
    addField<List<int>>("h", &Banner::h, "Height of ad in pixels",
             new FormatListDescription());
    addField("id", &Banner::id, "Ad ID");
    addField("pos", &Banner::pos, "Ad position");
    addField("btype", &Banner::btype, "Blocked creative types");
    addField("battr", &Banner::battr, "Blocked creative attributes");
    addField("mimes", &Banner::mimes, "Whitelist of content MIME types (none = all)");
    addField("topframe", &Banner::topframe, "Is it in the top frame or an iframe?");
    addField("expdir", &Banner::expdir, "Expandable ad directions");
    addField("api", &Banner::api, "Supported APIs");
    addField("ext", &Banner::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Video>::
DefaultDescription()
{
    addField("mimes", &Video::mimes, "Content MIME types supported");
    addField("linearity", &Video::linearity, "Ad linearity");
    addField("minduration", &Video::minduration, "Minimum duration in seconds");
    addField("maxduration", &Video::maxduration, "Maximum duration in seconds");
    addField("protocol", &Video::protocol, "Bid response protocols supported");
    addField("w", &Video::w, "Width of player in pixels");
    addField("h", &Video::h, "Height of player in pixels");
    addField("startdelay", &Video::startdelay, "Starting delay in seconds of video");
    addField("sequence", &Video::sequence, "Which ad number in the video");
    addField("battr", &Video::battr, "Which creative attributes are blocked");
    addField("maxextended", &Video::maxextended, "Maximum extended video ad duration");
    addField("minbitrate", &Video::minbitrate, "Minimum bitrate for ad in kbps");
    addField("maxbitrate", &Video::maxbitrate, "Maximum bitrate for ad in kbps");
    addField("boxingallowed", &Video::boxingallowed, "Is letterboxing allowed?");
    addField("playbackmethod", &Video::playbackmethod, "Available playback methods");
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
    addField("id", &Publisher::id, "Unique ID representing the publisher/producer");
    addField("name", &Publisher::name, "Publisher/producer name");
    addField("cat", &Publisher::cat, "Content categories");
    addField("domain", &Publisher::domain, "Domain name of publisher");
    addField("ext", &Publisher::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Context>::
DefaultDescription()
{
    addField("id", &Context::id, "Site or app ID on the exchange");
    addField("name", &Context::name, "Site or app name");
    addField("domain", &Context::domain, "Site or app domain");
    addField("cat", &Context::cat, "IAB content categories for the site/app");
    addField("sectioncat", &Context::sectioncat, "IAB content categories for site/app section");
    addField("pagecat", &Context::pagecat, "IAB content categories for site/app page");
    addField("privacypolicy", &Context::privacypolicy, "Site or app has a privacy policy");
    addField("publisher", &Context::publisher, "Publisher of site or app");
    addField("content", &Context::content, "Content of site or app");
    ValueDescriptionT<std::string> * kwdesc = new CommaSeparatedListDescription();
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
    addField("country", &Geo::country, "ISO 3166-1 country code");
    addField("region", &Geo::region, "ISO 3166-2 Region code");
    addField("regionfips104", &Geo::regionfips104, "FIPS 10-4 region code");
    addField("metro", &Geo::metro, "Metropolitan region (Google Metro code");
    addField("city", &Geo::city, "City name (UN Code for Trade and Transport)");
    addField("zip", &Geo::zip, "Zip or postal code");
    addField("type", &Geo::type, "Source of location data");
    addField("ext", &Geo::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Device>::
DefaultDescription()
{
    addField("dnt", &Device::dnt, "Is do not track set");
    addField("ua", &Device::ua, "User agent of device");
    addField("ip", &Device::ip, "IP address of device");
    addField("geo", &Device::geo, "Geographic location of device");
    addField("didsha1", &Device::didsha1, "SHA-1 Device ID");
    addField("didmd5", &Device::didmd5, "MD5 Device ID");
    addField("dpidsha1", &Device::dpidsha1, "SHA-1 Device Platform ID");
    addField("dpidmd5", &Device::dpidmd5, "MD5 Device Platform ID");
    addField("ipv6", &Device::ipv6, "Device IPv6 address");
    addField("carrier", &Device::carrier, "Carrier or ISP derived from IP address");
    addField("language", &Device::language, "Browser language");
    addField("make", &Device::make, "Device make");
    addField("model", &Device::model, "Device model");
    addField("os", &Device::os, "Device OS");
    addField("osv", &Device::osv, "Device OS version");
    addField("js", &Device::js, "Javascript is supported");
    addField("connectiontype", &Device::connectiontype, "Device connection type");
    addField("devicetype", &Device::devicetype, "Device type");
    addField("flashver", &Device::flashver, "Flash version on device");
    addField("ext", &Device::ext, "Extensions to device field go here");
}

DefaultDescription<OpenRTB::Segment>::
DefaultDescription()
{
    addField("id", &Segment::id, "Segment ID");
    addField("name", &Segment::name, "Segment name");
    addField("value", &Segment::value, "Segment value");
    addField("ext", &Segment::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Data>::
DefaultDescription()
{
    addField("id", &Data::id, "Segment ID");
    addField("name", &Data::name, "Segment name");
    addField("segment", &Data::segment, "Data segment");
    addField("ext", &Data::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::User>::
DefaultDescription()
{
    addField("id", &User::id, "Exchange specific user ID");
    addField("buyeruid", &User::buyeruid, "Seat specific user ID");
    addField("yob", &User::yob, "Year of birth");
    addField("gender", &User::gender, "Gender");
    ValueDescriptionT<std::string> * kwdesc = new CommaSeparatedListDescription();
    addField("keywords", &User::keywords, "Keywords about user", kwdesc);
    addField("customdata", &User::customdata, "Exchange-specific custom data");
    addField("geo", &User::geo, "Geolocation of user at registration");
    addField("data", &User::data, "User segment data");
    addField("ext", &User::ext, "Extensions to the protocol go here");
}

DefaultDescription<OpenRTB::Bid>::
DefaultDescription()
{
    addField("id", &Bid::id, "Bidder's ID to identify the bid");
    addField("impid", &Bid::impid, "ID of impression");
    addField("price", &Bid::price, "CPM price to bid for the impression");
    addField("adid", &Bid::adid, "ID of ad to be served if bid is won");
    addField("nurl", &Bid::nurl, "Win notice/ad markup URL");
    addField("adm", &Bid::adm, "Ad markup");
    addField("adomain", &Bid::adomain, "Advertiser domain(s)");
    addField("iurl", &Bid::iurl, "Image URL for content checking");
    addField("cid", &Bid::cid, "Campaign ID");
    addField("crid", &Bid::crid, "Creative ID");
    addField("attr", &Bid::attr, "Creative attributes");
    addField("ext", &Bid::ext, "Extensions");
}

DefaultDescription<OpenRTB::SeatBid>::
DefaultDescription()
{
    addField("bid", &SeatBid::bid, "Bids made for this seat");
    addField("seat", &SeatBid::seat, "Seat name who is bidding");
    addField("group", &SeatBid::group, "Do we require all bids to be won in a group?");
    addField("ext", &SeatBid::ext, "Extensions");
}

DefaultDescription<OpenRTB::BidResponse>::
DefaultDescription()
{
    addField("id", &BidResponse::id, "ID of auction");
    addField("seatbid", &BidResponse::seatbid, "Array of bids for each seat");
    addField("bidid", &BidResponse::bidid, "Bidder's internal ID for this bid");
    addField("cur", &BidResponse::cur, "Currency in which we're bidding");
    addField("customData", &BidResponse::customData, "Custom data to be stored for user");
    addField("ext", &BidResponse::ext, "Extensions");
}

} // namespace Datacratic
