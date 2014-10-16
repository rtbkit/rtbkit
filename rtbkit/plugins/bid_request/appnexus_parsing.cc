/* appnexus_parsing.cc
   Mark Weiss, 28 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Structure descriptions for AppNexus.
*/

#include "appnexus_parsing.h"
#include "soa/types/json_parsing.h"

//using namespace RTBKIT;
using namespace std;

namespace Datacratic {

DefaultDescription<AppNexus::BidRequest>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::BidRequest::unparseable);
    addField("bid_request", &AppNexus::BidRequest::bidRequest, "Bid Request");
}

DefaultDescription<AppNexus::BidRequestMsg>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::BidRequestMsg::unparseable);

    addField("member_ad_profile_id", &AppNexus::BidRequestMsg::memberAdProfileId, "Bid Request ID");
    addField("timestamp", &AppNexus::BidRequestMsg::timestamp, "Timestamp");
    addField("bidder_timeout_ms", &AppNexus::BidRequestMsg::bidderTimeoutMs, "Bidder timeout ms");
    addField("bid_info", &AppNexus::BidRequestMsg::bidInfo, "AN BidInfo object");
    addField("members", &AppNexus::BidRequestMsg::members, "Array of AN Members Json object");
    addField("tags", &AppNexus::BidRequestMsg::tags, "Array of AN Tags objects");
    addField("allow_exclusive", &AppNexus::BidRequestMsg::allowExclusive, "Allow exclusive inventory flag");
    addField("debug_requested", &AppNexus::BidRequestMsg::debugRequested, "Flag to mark this bid request for debugging");
    addField("debug_member_id", &AppNexus::BidRequestMsg::debugMemberId, "AN customer requesting debug");
    addField("test", &AppNexus::BidRequestMsg::test, "Flag marking bid request as a test call");
    addField("excluded_attributes", &AppNexus::BidRequestMsg::excludedAttributes, "Array of Ids of excluded attributes");
    addField("single_phase", &AppNexus::BidRequestMsg::singlePhase, "AN-specific flag");
    addField("unparseable", &AppNexus::BidRequestMsg::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<AppNexus::BidInfo>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::BidInfo::unparseable);

    addField("user_id_64", &AppNexus::BidInfo::userId64, "Bid request userId in AN namespace");
    addField("user_agent", &AppNexus::BidInfo::userAgent, "Bid request client user agent");
    addField("operating_system", &AppNexus::BidInfo::operatingSystem, "Bid request client user agent");
    addField("accepted_languages", &AppNexus::BidInfo::acceptedLanguages, "Allowed languages for a bid request");
    addField("language", &AppNexus::BidInfo::language, "Allowed languages for a bid request");
    addField("no_flash", &AppNexus::BidInfo::noFlash, "Flag indicating whether flash is allowed in the ad unit");
    addField("no_cookies", &AppNexus::BidInfo::noCookies, "Flag indcating whether cookies are allowed to be used in the ad unit");
    addField("gender", &AppNexus::BidInfo::gender, "Code indicating gender of user");
    addField("age", &AppNexus::BidInfo::age, "Age of the user");
    addField("segments", &AppNexus::BidInfo::segments, "AN Segment objects associated with the bid request");
    addField("ip_address", &AppNexus::BidInfo::ipAddress, "IP Address of the user");
    addField("country", &AppNexus::BidInfo::country, "Country of the user");
    addField("region", &AppNexus::BidInfo::region, "Region of the user");
    addField("city", &AppNexus::BidInfo::city, "City of the user");
    addField("postal_code", &AppNexus::BidInfo::postalCode, "Postal code of the user");
    addField("dma", &AppNexus::BidInfo::dma, "DMA of the user");
    addField("time_zone", &AppNexus::BidInfo::timeZone, "Time zone of the user");
    addField("userdata_json", &AppNexus::BidInfo::userdataJson, "AN user data from server-side cookie storage");
    addField("total_clicks", &AppNexus::BidInfo::totalClicks, "DEPRECATED");
    addField("selling_member_id", &AppNexus::BidInfo::sellingMemberId, "Server-sice cookie storage, AN seller Id");
    addField("url", &AppNexus::BidInfo::url, "Host URL of ad unit. 'page' where ad is running");
    addField("domain", &AppNexus::BidInfo::domain, "Domain of host URL of ad unit");
    addField("inventory_class", &AppNexus::BidInfo::inventoryClass, "DEPRECATED");
    addField("inventory_audits", &AppNexus::BidInfo::inventoryAudits, "Array of AN InventoryAudit objects");
    addField("within_iframe", &AppNexus::BidInfo::withinIframe, "Flag indicating whether Address unit is contained in iFrame");
    addField("publisher_id", &AppNexus::BidInfo::publisherId, "AN id of publisher of page that is host URL of ad unit");
    addField("is_secure", &AppNexus::BidInfo::isSecure, "Flag indicating whether protocol of ad call is secure (https)");
    addField("app_id", &AppNexus::BidInfo::appId, "Mobile-specific application Id");
    addField("loc", &AppNexus::BidInfo::loc, "lat / long in format: 'snnn.ddddddddddddd,snnn.ddddddddddddd'. West, south negative. Up to 13 places precision.");
    addField("carrier", &AppNexus::BidInfo::carrier, "Carrier code");
    addField("make", &AppNexus::BidInfo::make, "Make code");
    addField("model", &AppNexus::BidInfo::model, "Model code");
    addField("unparseable", &AppNexus::BidInfo::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<AppNexus::Segment>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::Segment::unparseable);

    addField("id", &AppNexus::Segment::id, "AN segment Id");
    addField("member_id", &AppNexus::Segment::memberId, "AN member that owns the segment");
    addField("code", &AppNexus::Segment::code, "Textual description of the segment");
    addField("last_seen_min", &AppNexus::Segment::lastSeenMin, "Timestamp indicating last time a bid request with the segment was seen");
    addField("unparseable", &AppNexus::Segment::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<AppNexus::InventoryAudit>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::InventoryAudit::unparseable);

    addField("auditor_member_id", &AppNexus::InventoryAudit::auditorMemberId, "AN member Id");
    addField("intended_audience", &AppNexus::InventoryAudit::intendedAudience, "AN enum set of audience textual codes");
    addField("inventory_attributes", &AppNexus::InventoryAudit::inventoryAttributes, "Integer codes from AN inventory attribute service");
    addField("content_categories", &AppNexus::InventoryAudit::contentCategories, "Conent codes from AN content attribute service");
    addField("unparseable", &AppNexus::InventoryAudit::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<AppNexus::Tag>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::Tag::unparseable);

    addField("auction_id_64", &AppNexus::Tag::auctionId64, "AN auction Id");
    addField("id", &AppNexus::Tag::id, "TinyTag Id. TinyTag is AN mechanism for user to label partitions of inventory.");
    addField("site_id", &AppNexus::Tag::siteId, "AN Site Id. Sites are subsets of Publisher inventory.");
    addField("inventory_source_id", &AppNexus::Tag::inventorySourceId, "Inventory source Id. May be hidden by AN.");
    addField("size", &AppNexus::Tag::size, "IAB allowed size for ad unit");
    addField("sizes", &AppNexus::Tag::sizes, "IAB allowed sizes for ad unit");
    addField("position", &AppNexus::Tag::position, "enum: {'below','above','unknown'}");
    addField("tag_format", &AppNexus::Tag::tagFormat, "enum: {'iframe','javascript'}");
    addField("allowed_media_types", &AppNexus::Tag::allowedMediaTypes, "Array of AN integer codes");
    addField("allowed_media_subtypes", &AppNexus::Tag::allowedMediaSubtypes, "Array of AN integer codes");
    addField("media_subtypes", &AppNexus::Tag::mediaSubtypes, "Array of AN string codes");
    addField("inventory_audits", &AppNexus::Tag::inventoryAudits, "Array of Inventory Audit objects");
    addField("venue_id", &AppNexus::Tag::venueId, "Not used");
    addField("ad_profile_id", &AppNexus::Tag::adProfileId, "Tag-level ad approval profile ID");
    addField("reserve_price", &AppNexus::Tag::reservePrice, "Seller reserve price");
    addField("estimated_clear_price", &AppNexus::Tag::estimatedClearPrice, "AN estimated clearing price for ad");
    addField("estimated_average_price", &AppNexus::Tag::estimatedAveragePrice, "AN estimated average price for ad");
    addField("estimated_price_verified", &AppNexus::Tag::estimatedPriceVerified, "Flag indicating estimated price has been verified");
    addField("tag_data", &AppNexus::Tag::tagData, "Additional buyer data related to the TinyTag label");
    addField("exclusive_default", &AppNexus::Tag::exclusiveDefault, "Flag indicating there is an exclusive default for this tag");
    addField("default_creative_id", &AppNexus::Tag::defaultCreativeId, "Creative Id of default creative if there is one");
    addField("supply_type", &AppNexus::Tag::supplyType, "AN docs not totally clear but seems to be an enum of {'web', 'mobile_browser', 'mobile_app'}");
    addField("creative_formats", &AppNexus::Tag::creativeFormats, "enum of {'text', 'image', 'url-html', 'url-js', 'flash', 'raw-js', 'raw-html', 'iframe-html', 'urlvast'}");
    addField("creative_actions", &AppNexus::Tag::creativeActions, "enum of {'click-to-web', 'click-to-call'}");
    addField("smaller_sizes_allowed", &AppNexus::Tag::smallerSizesAllowed, "Flag indicating smaller creative sizes than values given in 'size' or 'sizes' are allowed in ad unit");
    addField("unparseable", &AppNexus::Tag::unparseable, "Unparseable fields are collected here");
}

DefaultDescription<AppNexus::Member>::
DefaultDescription()
{
    collectUnparseableJson(&AppNexus::Member::unparseable);

    addField("id", &AppNexus::Member::id, "AN member Id");
    addField("unparseable", &AppNexus::Member::unparseable, "Unparseable fields are collected here");
}


} // namespace Datacratic

