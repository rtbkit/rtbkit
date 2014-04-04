/** appnexus.h                                                      -*- C++ -*-
    Mark Weiss, 8 May 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

    This file is part of RTBkit.

    Structs that map to the AppNexus JSON bid request data format.
*/

#pragma once

#include "soa/types/basic_value_descriptions.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/id.h"
#include "soa/types/string.h"
#include "soa/types/url.h"
#include "rtbkit/openrtb/openrtb.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace AppNexus {

struct AdPosition: public Datacratic::TaggedEnum<AdPosition, 0> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN = 0,
        ABOVE = 1,
        BELOW = 3
    };
};

struct Segment {
    Datacratic::Id id;
    Datacratic::Id memberId;
    std::string code;                // TODO Get values from "Segment Pixels" section in AN documentation
    Datacratic::TaggedInt lastSeenMin;      // TODO POSIX valid int ts
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct InventoryAudit {
    Datacratic::Id auditorMemberId;
    std::string intendedAudience;    // TODO enum, values in intendedAudences
    std::vector<Datacratic::TaggedInt> inventoryAttributes;  // TODO no enum values in spec
    std::vector<Datacratic::TaggedInt> contentCategories;    // TODO no enum values in spec
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct Member {
    Datacratic::Id id;
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct Tag {
    // Subsection: Auction data
    Datacratic::TaggedInt64 auctionId64;
    // /Subsection: Auction data
    // Subsection: Inventory hierarchy data
    Datacratic::Id id; // TinyTag id. Mechanism to anonymously label inventory segments
    Datacratic::Id siteId;
    Datacratic::Id inventorySourceId;  // TODO validation, value may be hidden
    // /Subsection: Inventory hierarchy data
    // Subsection: General data
    std::string size;            // TODO validation, need enum of allowed vals
    std::vector<std::string> sizes;   // TODO validation, need enum of allowed vals
    AdPosition position;    // TODO enum, values in positions
    std::string tagFormat;       // TODO enum, values in tagFormats
    std::vector<Datacratic::TaggedInt> allowedMediaTypes;      // TODO enum, need valid values
    std::vector<Datacratic::TaggedInt> allowedMediaSubtypes;     // TODO enum, need valid values
    std::vector<std::string> mediaSubtypes;       // TODO enum, need valid values
    std::vector<InventoryAudit> inventoryAudits;
    Datacratic::TaggedInt venueId;                  // TODO validation OPTIONAL, not used
    Datacratic::TaggedInt adProfileId;
    // /Subsection: General data
    // Subsection: Pricing data
    Datacratic::TaggedFloatDef<0> reservePrice;
    Datacratic::TaggedDoubleDef<0> estimatedClearPrice;
    Datacratic::TaggedDoubleDef<0> estimatedAveragePrice;
    Datacratic::TaggedBoolDef<false> estimatedPriceVerified;
    // /Subsection: Pricing data
    // Subsection: Owner-specific data
    Datacratic::UnicodeString tagData;                  // "Other data related to TinyTag ID"
    Datacratic::TaggedBoolDef<false> exclusiveDefault;
    Datacratic::TaggedInt defaultCreativeId;
    // /Subsection: Owner-specific data
    // TODO validation rule: always sent for "mobile_app", never sent for "web"
    // TODO clarify enum, spec implies {"web", "mobile_browser", "mobile_app"}
    // Subsection: Mobile-specific fields
    std::string supplyType;      // TODO enum, values in supplyTypes
    std::vector<std::string> creativeFormats;// TODO enum, values in creativeFormats
    std::vector<std::string> creativeActions;// TODO enum, values in creativeActions
    Datacratic::TaggedBoolDef<false> smallerSizesAllowed;
    // /Subsection: Mobile-specific fields
    Json::Value unparseable;    ///< Unparseable fields get put here
};

// Official from (15.11.2013)
// https://wiki.appnexus.com/display/adnexusdocumentation/Operating+System+Service
const std::unordered_map<int, std::string> deviceOs = {
		{ 0,  "Unknown"}, { 1,  "Windows 7"}, { 2,  "Windows Vista"},
		{ 3,  "Windows XP"}, { 4,  "Windows 2000"},
		{ 5,  "Windows (other versions)"}, { 6,  "Android"},
		{ 7,  "Linux"}, { 8,  "iPhone"}, { 9,  "iPod"},
		{ 10,  "iPad"}, { 11,  "Mac"}, { 12,  "Blackberry"},
		{ 13,  "Windows Phone 7"}
};


struct BidInfo {
    // Subsection: user
    Datacratic::TaggedInt64 userId64;
    Datacratic::UnicodeString userAgent;
    Datacratic::TaggedIntDef<0> operatingSystem;
    // \"Accept-Language\" header from browser (using ISO-639 language and ISO-3166 country codes)
    std::string acceptedLanguages;   // "en-US,en;q=0.8"
    Datacratic::TaggedInt language;         // value set by call to getLanguageCode() in this NS
    Datacratic::TaggedBoolDef<true> noFlash;
    Datacratic::TaggedBoolDef<true> noCookies;
    std::string gender;      // TODO enum, values in gender
    Datacratic::TaggedInt age;
    std::vector<Segment> segments;
    // /Subsection: user
    // Subsection: geographical data
    std::string ipAddress;           // TODO IP octets validation
    Datacratic::UnicodeString country;         // TODO no enum values in spec
    Datacratic::UnicodeString region;          // TODO no enum values in spec
    Datacratic::UnicodeString city;                // TODO no enum values in spec
    std::string postalCode;          // TODO validate postalCodes US etc. :-(
    Datacratic::TaggedInt dma;                    // TODO no enum values in spec
    std::string timeZone;            // TODO no enum values in spec
    // /Subsection: geographical data
    // Subsection: userdata from server-side cookie storage
    Json::Value userdataJson;   // TODO validate valid JSON
    Datacratic::TaggedInt totalClicks;           // DEPRECATED
    // /Subsection: userdata from server-side cookie storage
    // Subsection: Inventory (page) information
    Datacratic::Id sellingMemberId;
    Datacratic::UnicodeString url;             // TODO? validate valid URL
    std::string domain;
    std::string inventoryClass;  // DEPRECATED, TODO enum, values in inventoryClasses
    std::vector<InventoryAudit> inventoryAudits;
    Datacratic::TaggedBoolDef<false> withinIframe;
    Datacratic::TaggedInt publisherId;
    Datacratic::TaggedBoolDef<false> isSecure; // Note: All connections to secure inventory must be secure.
    // /Subsection: Inventory (page) information
    // Subsection: Mobile-specific fields
    std::string appId;   // Spec says this is a string and a UID
    // Spec: "Expressed in the format 'snnn.ddddddddddddd,snnn.ddddddddddddd',
    //  south and west are negative, up to 13 decimal places of precision."
    // Example: "38.7875232696533,-77.2614831924438"
    std::string loc;     // TODO validation per above comment
    // /Subsection: Mobile-specific fields
    // Subsection: Mobile fields not available in initial release
    Datacratic::TaggedInt carrier;    // TODO validation, valid values "WIFI" or from vendor
    Datacratic::TaggedInt make;       // TODO validation, valid values "WIFI" or from vendor
    Datacratic::TaggedInt model;      // TODO validation, valid values "WIFI" or from vendor
    // /Subsection: Mobile fields not available in initial release

    std::string getANDeviceOsStringForCode(int code) const {
        return deviceOs.at(code);   // at() throws if key not found
    }

    Json::Value unparseable;    ///< Unparseable fields get put here
};


struct BidRequestMsg
{
    // Subsection: General data
    Datacratic::Id memberAdProfileId;
    std::string timestamp;           // TODO timestamp with validation
    Datacratic::TaggedInt bidderTimeoutMs;
    // /Subsection: General data
    BidInfo bidInfo;
    std::vector<Member> members;
    std::vector<Tag> tags;
    // Subsection: Owner-specific data
    Datacratic::TaggedBoolDef<false> allowExclusive;
    // /Subsection: Owner-specific data
    // Subsection: Debug data
    Datacratic::TaggedBoolDef<false> debugRequested;
    Datacratic::Id debugMemberId;
    Datacratic::TaggedBoolDef<false> test;
    // /Subsection: Debug data
    // Subsection: Other data
    std::vector<Datacratic::TaggedInt> excludedAttributes;
    Datacratic::TaggedBool singlePhase;
    // /Subsection: Other data
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct BidRequest {
    BidRequestMsg  bidRequest;
    Json::Value    unparseable;    ///< Unparseable fields get put here
};

namespace ANHelpers {
// TODO Implement validation using these constants pulled from the AN Bid Request spec
// TODO REMOVE THIS const vector<string> positions = {"below", "above", "unknown"};
const std::vector<std::string> tagFormats = {"iframe", "javascript"};
const std::vector<std::string> supplyTypes = {"web", "mobile_browser", "mobile_app"};
const std::vector<std::string> creativeFormats = {
    "text", "image", "url-html",
    "url-js", "flash", "raw-js",
    "raw-html", "iframe-html", "urlvast"
};
const std::vector<std::string> creativeActions = {"click-to-web", "click-to-call"};
const std::vector<std::string> inventoryClasses = {"class_1", "class_2", "class_3", "unaudited", "blacklist"};
const std::vector<std::string> genders = {"male", "female"};
const std::vector<std::string> intendedAudences = {"general", "children", "young_adult", "mature"};
const std::vector<std::string> languageMap = {
    "Other", "English", "Chinese",
    "Spanish", "Japanese", "French"
    "German", "Arabic", "Portuguese",
    "Russian", "Korean", "Italian", "Dutch"
};
const std::unordered_map<std::string, int> languageCodeMap = {
    {"Other", 0}, {"English", 1}, {"Chinese", 2},
    {"Spanish", 3}, {"Japanese", 4}, {"French", 5},
    {"German", 6}, {"Arabic", 7}, {"Portuguese", 8},
    {"Russian", 9}, {"Korean", 10}, {"Italian", 11}, {"Dutch", 12}
};


//static const string getLanguage(int languageCode) {
//    return languageMap[languageCode];       // throws if index out of range
//}
//static const int getLanguageCode(const string & language) {
//    return languageCodeMap.at(language);    // at() throws if key not found
//}
} // namespace ANHelpers

} // namespace AppNexus

