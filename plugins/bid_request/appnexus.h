/** appnexus.h                                                      -*- C++ -*-
    Mark Weiss, 8 May 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

    This file is part of RTBkit.
    
    Structs that map to the AppNexus JSON bid request data format.
*/

#pragma once

#include "soa/jsoncpp/json.h"
#include "soa/types/id.h"
#include "soa/types/string.h"
#include "soa/types/url.h"
#include "openrtb/openrtb.h"
#include <string>
#include <vector>
#include <unordered_map>

namespace AppNexus {

using std::string;
using std::vector;
using std::unordered_map;
using Datacratic::Id;
using Datacratic::Utf8String;
using Datacratic::Url;
using OpenRTB::TaggedInt;
using OpenRTB::TaggedBool;
using OpenRTB::TaggedFloat;
using OpenRTB::TaggedInt64;
using OpenRTB::TaggedDouble;


struct AdPosition: public OpenRTB::TaggedEnum<AdPosition, 0> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN = 0,
        ABOVE = 1,
        BELOW = 3
    };
};

struct Segment {
    Id id;
    Id memberId;
    string code;                // TODO Get values from "Segment Pixels" section in AN documentation
    TaggedInt lastSeenMin;      // TODO POSIX valid int ts
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct InventoryAudit {
    Id auditorMemberId;
    string intendedAudience;    // TODO enum, values in intendedAudences
    vector<TaggedInt> inventoryAttributes;  // TODO no enum values in spec
    vector<TaggedInt> contentCategories;    // TODO no enum values in spec
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct Member {
    Id id;
    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct Tag {
    // Subsection: Auction data
    TaggedInt64 auctionId64;
    // /Subsection: Auction data
    // Subsection: Inventory hierarchy data
    Id id; // TinyTag id. Mechanism to anonymously label inventory segments
    Id siteId;
    Id inventorySourceId;  // TODO validation, value may be hidden
    // /Subsection: Inventory hierarchy data
    // Subsection: General data
    string size;            // TODO validation, need enum of allowed vals
    vector<string> sizes;   // TODO validation, need enum of allowed vals
    AdPosition position;    // TODO enum, values in positions
    string tagFormat;       // TODO enum, values in tagFormats
    vector<TaggedInt> allowedMediaTypes;      // TODO enum, need valid values
    vector<TaggedInt> allowedMediaSubtypes;     // TODO enum, need valid values
    vector<string> mediaSubtypes;       // TODO enum, need valid values
    vector<InventoryAudit> inventoryAudits;
    TaggedInt venueId;                  // TODO validation OPTIONAL, not used
    TaggedInt adProfileId;
    // /Subsection: General data
    // Subsection: Pricing data
    TaggedFloat reservePrice;
    TaggedDouble estimatedClearPrice;
    TaggedDouble estimatedAveragePrice;
    TaggedBool estimatedPriceVerified;
    // /Subsection: Pricing data
    // Subsection: Owner-specific data
    Utf8String tagData;                  // "Other data related to TinyTag ID"
    TaggedBool exclusiveDefault;
    TaggedInt defaultCreativeId;
    // /Subsection: Owner-specific data
    // TODO validation rule: always sent for "mobile_app", never sent for "web"
    // TODO clarify enum, spec implies {"web", "mobile_browser", "mobile_app"}
    // Subsection: Mobile-specific fields
    string supplyType;      // TODO enum, values in supplyTypes
    vector<string> creativeFormats;// TODO enum, values in creativeFormats
    vector<string> creativeActions;// TODO enum, values in creativeActions
    TaggedBool smallerSizesAllowed;
    // /Subsection: Mobile-specific fields
    Json::Value unparseable;    ///< Unparseable fields get put here
};

const unordered_map<int, string> deviceOs = {
      {0, "Unknown"}, {1, "Micosoft Windows"}, {2, "Micosoft Windows"}, {3, "Micosoft Windows"}, {4, "Micosoft Windows"}, {5, "Micosoft Windows"},
      {6, "Android"}, {7, "Linux"}, {8, "Apple iOS"}, {9, "Apple iOS"}, {10, "Apple iOS"}, {11, "Apple Mac OSX"}, {12, "Blackberry RIMOS"},
      {12, "Microsoft Windows Phone OS"}, {14, "Microsoft Windows"}, {15, "Android"}, {16, "Android"}, {17, "Android"}, {18, "Android"},
      {19, "Apple iOS"}, {20, "Apple iOS"}, {21, "Apple iOS"}, {22, "Apple iOS"}
};

const unordered_map<int, string> deviceOsVersion = {
      {0, "Unknown"}, {1, "Windows 7"}, {2, "Windows Vista"}, {3, "Windows XP"}, {4, "Windows 2000"}, {5, "Windows (other versions)"},
      {6, "Android (other versions)"}, {7, "Linux"}, {8, "iPhone - iOS (other versions)"}, {9, "iPod"}, {10, "iPad - iOS (other versions)"}, 
      {11, "Apple Mac OSX"}, {12, "Blackberry (other versions)"},
      {12, "Windows Phone 7 (other versions)"}, {14, "Windows 8"}, {15, "Android 2.0/2.1"}, {16, "Android 2.2/2.3"}, {17, "Android 3.x"}, {18, "Android 4.x"},
      {19, "iOS 3"}, {20, "iOS 4"}, {21, "iOS 5"}, {22, "iOS 6"}
};

struct BidInfo {
    // Subsection: user
    TaggedInt64 userId64;
    Utf8String userAgent;
    // TODO Get actual values from here: https://wiki.appnexus.com/display/adnexusdocumentation/Operating+System+Service
    TaggedInt operatingSystem;
    // \"Accept-Language\" header from browser (using ISO-639 language and ISO-3166 country codes)
    string acceptedLanguages;   // "en-US,en;q=0.8"
    TaggedInt language;         // value set by call to getLanguageCode() in this NS
    TaggedBool noFlash;
    TaggedBool noCookies;
    string gender;      // TODO enum, values in gender
    TaggedInt age;
    vector<Segment> segments;
    // /Subsection: user
    // Subsection: geographical data
    string ipAddress;           // TODO IP octets validation
    Utf8String country;         // TODO no enum values in spec
    Utf8String region;          // TODO no enum values in spec
    Utf8String city;                // TODO no enum values in spec
    string postalCode;          // TODO validate postalCodes US etc. :-(
    TaggedInt dma;                    // TODO no enum values in spec
    string timeZone;            // TODO no enum values in spec
    // /Subsection: geographical data
    // Subsection: userdata from server-side cookie storage
    Json::Value userdataJson;   // TODO validate valid JSON
    TaggedInt totalClicks;           // DEPRECATED
    // /Subsection: userdata from server-side cookie storage
    // Subsection: Inventory (page) information
    Id sellingMemberId;
    Utf8String url;             // TODO? validate valid URL
    string domain;
    string inventoryClass;  // DEPRECATED, TODO enum, values in inventoryClasses
    vector<InventoryAudit> inventoryAudits;
    TaggedBool withinIframe;
    TaggedInt publisherId;
    TaggedBool isSecure; // Note: All connections to secure inventory must be secure.
    // /Subsection: Inventory (page) information
    // Subsection: Mobile-specific fields
    string appId;   // Spec says this is a string and a UID
    // Spec: "Expressed in the format 'snnn.ddddddddddddd,snnn.ddddddddddddd',
    //  south and west are negative, up to 13 decimal places of precision."
    // Example: "38.7875232696533,-77.2614831924438"
    string loc;     // TODO validation per above comment
    // /Subsection: Mobile-specific fields
    // Subsection: Mobile fields not available in initial release
    TaggedInt carrier;    // TODO validation, valid values "WIFI" or from vendor
    TaggedInt make;       // TODO validation, valid values "WIFI" or from vendor
    TaggedInt model;      // TODO validation, valid values "WIFI" or from vendor
    // /Subsection: Mobile fields not available in initial release

    string getANDeviceOsStringForCode(int code) const {
        return deviceOs.at(code);   // at() throws if key not found
    }

    string getANDeviceOsVersionStringForCode(int code) const {
        return deviceOsVersion.at(code);   // at() throws if key not found
    }

    Json::Value unparseable;    ///< Unparseable fields get put here
};

struct BidRequest {
    // Subsection: General data
    Id memberAdProfileId;
    string timestamp;           // TODO timestamp with validation
    TaggedInt bidderTimeoutMs;
    // /Subsection: General data
    BidInfo bidInfo;
    vector<Member> members;
    vector<Tag> tags;
    // Subsection: Owner-specific data
    TaggedBool allowExclusive;
    // /Subsection: Owner-specific data
    // Subsection: Debug data
    TaggedBool debugRequested;
    Id debugMemberId;
    TaggedBool test;
    // /Subsection: Debug data
    // Subsection: Other data
    vector<TaggedInt> excludedAttributes;
    TaggedBool singlePhase;
    // /Subsection: Other data
    Json::Value unparseable;    ///< Unparseable fields get put here
};


namespace ANHelpers {
    // TODO Implement validation using these constants pulled from the AN Bid Request spec
    // TODO REMOVE THIS const vector<string> positions = {"below", "above", "unknown"};
    const vector<string> tagFormats = {"iframe", "javascript"};
    const vector<string> supplyTypes = {"web", "mobile_browser", "mobile_app"};
    const vector<string> creativeFormats = {
        "text", "image", "url-html",
        "url-js", "flash", "raw-js",
        "raw-html", "iframe-html", "urlvast"
    };
    const vector<string> creativeActions = {"click-to-web", "click-to-call"};
    const vector<string> inventoryClasses = {"class_1", "class_2", "class_3", "unaudited", "blacklist"};
    const vector<string> genders = {"male", "female"};
    const vector<string> intendedAudences = {"general", "children", "young_adult", "mature"};
    const vector<string> languageMap = {
        "Other", "English", "Chinese",
        "Spanish", "Japanese", "French"
        "German", "Arabic", "Portuguese",
        "Russian", "Korean", "Italian", "Dutch"
    };
    const unordered_map<string, int> languageCodeMap = {
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

