/* nexage.h                                               -*- C++ -*-
   Syed Murtaza Hussain Kazmi, 11 September 2013
   Copyright (c) 2013 Virtual Force.  All rights reserved.
    
    Structs that map to the Nexage JSON bid request data format.
*/

#pragma once
#include "openrtb/openrtb.h"
//#include <string>
//#include <memory>
//#include <vector>
//#include "soa/types/id.h"
//#include "soa/types/string.h"
//#include "soa/types/url.h"
//#include "jml/utils/compact_vector.h"
//#include "soa/jsoncpp/value.h"
//#include "soa/types/basic_value_descriptions.h"
//#include <iostream>


namespace Nexage {

	
using std::string;
using std::vector;
using std::unordered_map;
using Datacratic::Id;
using Datacratic::Utf8String;
using Datacratic::Url;
using Datacratic::CSList;
using Datacratic::Optional;
using Datacratic::List;

using OpenRTB::TaggedInt;
using OpenRTB::TaggedBool;
using OpenRTB::TaggedFloat;
using OpenRTB::TaggedInt64;
using OpenRTB::TaggedDouble;

struct NexageLocationType: public OpenRTB::TaggedEnum<NexageLocationType> {
    enum Vals {
        UNSPECIFIED = -1,		///< Not explicitly specified

        GPS			= 1,        ///< GPS/Location Services
        IP_ADDRESS	= 2,		///< Ip Address
        USER		= 3        ///< Provided by user
    };
};

struct NexageGeo {
    ~NexageGeo();
    TaggedFloat lat;        ///< Latitude of user (-90 to 90; South negative)
    TaggedFloat lon;        ///< Longtitude (-180 to 180; west is negative)
    string country;         ///< Country code (ISO 3166-1 Alpha-3)
    string region;          ///< Region code (ISO 3166-2)
    string regionfips104;   ///< Region using FIPS 10-4
    string metro;           ///< Metropolitan region (Google Metro code)
    Utf8String city;        ///< City name (UN Code for Trade and Transport Loc)
    string zip;             ///< Zip or postal code
    NexageLocationType type;      ///< Source of Geo data (table 6.15)
    Json::Value ext;        ///< Extensions go here, new in OpenRTB 2.1

    /// Datacratic extensions
    string dma;             ///< Direct Marketing Association code
   
};

struct NexageUser {
    ~NexageUser();
    Id id;                     ///< Exchange-specific user ID
    Id buyeruid;               ///< Exchange seat-specific user ID
    TaggedInt yob;             ///< Year of birth
    string gender;             ///< Gender: Male, Female, Other
    CSList keywords;           ///< List of keywords of consumer intent
    string customdata;         ///< Custom data from exchange
    NexageGeo geo;             ///< Geolocation of user at registration
    vector<OpenRTB::Data> data;         ///< User data segments
    Json::Value ext;           ///< Extensions go here, new in OpenRTB 2.1
};

struct BidRequest {
    ~BidRequest();
    Id id;                             ///< Bid request ID

    vector<OpenRTB::Impression> imp;            ///< List of impressions
    //unique_ptr<Context> context;				// TODO: factor out of site and app
    Optional<OpenRTB::Site> site;
    Optional<OpenRTB::App> app;
    Optional<OpenRTB::Device> device;
    Optional<NexageUser> user;

    OpenRTB::AuctionType at;           ///< Auction type (1=first/2=second party)
    TaggedInt tmax;                    ///< Max time avail in ms
    vector<string> wseat;              ///< Allowed buyer seats
    TaggedBool allimps;                ///< All impressions in BR (for road-blocking)
    vector<string> cur;                ///< Allowable currencies
    List<OpenRTB::ContentCategory> bcat;        ///< Blocked advertiser categories (table 6.1)
    vector<string> badv;               ///< Blocked advertiser domains
    Json::Value ext;                   ///< Protocol extensions
    Json::Value unparseable;           ///< Unparseable fields get put here
};

//OpenRTB::LocationType convertLocationType(Nexage::LocationType type) {
//    OpenRTB::LocationType ret;
//
//    if (type.value() == Nexage::LocationType::UNSPECIFIED) {
//        ret.val = OpenRTB::LocationType::UNSPECIFIED;
//    } 
//    else if (type.value() == Nexage::AdPosition::GPS) {
//        ret.val = OpenRTB::AdPosition::GPS;
//    } 
//    else if (type.value() == Nexage::AdPosition::IP_ADDRESS) {
//        ret.val = OpenRTB::AdPosition::IP_ADDRESS;
//    }
//	else if (type.value() == Nexage::AdPosition::USER) {
//        ret.val = OpenRTB::AdPosition::USER;
//    }
//   
//    return ret;
//}


} // namespace Nexage

