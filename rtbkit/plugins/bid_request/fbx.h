/** fbx.h                                                      -*- C++ -*-
    Jean-Sebastien Bejeau, 18 June 2013

    This file is part of RTBkit.

    Programmatic Description of the FBX specification.

    Documentation is taken from the FBX Specification available at
    http://developers.facebook.com/docs/reference/ads-api/rtb/
    
*/

#pragma once

#include <string>
#include <memory>
#include <vector>
#include "soa/types/id.h"
#include "soa/types/string.h"
#include "soa/types/url.h"
#include "jml/utils/compact_vector.h"
#include "soa/jsoncpp/value.h"
#include <iostream>
#include "rtbkit/openrtb/openrtb.h"

namespace FBX {


/*****************************************************************************/
/* MIME TYPES                                                                */
/*****************************************************************************/

struct MimeType : public OpenRTB::MimeType{};

/*****************************************************************************/
/* PAGE TYPE CODE	                                                     */
/*****************************************************************************/

/* RtbPageContext
	Page Type

*/

struct PageTypeCode: public Datacratic::TaggedEnum<PageTypeCode> {
    enum Vals {
    PAGE_UNSPECIFIED = 0,
    PAGE_CANVAS = 1,
    PAGE_PROFILE = 2,
    PAGE_SEARCH = 3,
    PAGE_EVENT = 4,
    PAGE_GROUP = 5,
    PAGE_PHOTO = 6,
    PAGE_HOME = 7,
    PAGE_OTHER = 8,
    PAGE_ALBUM = 9,
    PAGE_ADBOARD = 10,
    PAGE_PHOTOS = 11,
    PAGE_PERMALINK = 12,
    PAGE_REQS = 13,
    PAGE_PAGES = 14,
    PAGE_DASHBOARDS = 15,
    PAGE_HOME_FEED = 16,
    PAGE_WEB_MESSENGER = 26,
    PAGE_CALENDAR = 29,
    PAGE_NOTE = 30
    };
};

/*****************************************************************************/
/* RtbUserContext                                                            */
/*****************************************************************************/

/** RtbUserContext    
*/

struct RtbUserContext {
    std::string ipAddressMasked;    ///< User IP address
    std::string userAgent;		    ///< User agent from the user browser
    std::string country;          	///< Country code (ISO 3166-2)
};


/*****************************************************************************/
/* RtbPageContext                                                            */
/*****************************************************************************/

/** RtbPageContext
*/

struct RtbPageContext {
    PageTypeCode pageTypeId;        ///< Page type
    Datacratic::TaggedInt numSlots;    ///< Estimated number of ad slots in the placement

};

/*****************************************************************************/
/* BID REQUEST                                                               */
/*****************************************************************************/


/** 3.3.1 Bid Request Object

    The top-level bid request object contains a globally unique bid request
    or auction ID. This “id” attribute is required as is at least one “imp”
    (i.e., impression) object. Other attributes are optional since an exchange
    may establish default values.

    RTB transactions are initiated when an exchange or other supply source
    sends a bid request to a bidder. The bid request consists of a bid request
    object, at least one impression object, and may optionally include
    additional objects providing impression context.
*/

struct BidRequest {
    Datacratic::Id requestId;       ///< Bid request ID
    std::string partnerMatchId;		///< Partner’s user ID 

    RtbUserContext userContext;		///< An object of type UserContext
    RtbPageContext pageContext;     ///< An object of type PageContext
    Datacratic::TaggedBool istest;  ///< Indicates an auction being held purely for debugging purposes
    Datacratic::TaggedBool allowViewTag;   ///< Indicates if view tags are accepted.

    Json::Value unparseable;        ///< Unparseable fields get put here
};

/*****************************************************************************/
/* RtbBidDynamicCreativeSpec                                                                       */
/*****************************************************************************/

/** RtbBidDynamicCreativeSpec
  The fields below are optional, and will overwrite the corresponding fields
  in the original ad creative
*/
struct RtbBidDynamicCreativeSpec {
    Datacratic::Optional<std::string> title;
    Datacratic::Optional<std::string> body;
	Datacratic::Optional<std::string> link;
	Datacratic::Optional<std::string> creativeHash;
	Datacratic::Optional<std::string> imageUrl;
};


/*****************************************************************************/
/* BID                                                                       */
/*****************************************************************************/

/** Bid Object
*/

struct RtbBid {
    Datacratic::Id adId;                   ///< FB ad id for ad which partner wishes to show
    Datacratic::TaggedInt bidNative;      ///< the CPM bid in cents
    std::string impressionPayload;       ///< opaque blob which FB will return to the partner in the win notification
    std::string clickPayload;            ///< opaque blob which FB will return to the partner upon user click
    Datacratic::Optional<RtbBidDynamicCreativeSpec> dynamicCreativeSpec;
    std::vector<std::string> viewTagUrls;     ///< A list of view tag URL's to be fired when the impression is served.
};


/*****************************************************************************/
/* BID RESPONSE                                                              */
/*****************************************************************************/

/** 4.3.1 Bid Response Object

*/

struct BidResponse {
    Datacratic::Id requestId;   ///< Same requestId as in the bid request
    std::vector<RtbBid> bids;        ///< Array of type RtbBid
    Datacratic::Optional<Datacratic::TaggedInt> processingTimeMs; ///< Time it takes for your servers to process the bid request
};


} // namespace FBX
