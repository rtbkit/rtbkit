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
#include "openrtb/openrtb.h"

namespace FBX {

using std::string;
using std::vector;
using std::unique_ptr;

using namespace Datacratic;

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

struct PageTypeCode: public TaggedEnum<PageTypeCode> {
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
    TaggedInt numSlots;             ///< Estimated number of ad slots in the placement

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
    Id requestId;                   ///< Bid request ID
    string partnerMatchId;		    ///< Partner’s user ID 

    RtbUserContext userContext;		///< An object of type UserContext
    RtbPageContext pageContext;     ///< An object of type PageContext
    TaggedBool istest;              ///< Indicates an auction being held purely for debugging purposes
    TaggedBool allowViewTag;        ///< Indicates if view tags are accepted.

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
	Optional<string> title;
	Optional<string> body;
	Optional<string> link;
	Optional<string> creativeHash;
	Optional<string> imageUrl;
};


/*****************************************************************************/
/* BID                                                                       */
/*****************************************************************************/

/** Bid Object
*/

struct RtbBid {
	Id adId;                        ///< FB ad id for ad which partner wishes to show
	TaggedInt bidNative;            ///< the CPM bid in cents
	string impressionPayload;       ///< opaque blob which FB will return to the partner in the win notification
	string clickPayload;            ///< opaque blob which FB will return to the partner upon user click
	Optional<RtbBidDynamicCreativeSpec> dynamicCreativeSpec;
	vector<string> viewTagUrls;     ///< A list of view tag URL's to be fired when the impression is served.
};


/*****************************************************************************/
/* BID RESPONSE                                                              */
/*****************************************************************************/

/** 4.3.1 Bid Response Object

*/

struct BidResponse {
    Id requestId;                   ///< Same requestId as in the bid request
    vector<RtbBid> bids;            ///< Array of type RtbBid
    Optional<TaggedInt> processingTimeMs; ///< Time it takes for your servers to process the bid request
};


} // namespace FBX
