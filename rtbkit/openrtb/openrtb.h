/** openrtb.h                                                      -*- C++ -*-
    Jeremy Barnes, 21 February 2013
    Copyright (c) 2013 Datacratic Inc.  All rights reserved.

    This file is part of RTBkit.

    Programmatic Description of the OpenRTB 2.0 specification.

    Documentation is taken from the OpenRTB Specification 2.0 available at
    http://www.iab.net/media/file/OpenRTB_API_Specification_Version2.0_FINAL.PDF

    That document has the following license:

    OpenRTB Specification by OpenRTB is licensed under a Creative Commons
    Attribution 3.0 License, based on a work at openrtb.info. Permissions
    beyond the scope of this license may be available at http://openrtb.info.
    To view a copy of this license, visit
    http://creativecommons.org/licenses/by/3.0/
    or write to Creative Commons, 171 Second Street, Suite 300, San Francisco,
    CA 94105, USA.
*/

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "soa/types/id.h"
#include "soa/types/string.h"
#include "soa/types/url.h"
#include "jml/utils/compact_vector.h"
#include "soa/jsoncpp/value.h"
#include "soa/types/basic_value_descriptions.h"
#include <iostream>

namespace OpenRTB {


//using namespace Datacratic;
using Datacratic::Optional;
using Datacratic::List;

/*****************************************************************************/
/* MIME TYPES                                                                */
/*****************************************************************************/

struct MimeType {
    MimeType(const std::string & type = "")
        : type(type)
    {
    }

    std::string type;
    //int val;
};


/*****************************************************************************/
/* CONTENT CATEGORIES                                                        */
/*****************************************************************************/

/** 6.1 Content Categories

    The following list represents the IAB’s contextual taxonomy for
    categorization.  Standard IDs have been adopted to easily support the
    communication of primary and secondary categories for various objects.

    Note to the reader: This OpenRTB table has values derived from the IAB
    Quality Assurance Guidelines (QAG). Users of OpenRTB should keep in
    synch with updates to the QAG values as published on IAB.net.

    (This is a huge taxonomy... there are no enumerated values).
*/

struct ContentCategory {

    ContentCategory(const std::string & val = "")
        : val(val)
    {
    }

    std::string val;

    inline bool operator==(const ContentCategory & other) const {
        return this->val == other.val;
    }

#if 0    
    ContentCategory(int l1 = -1, int l2 = -1)
        : l1(l1), l2(l2)
    {
    }

    int l1;
    int l2;
#endif
};


/*****************************************************************************/
/* BANNER AD TYPES                                                           */
/*****************************************************************************/

/** 6.2 Banner Ad Types

    The following table indicates the types of ads that can be accepted by the
    exchange unless restricted by publisher site settings.
*/

struct BannerAdType: public Datacratic::TaggedEnum<BannerAdType> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        XHTML_TEXT = 1,    ///< XHTML text ad. (usually mobile)
        XHTML_BANNER = 2,  ///< XHTML banner ad. (usually mobile)
        JAVASCRIPT = 3,    ///< JavaScript ad; must be valid XHTML (i.e., script tags included).
        IFRAME = 4         ///< Full iframe HTML
    };
};

/*****************************************************************************/
/* CREATIVE ATTRIBUTES                                                       */
/*****************************************************************************/

/** 6.3 Creative Attributes

    The following table specifies a standard list of creative attributes that
    can describe an ad being served or serve as restrictions of thereof.
*/

struct CreativeAttribute: public Datacratic::TaggedEnum<CreativeAttribute> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        AUDIO_AD_AUTO_PLAY = 1,
        AUDIO_AD_USER_INITIATED = 2,
        EXPANDABLE_AUTOMATIC = 3,
        EXPANDABLE_USER_INITIATED_CLICK = 4,
        EXPANDABLE_USER_INITIATED_ROLLOVER = 5,
        IN_BANNER_VIDEO_AD_AUTO_PLAY = 6,
        IN_BANNER_VIDEO_AD_USER_INITIATED = 7,
        POP = 8,
        PROVOCATIVE_OR_SUGGESTIVE_IMAGERY = 9,
        SHAKY_FLASHING_FLICKERING_EXTREME_ANIMATION_SMILEYS = 10,
        SURVEYS = 11,
        TEXT_ONLY = 12,
        USER_INTERACTIVE = 13,
        WINDOWS_DIALOG_OR_ALERT_STYLE = 14,
        HAS_AUDIO_ON_OFF_BUTTON = 15,
        AD_CAN_BE_SKIPPED = 16
    };
};


/*****************************************************************************/
/* API FRAMEWORKS                                                            */
/*****************************************************************************/

/** 6.4 API Frameworks

    This is a list of API frameworks.
*/

struct ApiFramework: public Datacratic::TaggedEnum<ApiFramework> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        VPAID_1 = 1,    ///< IAB Video Player-Ad Interface Definitions V1
        VPAID_2 = 2,    ///< IAB Video Player-Ad Interface Definitions V2
        MRAID = 3,      ///< IAB Mobile Rich Media Ad Interface Definitions
        ORMMA = 4,       ///< Google Open Rich Media Mobile Advertising
        MRAID2 = 5      ///< IAB Mobile Rich Media Ad Interface Definitions V2
    };
};


/*****************************************************************************/
/* AD POSITION                                                               */
/*****************************************************************************/

/** 6.5 Ad Position

    The following table specifies the position of the ad as a relative
    measure of visibility or prominence.
    
    Note to the reader: This OpenRTB table has values derived from the IAB
    Quality Assurance Guidelines (QAG). Users of OpenRTB should keep in
    synch with updates to the QAG values as published on IAB.net.
*/

struct AdPosition: public Datacratic::TaggedEnum<AdPosition, 0> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN = 0,
        ABOVE = 1,
        BETWEEN_DEPRECATED = 2,
        BELOW = 3,
        HEADER = 4,
        FOOTER = 5,
        SIDEBAR = 6,
        FULLSCREEN = 7
    };
};


/*****************************************************************************/
/* VIDEO LINEARITY                                                           */
/*****************************************************************************/

/** 6.6 Video Linearity

    The following table indicates the options for video linearity.
    "In-stream" or "linear" video refers to pre-roll, post-roll, or mid-roll
    video ads where the user is forced to watch ad in order to see the video
    content. “Overlay” or “non-linear” refer to ads that are shown on top of
    the video content.  Note to the reader: This OpenRTB table has values
    derived from the IAB Quality Assurance Guidelines (QAG). Users of OpenRTB
    should keep in synch with updates to the QAG values as published on
    IAB.net.
*/

struct VideoLinearity: public Datacratic::TaggedEnum<VideoLinearity> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        IN_STREAM = 1,
        OVERLAY = 2
    };
};


/*****************************************************************************/
/* VIDEO BID RESPONSE PROTOCOLS                                              */
/*****************************************************************************/

/** 6.7 Video Bid Response Protocols

    The following table lists the options for video bid response protocols
    that could be supported by an exchange.
*/

struct VideoBidResponseProtocol: public Datacratic::TaggedEnum<VideoBidResponseProtocol> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        VAST1 = 1,
        VAST2 = 2,
        VAST3 = 3,
        VAST1_WRAPPER = 4,
        VAST2_WRAPPER = 5,
        VAST3_WRAPPER = 6
    };
};

/*****************************************************************************/
/* VIDEO PLAYBACK METHODS                                                    */
/*****************************************************************************/

/** 6.8 Video Playback Methods

 */

struct VideoPlaybackMethod: public Datacratic::TaggedEnum<VideoPlaybackMethod> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        AUTO_PLAY_SOUND_ON = 1,
        AUTO_PLAY_SOUND_OFF = 2,
        CLICK_TO_PLAY = 3,
        MOUSE_OVER = 4
    };
};


/*****************************************************************************/
/* VIDEO START DELAY                                                         */
/*****************************************************************************/

/** 6.9 Video Start Delay

    The following table lists the various options for the video start delay.
    If the start delay value is greater than 0 then the position is mid-roll,
    and the value represents the number of seconds into the content that the
    ad will be displayed.  If the start delay time is not available, the
    exchange can report the position of the ad in general terms using this
    table of negative numbers.
*/

struct VideoStartDelay: public Datacratic::TaggedEnum<VideoStartDelay> {
    enum Vals {
        UNSPECIFIED = -3,  ///< Not explicitly specified

        PRE_ROLL = 0,
        GENERIC_MID_ROLL = -1,
        GENERIC_POST_ROLL = -2
    };
};


/*****************************************************************************/
/* CONNECTION TYPE                                                           */
/*****************************************************************************/

/** 6.10 Connection Type

    The following table lists the various options for the connection type.
*/

struct ConnectionType: public Datacratic::TaggedEnum<ConnectionType> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN = 0,
        ETHERNET = 1,
        WIFI = 2,
        CELLULAR_UNKNOWN = 3,
        CELLULAR_2G = 4,
        CELLULAR_3G = 5,
        CELLULAR_4G = 6
    };
};


/*****************************************************************************/
/* EXPANDABLE DIRECTION                                                      */
/*****************************************************************************/

/** 6.11 Expandable Direction

    The following table lists the directions in which an expandable ad may
    expand, given the positioning of the ad unit on the page and constraints
    imposed by the content.
*/

struct ExpandableDirection: public Datacratic::TaggedEnum<ExpandableDirection> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        LEFT = 1,
        RIGHT = 2,
        UP = 3,
        DOWN = 4,
        FULLSCREEN = 5
    };
};


/*****************************************************************************/
/* CONTENT DELIVERY METHOD                                                   */
/*****************************************************************************/

/** 6.12 Content Delivery Methods

    The following table lists the various options for the delivery of video
    content.
*/

struct ContentDeliveryMethod: public Datacratic::TaggedEnum<ContentDeliveryMethod> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        STREAMING = 1,
        PROGRESSIVE = 2
    };
};


/*****************************************************************************/
/* CONTENT CONTEXT                                                           */
/*****************************************************************************/

/** 6.13 Content Context

    The following table lists the various options for the content context;
    what type of content is it.  Note to the reader: This OpenRTB table has
    values derived from the IAB Quality Assurance Guidelines (QAG). Users of
    OpenRTB should keep in synch with updates to the QAG values as published
    on IAB.net.
*/

struct ContentContext: public Datacratic::TaggedEnum<ContentContext> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        VIDEO = 1,
        GAME = 2,
        MUSIC = 3,
        APPLICATION = 4,
        TEXT = 5,
        OTHER = 6,
        UNKNOWN = 7
    };
};


/*****************************************************************************/
/* VIDEO QUALITY                                                             */
/*****************************************************************************/

/** 6.14 Video Quality

    The following table lists the options for the video quality (as defined by
    the IAB – http://www.iab.net/media/file/long-form-video-final.pdf).
*/

struct VideoQuality: public Datacratic::TaggedEnum<VideoQuality> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN = 0,
        PROFESSIONAL = 1,
        PROSUMER = 2,
        USER_GENERATED = 3
    };
};


/*****************************************************************************/
/* LOCATION TYPE                                                             */
/*****************************************************************************/

/** 6.15 Location Type

    The following table lists the options to indicate how the geographic
    information was determined.
*/

struct LocationType: public Datacratic::TaggedEnum<LocationType> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        GPS = 1,        ///< GPS/Location Services
        IP_ADDRESS = 2, ///< Ip Address
        USER = 3        ///< Provided by user
    };
};


/*****************************************************************************/
/* DEVICE TYPE                                                               */
/*****************************************************************************/

/** 6.16 Device Type

    The following table lists the options to indicate how the geographic
    information was determined.   Note to the reader: This OpenRTB table has
    values derived from the IAB Quality Assurance Guidelines (QAG). Users of
    OpenRTB should keep in synch with updates to the QAG values as published
    on IAB.net.
*/

struct DeviceType: public Datacratic::TaggedEnum<DeviceType> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        MOBILE_OR_TABLET = 1,
        PC = 2,
        TV = 3,
        PHONE = 4,  ///< OpenRTB 2.2
        TABLET = 5, ///< OpenRTB 2.2
        CONNECTED_DEVICE = 6, ///< OpenRTB 2.2
        SET_TOP_BOX = 7 ///< OpenRTB 2.2
    };
};


/*****************************************************************************/
/* VAST COMPANION TYPES                                                      */
/*****************************************************************************/

/** 6.17 VAST Companion Types

    The following table lists the options to indicate markup types allowed
    for video companion ads.  This table is derived from IAB VAST 2.0+.  See
    www.iab.net/vast/ for more information.
*/

struct VastCompanionType: public Datacratic::TaggedEnum<VastCompanionType> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        STATIC_RESOURCE = 1,
        HTML_RESOURCE = 2,
        IFRAME_RESOURCE = 3
    };
};


/*****************************************************************************/
/* QAG MEDIA RATINGS                                                         */
/*****************************************************************************/

/** 6.18 QAG Media Ratings

    The following table lists the media ratings using the QAG categorization.
    See http://www.iab.net/ne_guidelines for more information.
*/

struct MediaRating: public Datacratic::TaggedEnum<MediaRating> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        ALL_AUDIENCES = 1,
        OVER_12 = 2,
        MATURE_AUDIENCES =3 
    };
};

/*****************************************************************************/
/* No-Bid Reason Codes                                                       */
/*****************************************************************************/

/** 6.19 No-Bid Reason Codes

    The following table lists the options to signal the exchange why the impression
    was not bid on.
*/

struct NoBidReason: public Datacratic::TaggedEnum<NoBidReason> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        UNKNOWN_ERROR = 0,
        TECHNICAL_ERROR = 1,
        INVALID_REQUEST = 2,
        KNOWN_WEB_SPIDER = 3,
        SUSPECTED_NON_HUMAN_TRAFFIC = 4,
        CLOUD_DATACENTER_OR_PROXY_IP = 5,
        UNSUPPORTED_DEVICE = 6,
        BLOCKED_PUBLISHER_OR_SITE = 7,
        UNMATCHED_USER = 8
    };
};


/*****************************************************************************/
/* FRAME POSITION                                                            */
/*****************************************************************************/

struct FramePosition: public Datacratic::TaggedEnum<FramePosition, 0> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        IFRAME = 0,
        TOP_FRAME = 1
    };
};

/*****************************************************************************/
/* SOURCE RELATIONSHIP                                                       */
/*****************************************************************************/

struct SourceRelationship: public Datacratic::TaggedEnum<SourceRelationship> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        INDIRECT = 0,
        DIRECT = 1
    };
};

/*****************************************************************************/
/* EMBEDDABLE                                                                */
/*****************************************************************************/

struct Embeddable: public Datacratic::TaggedEnum<Embeddable> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        NOT_EMBEDDABLE = 0,
        EMBEDDABLE = 1
    };
};

/*****************************************************************************/
/* AUCTION TYPE                                                              */
/*****************************************************************************/

struct AuctionType: public Datacratic::TaggedEnum<AuctionType, 2> {
    enum Vals {
        UNSPECIFIED = -1,  ///< Not explicitly specified

        FIRST_PRICE = 1,
        SECOND_PRICE = 2
    };

    AuctionType(Vals val = UNSPECIFIED)
    {
        this->val = val;
    }
};


/*****************************************************************************/
/* BANNER                                                                    */
/*****************************************************************************/

/** 3.3.3 Banner Object

    The “banner” object must be included directly in the impression object
    if the impression offered for auction is display or rich media, or it
    may be optionally embedded in the video object to describe the companion
    banners available for the linear or non-linear video ad.  The banner 
    object may include a unique identifier; this can be useful if these IDs
    can be leveraged in the VAST response to dictate placement of the
    companion creatives when multiple companion ad opportunities of the same
    size are available on a page.
*/
struct Banner {
    ~Banner();

    ///< NOTE: RTBkit extension: support for multiple formats
    Datacratic::List<int> w;                     ///< Width of ad
    Datacratic::List<int> h;                     ///< Height of ad
    Datacratic::TaggedInt wmax;                  ///< max width of ad (OpenRTB 2.2)
    Datacratic::TaggedInt hmax;                  ///< max height of ad (OpenRTB 2.2)
    Datacratic::TaggedInt wmin;                  ///< min width of ad (OpenRTB 2.2)
    Datacratic::TaggedInt hmin;                  ///< min height of ad (OpenRTB 2.2)
    Datacratic::Id id;                           ///< Ad ID
    AdPosition pos;                  ///< Ad position (table 6.5)
    Datacratic::List<BannerAdType> btype;        ///< Blocked creative types (table 6.2)
    Datacratic::List<CreativeAttribute> battr;   ///< Blocked creative attributes (table 6.3)
    Datacratic::List<MimeType> mimes;            ///< Whitelist of content MIME types
    FramePosition topframe;          ///< Is it in the top frame (1) or an iframe (0)?
    Datacratic::List<ExpandableDirection> expdir;///< Expandable ad directions (table 6.11)
    Datacratic::List<ApiFramework> api;          ///< Supported APIs (table 6.4)
    Json::Value ext;                 ///< Extensions go here, new in OpenRTB 2.1
};


/*****************************************************************************/
/* VIDEO                                                                     */
/*****************************************************************************/

/** 3.3.4 Video Object

    The “video” object must be included directly in the impression object if
    the impression offered for auction is an in-stream video ad opportunity.  

    Note that for the video object, many of the fields are non-essential for
    a minimally viable exchange interfaces.  These parameters do not
    necessarily need to be specified to the bidder, if they are always the
    same for all impression, of if the exchange chooses not to supply the 
    additional information to the bidder.
*/
struct Video {
    ~Video();
    Datacratic::List<MimeType> mimes;       ///< Content MIME types supported
    VideoLinearity linearity;               ///< Whether it's linear or not (table 6.6)
    Datacratic::TaggedFloat minduration;      ///< Minimum ad duration in seconds
    Datacratic::TaggedFloat maxduration;      ///< Maximum ad duration in seconds
    VideoBidResponseProtocol protocol;  ///< Bid response protocols (table 6.7)
    Datacratic::List<VideoBidResponseProtocol> protocols; ///< Bid response protocols array (table 6.7) (OpenRTB2.2)
    Datacratic::TaggedInt w;                ///< Width of player in pixels
    Datacratic::TaggedInt h;                ///< Height of player in pixels
    ///< Starting delay in seconds for placement (table 6.9)
    Datacratic::TaggedIntDef<VideoStartDelay::UNSPECIFIED> startdelay;
    Datacratic::TaggedIntDef<1> sequence;   ///< Which ad number in the bid request
    Datacratic::List<CreativeAttribute> battr; ///< Which creative attributes are blocked
    Datacratic::TaggedIntDef<0> maxextended;///< Max extended video ad duration
    Datacratic::TaggedInt minbitrate;       ///< Minimum bitrate for ad in kbps
    Datacratic::TaggedInt maxbitrate;       ///< Maximum bitrate for ad in kbps
    Datacratic::TaggedBoolDef<1> boxingallowed;           ///< Is letterboxing allowed
    Datacratic::List<VideoPlaybackMethod> playbackmethod; ///< Available playback methods
    Datacratic::List<ContentDeliveryMethod> delivery;     ///< Available delivery methods
    AdPosition pos;             ///< Ad position (table 6.5)
    std::vector<Banner> companionad; ///< List of companion banners available
    Datacratic::List<ApiFramework> api;     ///< List of supported API frameworks (table 6.4)
    Datacratic::List<VastCompanionType> companiontype;    ///< VAST Companion Types (table 6.17)
    Json::Value ext;            ///< Extensions go here, new in OpenRTB 2.1
};


/*****************************************************************************/
/* PRODUCER / PUBLISHER                                                      */
/*****************************************************************************/

/** 3.3.8 Publisher Object

    The publisher object itself and all of its parameters are optional, so
    default values are not provided.  If an optional parameter is not
    specified, it should be considered unknown.
*/

struct Publisher {
    ~Publisher();
    Datacratic::Id id;                       ///< Unique ID representing the publisher
    Datacratic::UnicodeString name;             ///< Publisher name
    Datacratic::List<ContentCategory> cat; ///< Content categories     
    Datacratic::UnicodeString domain;               ///< Domain name of publisher
    Json::Value ext;             ///< Extensions go here, new in OpenRTB 2.1
};

/** 3.3.9 Producer Object

    The producer is useful when content where the ad is shown is syndicated,
    and may appear on a completely different publisher.  The producer object
    itself and all of its parameters are optional, so default values are not
    provided.  If an optional parameter is not specified, it should be 
    considered unknown.   This object is optional, but useful if the content
    producer is different from the site publisher.    
*/

typedef Publisher Producer;  /// They are the same...

/*****************************************************************************/
/* DEAL                                                                      */
/*****************************************************************************/

/** 3.3.17 Direct deals object

    A "deal" object constitutes a deal struck a priori between a buyer and a seller and indicates that
    this impression is available under the terms of that deal.
*/
struct Deal { // New in OpenRTB 2.2
    ~Deal();
    Datacratic::Id id;                      ///< unique id for a direct deal
    Datacratic::TaggedFloat bidfloor;       ///< bid floor for this impression in CPM of bidfloorcur
    std::string bidfloorcur;                ///< currency of the bidfloor, ISO-4217
    Datacratic::List<std::string> wseat;    ///< array of buyer seats allowed to bid on this deal
    Datacratic::List<std::string> wadomain; ///< array of advertiser domains allowed to bid on this deal
    Datacratic::TaggedInt at;               ///< type of auction : first / second price
    Json::Value ext;                        ///< Extension object
};


/*****************************************************************************/
/* PMP                                                                       */
/*****************************************************************************/

/** 3.3.16 PMP object

    The “pmp” object contains a parent object for usage within the context of private marketplaces 
    and the use of the RTB protocol to execute Direct Deals.
 
*/
struct PMP { // New in OpenRTB 2.2
    ~PMP();
    Datacratic::TaggedIntDef<0> privateAuction;    ///< Flag for private auction traffic : = 0 all bids, 1 = private deal
    std::vector<Deal> deals;   ///< List of deals eligible for this impression
    Json::Value ext;                ///< Extensions related to private deals between parties 
};

/*****************************************************************************/
/* IMPRESSION                                                                */
/*****************************************************************************/

/** 3.3.2 Impression Object 

    The “imp” object describes the ad position or impression being auctioned.
    A single bid request can include multiple “imp” objects, a use case for
    which might be an exchange that supports selling all ad positions on a
    given page as a bundle.  Each “imp” object has a required ID so that 
    bids can reference them individually.  An exchange can also conduct
    private auctions by restricting involvement to specific subsets of seats
    within bidders.
*/
struct Impression {
    ~Impression();
    Datacratic::Id id;                             ///< Impression ID within BR
    Datacratic::Optional<Banner> banner;           ///< If it's a banner ad
    Datacratic::Optional<Video> video;             ///< If it's a video ad
    Datacratic::UnicodeString displaymanager;          ///< What renders the ad
    Datacratic::UnicodeString displaymanagerver;        ///< What version of that thing
    Datacratic::TaggedBoolDef<0> instl;            ///< Is it interstitial
    Datacratic::UnicodeString tagid;                   ///< ad tag ID for auction
    Datacratic::TaggedDoubleDef<0> bidfloor;        ///< CPM bid floor
    std::string bidfloorcur;                ///< Bid floor currency
    Datacratic::TaggedInt secure;           ///< Flag that requires secure https assets (1 == yes) (OpenRTB 2.2)
    Datacratic::List<std::string> iframebuster;         ///< Supported iframe busters (for expandable/video ads)
    Datacratic::Optional<OpenRTB::PMP> pmp;        ///< Containing any Deals eligible for the impression object
    Json::Value ext;                   ///< Extended impression attributes
};


/*****************************************************************************/
/* CONTENT                                                                   */
/*****************************************************************************/

/** 3.3.7 Content Object

    The content object itself and all of its parameters are optional, so
    default values are not provided. If an optional parameter is not specified,
    it should be considered unknown.

    This object describes the content in which the impression will appear
    (may be syndicated or nonsyndicated content).  This object may be useful
    in the situation where syndicated content contains impressions and 
    does not necessarily match the publisher’s general content.  The exchange
    might or might not have knowledge of the page where the content is
    running, as a result of the syndication method.  (For example, video
    impressions embedded in an iframe on an unknown web property 
    or device.)
*/

struct Content {
    ~Content();
    Datacratic::Id id;                   ///< Unique ID identifying the content
    Datacratic::TaggedInt episode;       ///< Episode number of a series
    Datacratic::UnicodeString title;        ///< Content title
    Datacratic::UnicodeString series;       ///< Content series
    Datacratic::UnicodeString season;       ///< Content season
    Datacratic::Url url;                 ///< Original content URL
    Datacratic::List<ContentCategory> cat; ///< IAB content category (table 6.1)
    VideoQuality videoquality; ///< Video quality (table 6.14)
    Datacratic::CSList keywords;         ///< Content keywords
    Datacratic::UnicodeString contentrating;    ///< Content rating (eg Mature)
    Datacratic::UnicodeString userrating;       ///< Content user rating (eg 3 stars)
    Datacratic::UnicodeString context;          ///< Content context (table 6.13)
    Datacratic::TaggedBool livestream;   ///< Is this being live streamed?
    SourceRelationship sourcerelationship;  ///< 1 = direct, 0 = indirect
    Datacratic::Optional<Producer> producer;  ///< Content producer
    Datacratic::TaggedInt len;           ///< Length of content in seconds
    MediaRating qagmediarating;///< Media rating per QAG guidelines (table 6.18).
    Embeddable embeddable;   ///< 1 if embeddable, 0 otherwise
    std::string language;     ///< Content language.  ISO 639-1 (alpha-2).
    Json::Value ext;         ///< Extensions go here, new in OpenRTB 2.1
};


/*****************************************************************************/
/* CONTEXT                                                                   */
/*****************************************************************************/

/** Common information between a Site and App. */

struct Context {
    ~Context();
    Datacratic::Id id;        ///< Site ID on the exchange
    Datacratic::UnicodeString name;  ///< Site name
    Datacratic::UnicodeString domain;///< Site or app domain
    Datacratic::List<ContentCategory> cat;        ///< IAB content categories for site/app
    Datacratic::List<ContentCategory> sectioncat; ///< IAB content categories for subsection
    Datacratic::List<ContentCategory> pagecat;    ///< IAB content categories for page/view
    Datacratic::TaggedBool privacypolicy;           ///< Has a privacy policy
    Datacratic::Optional<Publisher> publisher;    ///< Publisher of the site or app
    Datacratic::Optional<Content> content;        ///< Content of the site or app
    Datacratic::CSList keywords;                    ///< Keywords describing app
    Json::Value ext;
};


/*****************************************************************************/
/* SITE                                                                      */
/*****************************************************************************/

/** 3.3.5 Site Object

    A site object should be included if the ad supported content is part of
    a website (as opposed to an application).  A bid request must not contain
    both a site object and an app object.

    The site object itself and all of its parameters are optional, so default
    values are not provided. If an optional parameter is not specified, it
    should be considered unknown.  At a minimum, it’s useful to provide a
    page URL or a site ID, but this is not strictly required.
*/

struct SiteInfo {
    Datacratic::Url page;          ///< URL of the page to be shown
    Datacratic::Url ref;           ///< Referrer URL that got user to page
    Datacratic::UnicodeString search; ///< Search string that got user to page
};

struct Site: public Context, public SiteInfo {
};


/*****************************************************************************/
/* APP                                                                       */
/*****************************************************************************/

/** 3.3.6 App Object

    An “app” object should be included if the ad supported content is part of
    a mobile application (as opposed to a mobile website).  A bid request
    must not contain both an “app” object and a “site” object.
    
    The app object itself and all of its parameters are optional, so default
    values are not provided. If an optional parameter is not specified, it
    should be considered unknown. .  At a minimum, it’s useful to provide an
    App ID or bundle, but this is not strictly required.
*/
struct AppInfo {
    std::string ver;             ///< Application version
    Datacratic::UnicodeString bundle;      ///< Application bundle name (unique across multiple exchanges)
    Datacratic::TaggedBool paid;        ///< Is a paid version of the app
    Datacratic::Url storeurl;           ///< For QAG 1.5 compliance, new in OpenRTB 2.1
};

struct App: public Context, public AppInfo {
};


/*****************************************************************************/
/* GEO                                                                       */
/*****************************************************************************/

/** 3.3.11 Geo Object

    The geo object itself and all of its parameters are optional, so default
    values are not provided. If an optional parameter is not specified, it
    should be considered unknown.

    Note that the Geo Object may appear in one or both the Device Object and
    the User Object.  This is intentional, since the information may be
    derived from either a device-oriented source (such as IP geo lookup), or
    by user registration information (for example provided to a publisher
    through a user registration). If the information is in conflict, it’s up
    to the bidder to determine which information to use.
*/

struct Geo {
    ~Geo();
    Datacratic::TaggedDouble lat;        ///< Latitude of user (-90 to 90; South negative)
    Datacratic::TaggedDouble lon;        ///< Longtitude (-180 to 180; west is negative)
    std::string country;         ///< Country code (ISO 3166-1 Alpha-3)
    std::string region;          ///< Region code (ISO 3166-2)
    std::string regionfips104;   ///< Region using FIPS 10-4
    std::string metro;           ///< Metropolitan region (Google Metro code)
    Datacratic::UnicodeString city;        ///< City name (UN Code for Trade and Transport Loc)
    Datacratic::UnicodeString zip;             ///< Zip or postal code
    LocationType type;      ///< Source of Geo data (table 6.15)
    Json::Value ext;        ///< Extensions go here, new in OpenRTB 2.1

    /// Datacratic extensions
    std::string dma;             ///< Direct Marketing Association code
    /// Rubicon extensions
    Datacratic::TaggedBool latlonconsent;  ///< Has user given consent for lat/lon use?
};


/*****************************************************************************/
/* DEVICE                                                                    */
/*****************************************************************************/

/** 3.3.10 Device Object

    The “device” object provides information pertaining to the device
    including its hardware, platform, location, and carrier.

    This device can refer to a mobile handset, a desktop computer, set
    top box or other digital device.

    The device object itself and all of its parameters are optional,
    so default values are not provided.

    If an optional parameter is not specified, it should be considered
    unknown.

    In general, the most essential fields are either the IP address
    (to enable geo-lookup for the bidder), or providing geo information
    directly in the geo object.
*/

struct Device {
    ~Device();
    Datacratic::TaggedBool dnt;        ///< If 1 then do not track is on
    Datacratic::UnicodeString ua;         ///< User agent of device
    std::string ip;             ///< IP address of device
    Datacratic::Optional<Geo> geo;     ///< Geolocation of device
    std::string didsha1;        ///< Device ID: SHA1
    std::string didmd5;         ///< Device ID: MD5
    std::string dpidsha1;       ///< Device Platform ID: SHA1
    std::string dpidmd5;        ///< Device Platform ID: MD5
    std::string macsha1;       ///< MAC ADDRESS: SHA1 (OpenRTB 2.2)
    std::string macmd5;        ///< MAC ADDRESS: MD5 (OpenRTB 2.2)
    std::string ipv6;           ///< IPv6 address
    Datacratic::UnicodeString carrier;    ///< Carrier or ISP (derived from IP address)
    Datacratic::UnicodeString language;   ///< Browser language.  ISO 639-1 (alpha-2).
    Datacratic::UnicodeString make;       ///< Device make
    Datacratic::UnicodeString model;      ///< Device model
    Datacratic::UnicodeString os;         ///< Device OS
    Datacratic::UnicodeString osv;         ///< Device OS version
    Datacratic::TaggedBool js;         ///< Javascript is supported? 1 or 0
    ConnectionType connectiontype;    ///< Connection type (table 6.10)
    DeviceType devicetype; ///< Device type (table 6.16)
    std::string flashver;   ///< Flash on device
    std::string ifa;        ///< Native identifier for advertisers
    Json::Value ext;       ///< Extensions go here
};


/*****************************************************************************/
/* SEGMENT                                                                   */
/*****************************************************************************/

/** 3.3.14 Segment Object

    The data and segment objects together allow data about the user to be
    passed to bidders in the bid request.  Segment objects convey specific
    units of information from the provider identified in the parent data
    object.

    The segment object itself and all of its parameters are optional, so
    default values are not provided; if an optional parameter is not
    specified, it should be considered unknown.
*/
struct Segment {
    Datacratic::Id id;                         ///< Segment ID
    Datacratic::UnicodeString name;                   ///< Segment name
    Datacratic::UnicodeString value;                  ///< Segment value
    Json::Value ext;               ///< Extensions go here, new in OpenRTB 2.1

    /// Datacratic Extensions
    Datacratic::TaggedDouble segmentusecost;    ///< Cost of using segment in CPM
};


/*****************************************************************************/
/* DATA                                                                      */
/*****************************************************************************/

/** 3.3.13 Data Object

    The data and segment objects together allow data about the user to be
    passed to bidders in the bid request.  This data may be from multiple
    sources (e.g., the exchange itself, third party providers) as specified
    by the data object ID field.  A bid request can mix data objects from 
    multiple providers.

    The data object itself and all of its parameters are optional, so
    default values are not provided.  If an optional parameter is not
    specified, it should be considered unknown.
*/
struct Data {
    Datacratic::Id id;                           ///< Exchange specific data prov ID
    Datacratic::UnicodeString name;                  ///< Data provider name
    std::vector<Segment> segment;         ///< Segment of data
    Json::Value ext;                 ///< Extensions go here, new in OpenRTB 2.1

    /// Datacratic Extensions
    std::string usecostcurrency;          ///< Currency of use cost
    Datacratic::TaggedDouble datausecost;         ///< Cost of using the data (CPM)
};


/*****************************************************************************/
/* USER                                                                      */
/*****************************************************************************/

/** 3.3.12 User Object

    The “user” object contains information known or derived about the
    human user of the device.  Note that the user ID is an exchange
    artifact (refer to the “device” object for hardware or platform
    derived IDs) and may be subject to rotation policies. However, this
    user ID must be stable long enough to serve reasonably as the basis
    for frequency capping.

    The user object itself and all of its parameters are optional, so
    default values are not provided.  If an optional parameter is not
    specified, it should be considered unknown.
    
    If device ID is used as a proxy for unique user ID, use the device
    object.
*/
struct User {
    ~User();
    Datacratic::Id id;                     ///< Exchange-specific user ID
    Datacratic::Id buyeruid;               ///< Exchange seat-specific user ID
    Datacratic::TaggedInt yob;             ///< Year of birth
    std::string gender;             ///< Gender: Male, Female, Other
    Datacratic::CSList keywords;           ///< List of keywords of consumer intent
    Datacratic::UnicodeString customdata;         ///< Custom data from exchange
    Datacratic::Optional<Geo> geo;                   ///< Geolocation of user at registration
    std::vector<Data> data;         ///< User data segments
    Json::Value ext;           ///< Extensions go here, new in OpenRTB 2.1

    /// Rubicon extensions
    Datacratic::TaggedInt tz;              ///< User time zone in seconds after GMT
    Datacratic::TaggedInt sessiondepth;    ///< User session depth
};

/*****************************************************************************/
/* REGULATIONS                                                               */
/*****************************************************************************/

/** 3.3.15 Regulations object
  
    The “regs” object contains any legal, governmental, or industry regulations that apply to the 
    request. The first regulation added signal whether or not the request falls under the United States 
    Federal Trade Commission’s regulations for the United States Children’s Online Privacy 
    Protection Act (“COPPA”). See the COPPA appendix for details.
    
    The regs object itself and all of its parameters are optional, so default values are not provided. 
    If an optional parameter is not specified, it should be considered unknown.

*/
struct Regulations { // New in OpenRTB 2.2
    ~Regulations();
    Datacratic::TaggedIntDef<0> coppa;    ///< Flag for coppa regulated traffic : = 0 no, 1 = yes
    Json::Value ext;                      ///< Extensions related to regulations 
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
    ~BidRequest();
    Datacratic::Id id;                             ///< Bid request ID
    std::vector<Impression> imp;            ///< List of impressions
    //unique_ptr<Context> context;     // TODO: factor out of site and app
    Datacratic::Optional<Site> site;
    Datacratic::Optional<App> app;
    Datacratic::Optional<Device> device;
    Datacratic::Optional<User> user;
    AuctionType at;                    ///< Auction type (1=first/2=second party)
    Datacratic::TaggedInt tmax;                    ///< Max time avail in ms
    std::vector<std::string> wseat;              ///< Allowed buyer seats
    Datacratic::TaggedBool allimps;                ///< All impressions in BR (for road-blocking)
    std::vector<std::string> cur;                ///< Allowable currencies
    Datacratic::List<ContentCategory> bcat;        ///< Blocked advertiser categories (table 6.1)
    std::vector<Datacratic::UnicodeString> badv;           ///< Blocked advertiser domains
    Datacratic::Optional<Regulations> regs; ///< Regulations Object list (OpenRTB 2.2)
    Json::Value ext;                   ///< Protocol extensions
    Json::Value unparseable;           ///< Unparseable fields get put here
};


/*****************************************************************************/
/* BID                                                                       */
/*****************************************************************************/

/** 4.3.3 Bid Object

    For each bid, the “nurl” attribute contains the win notice URL.  If the
    bidder wins the impression, the exchange calls this notice URL a) to
    inform the bidder of the win and b) to convey certain information using
    substitution macros (see Section 4.6 Substitution Macros).

    The “adomain” attribute can be used to check advertiser block list
    compliance. The “iurl” attribute can provide a link to an image that
    is representative of the campaign’s content (irrespective of whether
    the campaign may have multiple creatives). This enables human review for
    spotting inappropriate content. The “cid” attribute can be used to block
    ads that were previously identified as inappropriate; essentially a safety
    net beyond the block lists. The “crid” attribute can be helpful in
    reporting creative issues back to bidders. Finally, the “attr” array 
    indicates the creative attributes that describe the ad to be served.

    BEST PRACTICE: Substitution macros may allow a bidder to use a static
    notice URL for all of its bids. Thus, exchanges should offer the option
    of a default notice URL that can be preconfigured per bidder to reduce
    redundant data transfer.
*/

struct Bid {
    Datacratic::Id id;                        ///< Bidder's bid ID to identify bid
    Datacratic::Id impid;                     ///< ID of the impression we're bidding on
    Datacratic::TaggedDouble price;           ///< Price to bid
    Datacratic::Id adid;                      ///< Id of ad to be served if won
    Datacratic::UnicodeString nurl;                  ///< Win notice/ad markup URL
    Datacratic::UnicodeString adm;                   ///< Ad markup
    std::vector<std::string> adomain;       ///< Advertiser domains
    Datacratic::UnicodeString iurl;                  ///< Image URL for content checking
    Datacratic::Id cid;                       ///< Campaign ID
    Datacratic::Id crid;                      ///< Creative ID
    Datacratic::List<CreativeAttribute> attr; ///< Creative attributes
    std::string dealid;                     ///< unique id for the deal associated with bid
                                            ///< if its in bid request, required in bid response
    Datacratic::TaggedInt w;                ///< width of ad in pixels
    Datacratic::TaggedInt h;                ///< height of ad in pixels
    Json::Value ext;              ///< Extended bid fields
};


/*****************************************************************************/
/* SEAT BID                                                                  */
/*****************************************************************************/

/** 4.3.2 Seat Bid Object

    A bid response can contain multiple “seatbid” objects, each on behalf of
    a different bidder seat.   Since a bid request can include multiple
    impressions, each “seatbid” object can contain multiple bids each
    pertaining to a different impression on behalf of a seat. Thus, each
    “bid” object must include the impression ID to which it pertains as well
    as the bid price. The “group” attribute can be used to specify if a seat
    is willing to accept any impressions that it can win (default) or if it is 
    only interested in winning any if it can win them all (i.e., all or
    nothing).
*/

struct SeatBid {
    std::vector<Bid> bid;              ///< Array of bid objects  (relating to imps)
    Datacratic::Id seat;                      ///< Seat on behalf of whom the bid is made
    Datacratic::TaggedInt group;              ///< If true, imps must be won as a group
    Json::Value ext;              ///< Extension fields
};


/*****************************************************************************/
/* BID RESPONSE                                                              */
/*****************************************************************************/

/** 4.3.1 Bid Response Object

    The top-level bid response object is defined below. The “id” attribute
    is a reflection of the bid request ID for logging purposes. Similarly,
    “bidid” is an optional response tracking ID for bidders. If specified,
    it can be included in the subsequent win notice call if the bidder wins.
    At least one “seatbid” object is required, which contains a bid on at
    least one impression. Other attributes are optional since an exchange
    may establish default values.

    No-Bids on all impressions should be indicated as a HTTP 204 response.
    For no-bids on specific impressions, the bidder should omit these from
    the bid response.
*/

struct BidResponse {
    Datacratic::Id id;
    std::vector<SeatBid> seatbid;
    Datacratic::Id bidid;
    std::string cur;
    Datacratic::UnicodeString customData;
    NoBidReason nbr; ///< reason for not bidding
    Json::Value ext;
};

} // namespace OpenRTB


/******************************************************************************/
/* HASH                                                                       */
/******************************************************************************/

namespace std {

template<>
struct hash<OpenRTB::AdPosition>
{
    size_t operator() (OpenRTB::AdPosition obj) const
    {
        return std::hash<int>()(static_cast<int>(obj.val));
    }
};

} // namespace std
