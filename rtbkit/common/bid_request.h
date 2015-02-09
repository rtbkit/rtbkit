/* bid_request.h                                                    -*- C++ -*-
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Bid request class.
*/

#ifndef __rtb__bid_request_h__
#define __rtb__bid_request_h__

#include "soa/jsoncpp/json.h"
#include "soa/types/date.h"
#include "soa/types/string.h"
#include "jml/arch/exception.h"
#include "jml/utils/compact_vector.h"
#include "jml/utils/less.h"
#include <boost/function.hpp>
#include "soa/types/id.h"
#include "soa/types/url.h"
#include "rtbkit/common/segments.h"
#include <set>
#include "rtbkit/common/currency.h"
#include "tags.h"
#include "rtbkit/openrtb/openrtb.h"
#include "rtbkit/common/plugin_interface.h"

namespace RTBKIT {

using namespace Datacratic;

typedef ML::compact_vector<uint16_t, 5, uint16_t> SmallIntVector;
typedef ML::compact_vector<uint32_t, 3, uint32_t> IntVector;


/*****************************************************************************/
/* FORMAT                                                                    */
/*****************************************************************************/

/** Tiny structure describing the format of an ad. */

struct Format {
    Format(int16_t width = -1, int16_t height = -1)
        : width(width), height(height)
    {
    }

    bool valid() const
    {
        return width > 0 && height > 0;
    }

    Json::Value getWidthsJson() const;
    Json::Value getHeightsJson() const;
    void fromJson(const Json::Value & val);
    void fromString(const std::string &val);
    Json::Value toJson() const;
    std::string toJsonStr() const;
    std::string print() const;

    bool operator == (const Format & other) const
    {
        return width == other.width && height == other.height;
    }
    
    bool operator != (const Format & other) const
    {
        return ! operator == (other);
    }

    bool operator < (const Format & other) const
    {
        return ML::less_all(width, other.width,
                            height, other.height);
    }

    int16_t width;
    int16_t height;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(Format);


/*****************************************************************************/
/* FORMAT SET                                                                */
/*****************************************************************************/

struct FormatSet : public ML::compact_vector<Format, 3, uint16_t> {

    bool compatible(const Format & format) const
    {
        for (auto it = begin(), end = this->end();  it != end;  ++it)
            if (*it == format) return true;
        return false;
    }

    void fromJson(const Json::Value & val);
    Json::Value toJson() const;
    std::string toJsonStr() const;
    std::string print() const;

    void sort();

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(FormatSet);

ValueDescriptionT<RTBKIT::FormatSet> *
getDefaultDescription(RTBKIT::FormatSet * = 0);



/*****************************************************************************/
/* AD SPOT                                                                   */
/*****************************************************************************/

/** Information about an ad spot that can be bid on. */

struct AdSpot: public OpenRTB::Impression {
    AdSpot()
    {
    }

    AdSpot(OpenRTB::Impression && imp)
        : OpenRTB::Impression(std::move(imp))
    {
    }

    void fromJson(const Json::Value & val);
    Json::Value toJson() const;
    std::string toJsonStr() const;
    static AdSpot createFromJson(const Json::Value & json);

    std::string format() const;
    std::string firstFormat() const;

    /// Derived set of formats for the creative
    FormatSet formats;

    /// Fold position (deprecated)
    OpenRTB::AdPosition position;

    /// Minimum price for the bid request (deprecated)
    Amount reservePrice;

    /// This field indicates the segments for filtering of creatives
    SegmentsBySource restrictions;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(AdSpot);


/*****************************************************************************/
/* ID DOMAIN                                                                 */
/*****************************************************************************/

/** A set of IDs broken out that can be accessed very efficiently. */
enum IdDomain {
    ID_PROVIDER,
    ID_EXCHANGE,
    ID_MAX
};


/*****************************************************************************/
/* USER IDS                                                                  */
/*****************************************************************************/

/** Information known about a user and passed in as part of the bid */

struct UserIds : public std::map<std::string, Id> {

    void add(const Id & id, IdDomain domain);
    void add(const Id & id, const std::string & domain);
    void add(const Id & id, const std::string & domain, IdDomain domain2);

    void set(const Id & id, const std::string & domain);
    
    // These are always present
    Id exchangeId;
    Id providerId;

    /** Return a canonical JSON version of the bid request. */
    Json::Value toJson() const;

    /** Return a canonical stringified JSON version of the bid request. */
    std::string toJsonStr() const;

    /** Return an array of all the userIds as strings without xchg or prov key. */
    Json::Value toJsonArray() const;

    std::string toString() const
    {
        return toJsonStr();
    }

    static UserIds createFromJson(const Json::Value & json);

    /** Update the static entry belonging to a given domain. */
    void setStatic(const Id & id, const std::string & domain);
    void setStatic(const Id & id, IdDomain domain);

    static const char * domainToString(IdDomain domain);

    std::string serializeToString() const;
    static UserIds createFromString(const std::string & str);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(UserIds);

ValueDescriptionT<RTBKIT::UserIds> *
getDefaultDescription(RTBKIT::UserIds * = 0);



/*****************************************************************************/
/* LOCATION                                                                  */
/*****************************************************************************/

struct Location {
    Location()
        : cityName(""),dma(-1),metro(-1), timezoneOffsetMinutes(-1)
    {
    }

    std::string countryCode;
    std::string regionCode;
    Datacratic::UnicodeString cityName;
    Datacratic::UnicodeString postalCode;

    int dma;
    int metro;
    int timezoneOffsetMinutes;

    static Location createFromJson(const Json::Value & json);

    /** Return a location string with COUNTRY:REGION:CITY:POSTAL:DMA:METRO */
    Datacratic::UnicodeString fullLocationString() const;

    /** Return a canonical JSON version of the bid request. */
    Json::Value toJson() const;

    /** Return a canonical stringified JSON version of the bid request. */
    std::string toJsonStr() const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(Location);


using OpenRTB::AuctionType;

/*****************************************************************************/
/* BID REQUEST                                                               */
/*****************************************************************************/

struct BidRequest {
    BidRequest()
        : auctionType(AuctionType::SECOND_PRICE), timeAvailableMs(0.0),
          isTest(false) 
    {
    }

    Id auctionId;
    AuctionType auctionType;
    double timeAvailableMs;
    Date timestamp;
    bool isTest;

    std::string protocolVersion;  ///< What protocol version, eg OpenRTB
    std::string exchange;
    std::string provider;
    Id userAgentIPHash;  ///< Concatenation of the user agent and IP hashed

    /* The following fields indicate the contents of the OpenRTB bid request
       that is being processed.
    */

    /** Information specific to the site that generated the request.  Only
        one of site or app will be present.
    */
    OpenRTB::Optional<OpenRTB::Site> site;

    /** Information specific to the app that generated the request.  Only
        one of site or app will be present.
    */
    OpenRTB::Optional<OpenRTB::App> app;

    /** Information about the device that generated the request. */
    OpenRTB::Optional<OpenRTB::Device> device;

    /** Information about the user that generated the request. */
    OpenRTB::Optional<OpenRTB::User> user;

    /** The impressions that are available within the bid request. */
    std::vector<AdSpot> imp;

    /** The regulations that are related to this bid request. */
    OpenRTB::Optional<OpenRTB::Regulations> regs;

    /* The following fields are all mirrored from the information in the rest
       of the bid request.  They provide a way for the bid request parser to
       indicate the value of commonly used values in such a way that any
       optimization algorithm can make use of them.
    */
       
    Datacratic::UnicodeString language;   ///< User's language.
    Location location;      ///< Best available location information
    Url url;
    std::string ipAddress;
    Datacratic::UnicodeString userAgent;

    /** This field should be used to indicate what User IDs are available
        in the bid request.  These are normally used by the augmentors to
        attach first or third party data to the bid request.
    */
    UserIds userIds;

    SegmentsBySource restrictions;


    /** This field indicates the segments that are available in the bid
        request for the user.
    */
    SegmentsBySource segments;
    
    Json::Value meta;

    /** Extra fields included in the JSON that are unparseable by the bid
        request parser.  Recorded here so that no information is lost in the
        round trip.
    */
    Json::Value unparseable;

    /** Set of currency codes in which the bid can occur. */
    std::vector<CurrencyCode> bidCurrency;

    /** Blocked Categories. */
    OpenRTB::List<OpenRTB::ContentCategory> blockedCategories;

    /** Blocked TLD Advertisers (badv) */
    std::vector<Datacratic::UnicodeString> badv ;

    /** Amount of extras that will be paid if we win the auction.  These will
        be accumulated in the banker against the winning account.
    */
    LineItems winSurcharges;

    /** Transposition of the "ext" field of the OpenRTB request */
    Json::Value ext;

    /** Return a canonical JSON version of the bid request. */
    Json::Value toJson() const;

    /** Return a canonical stringified JSON version of the bid request. */
    std::string toJsonStr() const;

    /** Create a new BidRequest from a canonical JSON value. */
    static BidRequest createFromJson(const Json::Value & json);

    /** Return the ID for the given domain. */
    Id getUserId(IdDomain domain) const;
    Id getUserId(const std::string & domain) const;

    /** Return the spot number with the given ID.  -1 on not found. */
    int findAdSpotIndex(const Id & adSpotId) const
    {
        for (unsigned i = 0; i < imp.size();  ++i)
            if (imp[i].id == adSpotId)
                return i;
        return -1;
    }
    
    /** Query the presence of the given segment in the given source. */
    SegmentResult
    segmentPresent(const std::string & source,
                   const std::string & segment) const;

    /** Query the presence of the given segment in the given source. */
    SegmentResult
    segmentPresent(const std::string & source, int segment) const;
    
    void sortAll();

    typedef boost::function<BidRequest * (const std::string)> Parser;

    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer  
    // so any use of it should be considered deprecated
    static void registerParser(const std::string & source,
			       Parser parser)
    {
        PluginInterface<BidRequest>::registerPlugin(source, parser);
    }
  
    /** plugin interface requirements */
    typedef Parser Factory; // plugin interface expects this tipe to be called Factory
  
    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "bid_request";};

  
    /** Parse the given bid request from the given source.  The correct
        parser will be looked up in a registry based upon the source.*/
    static BidRequest *
    parse(const std::string & source, const std::string & bidRequest);

    static BidRequest *
    parse(const std::string & source, 
          const Datacratic::UnicodeString & bidRequest);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string serializeToString() const;
    static BidRequest createFromString(const std::string & str);
};

IMPL_SERIALIZE_RECONSTITUTE(BidRequest);

} // namespace RTBKIT

namespace Datacratic {

template<>
struct DefaultDescription<RTBKIT::Location>
    : public StructureDescription<RTBKIT::Location> {

    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::Format>
    : public StructureDescription<RTBKIT::Format> {

    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::AdSpot>
    : public StructureDescription<RTBKIT::AdSpot> {

    DefaultDescription();
};

template<>
struct DefaultDescription<RTBKIT::BidRequest>
    : public StructureDescription<RTBKIT::BidRequest> {

    DefaultDescription();
};


} // namespace Datacratic

#endif /* __rtb__auction_h__ */
