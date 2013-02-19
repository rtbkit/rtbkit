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


/*****************************************************************************/
/* AD SPOT                                                                   */
/*****************************************************************************/

/** Information about an ad spot that can be bid on. */

struct AdSpot {
    enum Position {NONE, ABOVE_FOLD, BELOW_FOLD};
    AdSpot(const Id & id = Id(), int reservePrice = 0);
        
    void fromJson(const Json::Value & val);
    Json::Value toJson() const;
    std::string toJsonStr() const;
    static AdSpot createFromJson(const Json::Value & json);
    std::string positionToStr() const ;
    static Position stringToPosition(const std::string &pos) ;
    static std::string positionToStr(Position pos);

    Id id;
    FormatSet formats;
    int reservePrice;

    Position position;
    std::string format() const;
    std::string firstFormat() const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(AdSpot);

void jsonParse(const Json::Value & value, AdSpot::Position & pos);
Json::Value jsonPrint(const AdSpot::Position & pos);


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


/*****************************************************************************/
/* LOCATION                                                                  */
/*****************************************************************************/

struct Location {
    Location()
        : cityName(""),dma(-1), timezoneOffsetMinutes(-1)
    {
    }

    std::string countryCode;
    std::string regionCode;
    Utf8String cityName;
    std::string postalCode;

    int dma;
    int timezoneOffsetMinutes;

    static Location createFromJson(const Json::Value & json);

    /** Return a location string with COUNTRY:REGION:CITY:POSTAL:DMA */
    Utf8String fullLocationString() const;

    /** Return a canonical JSON version of the bid request. */
    Json::Value toJson() const;

    /** Return a canonical stringified JSON version of the bid request. */
    std::string toJsonStr() const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};

IMPL_SERIALIZE_RECONSTITUTE(Location);


/*****************************************************************************/
/* BID REQUEST                                                               */
/*****************************************************************************/

struct BidRequest {
    BidRequest()
        : timestamp(0), isTest(false)
    {
    }

    Id auctionId;
    std::string language;
    std::string protocolVersion;
    std::string exchange;
    std::string provider;
    double timestamp;
    bool isTest;

    Location location;
    UserIds userIds;
    std::vector<AdSpot> spots;

    Url url;

    std::string ipAddress;
    std::string userAgent;

    SegmentsBySource restrictions;
    SegmentsBySource segments;
    
    Json::Value meta;
    Json::Value creative;

    /** Amount of extras that will be paid if we win the auction. */
    LineItems winSurcharges;

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
        for (unsigned i = 0; i < spots.size();  ++i)
            if (spots[i].id == adSpotId)
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

    /** Register the given parser for the bid request.  Should be done
        in a static initilalizer on shared library load.
    */
    static void registerParser(const std::string & source,
                               Parser parser);

    /** Parse the given bid request from the given source.  The correct
        parser will be looked up in a registry based upon the source.
    */
    static BidRequest *
    parse(const std::string & source, const std::string & bidRequest);

    static BidRequest *
    parse(const std::string & source, const Utf8String & bidRequest);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string serializeToString() const;
    static BidRequest createFromString(const std::string & str);
};

IMPL_SERIALIZE_RECONSTITUTE(BidRequest);

} // namespace RTBKIT


#endif /* __rtb__auction_h__ */
