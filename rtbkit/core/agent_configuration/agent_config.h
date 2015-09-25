/* rtb_agent_config.h                                             -*- C++ -*-
   Jeremy Barnes, 24 March 2011
   Configuration for an RTB agent.
*/

#ifndef __rtb_agent_config_h__
#define __rtb_agent_config_h__

#include <string>
#include <vector>
#include <set>
#include "jml/arch/spinlock.h"
#include "soa/jsoncpp/json.h"
#include <boost/regex.hpp>
#include <boost/regex/icu.hpp>
#include "rtbkit/common/bid_request.h"
#include "include_exclude.h"
#include "fees.h"
#include "rtbkit/common/account_key.h"
#include "rtbkit/core/agent_configuration/latlonrad.h"

namespace RTBKIT {


struct BiddableSpots;
struct AgentStats;
struct ExchangeConnector;


/*****************************************************************************/
/* CREATIVE                                                                  */
/*****************************************************************************/

/** Describes a creative that an agent has available.

    A creative is a general-purpose entity that can represent either an image
    or a video
 */

struct Creative {

    enum class Type { Image, Video };

    Creative(int width = 0, int height = 0, std::string name = "", int id = -1,
            const std::string dealId = "");

    // Three samples that can be used for testing...
    static const Creative sampleLB;
    static const Creative sampleBB;
    static const Creative sampleWS;

    void fromJson(const Json::Value & val);
    Json::Value toJson() const;

    Format format;

    /// Purely for information (used internally)
    std::string name;
    int id;

    /// Video creative information
    uint32_t duration;
    uint64_t bitrate;

    /// Configuration values; per provider
    /// eg: OpenRTB, ...
    Json::Value providerConfig;

    /// lock for the provider data
    mutable ML::Spinlock lock;

    /// List of provider-specific creative data
    std::map<std::string, std::shared_ptr<void> > providerData;

    template<typename T>
    const T * getProviderData(const std::string & provider) const
    {
        auto it = providerData.find(provider);
        if (it == providerData.end())
            throw ML::Exception("provider data for " + provider + " not found");
        if (it->second.get() == nullptr)
            throw ML::Exception("provider data for " + provider + " is null");
        
        return reinterpret_cast<const T *>(it->second.get());
    }

    /// Tags set on the creative for elibibility filtering
    Tags tags;

    /// Filter to filter the creative against campaign eligibility
    TagFilterExpression eligibilityFilter;

    IncludeExclude<std::string> languageFilter;
    IncludeExclude<CachedRegex<boost::u32regex, Datacratic::UnicodeString> > locationFilter;
    IncludeExclude<std::string> exchangeFilter;
   
    // Needed for PMP filter
    std::string dealId;

    // Fees
    shared_ptr<Fees> fees;

    struct SegmentInfo {

        SegmentInfo() : excludeIfNotPresent(false){}

        bool excludeIfNotPresent;
        SegmentList include;
        SegmentList exclude;

        void fromJson(const Json::Value & val);

        Json::Value toJson() const;

        IncludeExcludeResult process(const SegmentList & segments) const;

    };

    std::map<std::string, SegmentInfo> segments;

    /** Is the given ad spot compatible with the given creative format? */
    bool compatible(const AdSpot & spot) const;

    /** Is this creative biddable on the given exchange and protocol version? */
    bool biddable(const std::string & exchange,
                  const std::string & protocolVersion) const;

    /** Is this an image creative ? */
    bool isImage() const;

    /** Is this  a video creative ? */
    bool isVideo() const;

    static Creative image(int width, int height, std::string name = "", int id = -1, std::string dealId = "");
    static Creative video(int width, int height, uint32_t duration, uint64_t bitrate,
                std::string name = "", int id = -1, std::string dealId = "");

private:
    Type type;

    std::string typeString() const;

};


/*****************************************************************************/
/* USER PARTITION                                                            */
/*****************************************************************************/

/** Describes a way in which users are partitioned consistently in order to
    allow for A/B testing.
*/

struct UserPartition {
    UserPartition();

    void swap(UserPartition & other);

    void clear();

    enum HashOn {
        NONE,        ///< Hash always returns zero
        RANDOM,      ///< Random number

        EXCHANGEID,  ///< Hash on md5(exchange ID)
        PROVIDERID,  ///< Hash on md5(provider ID)
        IPUA,        ///< hash on md5(IP + UserAgent) (no delimiter)
    } hashOn;

    int modulus;     ///< Max value of hash that's achievable

    struct Interval {
        Interval(int first = 0, int last = 0)
            : first(first), last(last)
        {
        }

        int first, last;

        bool in(int val) const
        {
            return val >= first && val < last;
        }

        Json::Value toJson() const;
    };

    bool empty() const
    {
        return hashOn == NONE && modulus == 1 && includeRanges.size() == 1
            && includeRanges[0].first == 0 && includeRanges[0].last == 1;
    }

    /** A list of the hash ranges that are accepted. */
    std::vector<Interval> includeRanges;

    /** Return true if the user matches the user partition. */
    bool matches(const UserIds & ids,
                 const std::string& ip,
                 const Datacratic::UnicodeString& userAgent) const;

    /** Parse from JSON. */
    void fromJson(const Json::Value & json);

    /** Return as JSON */
    Json::Value toJson() const;
};


/******************************************************************************/
/* AUGMENTATION CONFIG                                                        */
/******************************************************************************/

/** Configuration for a given augmentor desired by an agent. */
struct AugmentationConfig
{
    AugmentationConfig(const std::string& name = "") :
        name(name), required(false)
    {}

    std::string name;
    Json::Value config;
    IncludeExclude<std::string> filters;
    bool required;

    bool operator < (const AugmentationConfig & other) const
    {
        return name < other.name;
    }

    Json::Value toJson() const;
    void fromJson(const Json::Value&);

    static AugmentationConfig createFromJson(
            const Json::Value& json, const std::string& name = "");
};



/*****************************************************************************/
/* BLACKLIST CONTROL                                                         */
/*****************************************************************************/

enum BlacklistType {
    BL_OFF,       ///< Don't blacklist
    BL_USER,      ///< Blacklist the user
    BL_USER_SITE  ///< Blacklist the user on the given site
};

enum BlacklistScope {
    BL_AGENT,      ///< Blacklist for the agent
    BL_ACCOUNT,    ///< Blacklist for an account
};


/*****************************************************************************/
/* BID CONTROL TYPE                                                          */
/*****************************************************************************/

enum BidControlType {
    BC_RELAY,       ///< Relay to agent which will compute the price
    BC_RELAY_FIXED, ///< Relay to agent but bid fixed price
    BC_FIXED        ///< Bid fixed price and don't relay
};


/*****************************************************************************/
/* BID RESULT FORMAT                                                         */
/*****************************************************************************/

enum BidResultFormat {
    BRF_FULL,         ///< Full message
    BRF_LIGHTWEIGHT,  ///< Lightweight message
    BRF_NONE          ///< No message
};

Json::Value toJson(BidResultFormat fmt);
void fromJson(BidResultFormat & fmt, const Json::Value & j);

/*****************************************************************************/
/* AGENT CONFIG                                                              */
/*****************************************************************************/

/** Describes the configuration state of an RTB agent.  Passed through by
    a agent to the router to describe how the routes should be set up.
*/
struct AgentConfig {
    AgentConfig();

    static AgentConfig createFromJson(const Json::Value & json);

    void parse(const std::string & jsonStr);
    void fromJson(const Json::Value & json);

    Json::Value toJson(bool includeCreatives = true) const;

    AccountKey account;   ///< Who to bill this to

    uint64_t externalId;  ///< Simplifies id reconciliation with external systems

    bool external;        ///< Forward bid request that have this configuration
    bool test;            ///< Can't make real bids
    
    std::string roundRobinGroup;
    int roundRobinWeight;

    float bidProbability;
    float minTimeAvailableMs;

    int maxInFlight;

    std::string bidderInterface;

    std::vector<std::string> requiredIds;

    IncludeExclude<DomainMatcher> hostFilter;
    IncludeExclude<CachedRegex<boost::regex, std::string> > urlFilter;
    IncludeExclude<CachedRegex<boost::regex, std::string> > languageFilter;
    IncludeExclude<CachedRegex<boost::u32regex, Datacratic::UnicodeString> > locationFilter;
    LatLonRadList latLongDevFilter; // latitude and longitude device filter


    struct SegmentInfo {
        SegmentInfo()
            : excludeIfNotPresent(false)
        {
        }

        bool excludeIfNotPresent;
        SegmentList include;
        SegmentList exclude;

        /** What exchanges is this filter applied to?  If the exchange
            is excluded by the filter, then the filter is bypassed. */
        IncludeExclude<std::string> applyToExchanges;
        
        IncludeExcludeResult process(const SegmentList & segments) const;
        
        void fromJson(const Json::Value & val);
        Json::Value toJson() const;
    };

    std::map<std::string, SegmentInfo> segments;

    IncludeExclude<std::string> exchangeFilter;

    IncludeExclude<OpenRTB::AdPosition> foldPositionFilter;

    SegmentInfo tagFilter;

    struct HourOfWeekFilter {

        HourOfWeekFilter();

        bool isIncluded(Date auctionDate) const;

        bool isDefault() const;  // true if all hours are 1

        void fromJson(const Json::Value & val);
        Json::Value toJson() const;

        std::bitset<168> hourBitmap;
    };
    
    HourOfWeekFilter hourOfWeekFilter;

    UserPartition userPartition;

    std::vector<Creative> creatives;

    BlacklistType blacklistType;
    BlacklistScope blacklistScope;
    double blacklistTime;
    
    bool hasBlacklist() const
    {
        return blacklistType != BL_OFF && blacklistTime > 0.0;
    }

    BidControlType bidControlType;
    uint32_t fixedBidCpmInMicros;

    /** Add the given augmentation to the list of augmentations.  This will
        fail if the given augmentation already exists in the list.
    */
    void addAugmentation(const std::string & name,
                         Json::Value config = Json::Value());

    /** Add the given augmentation to the list of augmentations.  This will
        fail if the given augmentation already exists in the list.
    */
    void addAugmentation(AugmentationConfig info);

    std::vector<AugmentationConfig> augmentations;

    /** JSON value that is passed through with each bid. */
    Json::Value providerConfig;

    /// lock for the provider data
    mutable ML::Spinlock lock;

    /// List of provider-specific creative data
    std::map<std::string, std::shared_ptr<void> > providerData;

    template<typename T>
    const T * getProviderData(const std::string & provider) const
    {
        auto it = providerData.find(provider);
        if (it == providerData.end())
            throw ML::Exception("provider data for " + provider + " not found");
        if (it->second.get() == nullptr)
            throw ML::Exception("provider data for " + provider + " is null");
        
        return reinterpret_cast<const T *>(it->second.get());
    }

    /** List of channels for which we subscribe to post impression
        visit events.
    */
    SegmentList visitChannels;

    /** Do we include visits not matched to a conversion? */
    bool includeUnmatchedVisits;

    /** Message formats */
    BidResultFormat winFormat, lossFormat, errorFormat;

    /** custom extensions */
    Json::Value ext;
};


} // namespace RTBKIT

#endif /* __rtb_agent_config_h__ */
