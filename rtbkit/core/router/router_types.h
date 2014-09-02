/* rtb_router_types.h                                              -*- C++ -*-
   Jeremy Barnes, 1 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Types for the rtb router.
*/

#ifndef __rtb_router__rtb_router_types_h__
#define __rtb_router__rtb_router_types_h__

#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/auction.h"
#include "jml/stats/distribution.h"
#include <set>
#include "rtbkit/common/currency.h"
#include "rtbkit/common/bids.h"


namespace RTBKIT {

struct AgentConfig;


/*****************************************************************************/
/* BIDDABLE SPOTS                                                            */
/*****************************************************************************/

typedef ML::compact_vector<std::pair<int, SmallIntVector>, 3, uint32_t >
BiddableSpotsBase;

/** Information about which adimp are biddable
    List of (adspot index, [creative indexes]) pairs that are compatible with
    this agent.
*/
struct BiddableSpots : public BiddableSpotsBase {
    Json::Value toJson() const;
    std::string toJsonStr() const;
};

struct AgentStats {

    AgentStats();

    Json::Value toJson() const;

    uint64_t auctions;
    uint64_t bids;
    uint64_t wins;
    uint64_t losses;
    uint64_t tooLate;
    uint64_t invalid;
    uint64_t noBudget;

    CurrencyPool totalBid;
    CurrencyPool totalBidOnWins;
    CurrencyPool totalSpent;

    uint64_t tooManyInFlight;
    uint64_t noSpots;
    uint64_t skippedBidProbability;
    uint64_t urlFiltered;
    uint64_t hourOfWeekFiltered;
    uint64_t locationFiltered;
    uint64_t languageFiltered;
    uint64_t userPartitionFiltered;
    uint64_t dataProfileFiltered;
    uint64_t exchangeFiltered;
    uint64_t segmentsMissing;
    uint64_t segmentFiltered;
    uint64_t augmentationTagsExcluded;
    uint64_t userBlacklisted;
    uint64_t notEnoughTime;
    uint64_t requiredIdMissing;

    uint64_t intoFilters;
    uint64_t passedStaticFilters;
    uint64_t passedStaticPhase1;
    uint64_t passedStaticPhase2;
    uint64_t passedStaticPhase3;
    uint64_t passedDynamicFilters;
    uint64_t bidErrors;
 
    uint64_t filter1Excluded;
    uint64_t filter2Excluded;
    uint64_t filternExcluded;

    uint64_t unknownWins;
    uint64_t unknownLosses;

    uint64_t requiredAugmentorIsMissing;
    uint64_t augmentorValueIsNull;
};


struct AgentStatus {
    AgentStatus()
        : dead(false), numBidsInFlight(0)
    {
        lastHeartbeat = Date::now();
    }

    bool dead;
    Date lastHeartbeat;
    size_t numBidsInFlight;
};

/// Information about a agent
struct AgentInfo {
    AgentInfo()
        : bidRequestFormat(BRF_JSON_RAW),
          configured(false),
          status(new AgentStatus()),
          stats(new AgentStats()),
          throttleProbability(1.0)
    {
    }

    enum BidRequestFormat {
        BRF_JSON_RAW,  ///< Send raw exchangeJSON bid requests
        BRF_JSON_NORM, ///< Send normalized JSON bid requests
        BRF_BINARY_V1  ///< Send binary bid requests
    } bidRequestFormat;
    
    bool configured;
    unsigned filterIndex;
    std::shared_ptr<AgentConfig> config;
    std::shared_ptr<AgentStatus> status;
    std::shared_ptr<AgentStats> stats;
    double throttleProbability;

    /** Address of the zeromq socket for this agent. */
    std::string address;
    
    /** Encode the given bid request ready to be sent to the given
        agent in its configured format.
    */
    const std::string & encodeBidRequest(const BidRequest & br) const;

    const std::string & encodeBidRequest(const Auction & auction) const;
    const std::string & getBidRequestEncoding(const Auction & auction) const;

    /** Set the bid request format. */
    void setBidRequestFormat(const std::string & val);

    /** Structure in which we record the information on ping timings. */
    struct PingInfo {
        PingInfo()
        {
        }

        Date lastSent;     ///< If in flight, it's non-null.  Otherwise null
        ML::distribution<double> history;  ///< History of ping times
    };

    PingInfo pingInfo[2];

    void gotPong(int level, Date sent, Date received, Date finished)
    {
        if (level < 0 || level >= 2)
            throw ML::Exception("wrong pong level");
        PingInfo & info = pingInfo[level];
        if (info.lastSent == Date())
            throw ML::Exception("got double pong response");
        double difference = info.lastSent.secondsSince(sent);
        if (difference > 0.001)
            throw ML::Exception("sent and lastSent differed by %f seconds",
                                difference);

        double time = finished.secondsSince(sent);
        info.history.push_back(time);
        if (info.history.size() > 100)
            info.history.erase(info.history.begin(),
                               info.history.begin() + info.history.size() - 50);
        info.lastSent = Date();
    }

    bool sendPing(int level, Date & start)
    {
        if (level < 0 || level >= 2)
            throw ML::Exception("wrong pong level");
        PingInfo & info = pingInfo[level];

        if (info.lastSent != Date())
            return false;
        info.lastSent = start;
        return true;
    }

    Json::Value toJson(bool includeConfig = true,
                       bool includeStats = true) const;

    void gotHeartbeat(Date when = Date::now())
    {
        status->lastHeartbeat = when;
        status->dead = false;
    }

    template<typename Fn>
    void forEachInFlight(const Fn & fn) const
    {
        for (auto it = bidsInFlight.begin(), end = bidsInFlight.end();
             it != end;  ++it) {
            fn(it->first, it->second);
        }
    }

    size_t numBidsInFlight() const
    {
        // DEBUG
        if (status->numBidsInFlight != bidsInFlight.size())
            throw ML::Exception("numBidsInFlight is wrong");
        return status->numBidsInFlight;
    }
    
    bool expireBidInFlight(const Id & id)
    {
        bool result = bidsInFlight.erase(id);
        status->numBidsInFlight = bidsInFlight.size();
        return result;
    }

    // Returns true if it was successfully inserted
    bool trackBidInFlight(const Id & id, Date date = Date::now())
    {
        bool result = bidsInFlight.insert(std::make_pair(id, date)).second;
        status->numBidsInFlight = bidsInFlight.size();
        return result;
    }

private:
    std::map<Id, Date> bidsInFlight;  /// Auctions in which we're participating
    //std::set<std::pair<Id, Id> > awaitingResult;  ///< Auctions which are awaiting a win/loss result
};

/** Information about one of the agents in a round robin group. */
struct PotentialBidder {
    // If inFlightProp == NULL_PROP then the bidder has been filtered out.
    enum { NULL_PROP = 1000000 };

    PotentialBidder() : inFlightProp(NULL_PROP) {}

    std::string agent;
    float inFlightProp;
    BiddableSpots imp;
    std::shared_ptr<const AgentConfig> config;
    std::shared_ptr<AgentStats> stats;

    bool operator < (const PotentialBidder & other) const
    {
        return inFlightProp < other.inFlightProp
            || (inFlightProp == other.inFlightProp
                && agent < other.agent);
    }
};

/** Information about an entire round robin group (including all of its
    agents) and how it relates to an auction.
*/
struct GroupPotentialBidders : public std::vector<PotentialBidder> {
    GroupPotentialBidders()
        : totalBidProbability(0.0)
    {
    }
    
    double totalBidProbability;
};


struct AuctionInfoBase {
    AuctionInfoBase() {}
    AuctionInfoBase(const std::shared_ptr<Auction> & auction,
                    Date lossTimeout)
        : auction(auction), lossTimeout(lossTimeout)
    {
    }

    std::shared_ptr<Auction> auction;   ///< Our copy of the auction
    Date lossTimeout;                     ///< When we send a loss if
    ///< there is no win
};

struct BidInfo {
    Date bidTime;
    BiddableSpots imp;
    std::shared_ptr<const AgentConfig> agentConfig;  //< config active at auction
};

// Information about an in-flight auction
struct AuctionInfo : public AuctionInfoBase {
    AuctionInfo() {}
    AuctionInfo(const std::shared_ptr<Auction> & auction,
                Date lossTimeout)
        : AuctionInfoBase(auction, lossTimeout)
    {
    }

    std::map<std::string, BidInfo> bidders;  ///< List of bidders

};

struct FormatInfo {
    FormatInfo()
        : numSpots(0), numBids(0)
    {
    }
        
    uint64_t numSpots;
    uint64_t numBids;

    Json::Value toJson() const;
};

struct DutyCycleEntry {
    DutyCycleEntry()
    {
        clear();
    }

    Date starting, ending;
    uint64_t nsSleeping;
    uint64_t nsProcessing;
    uint64_t nEvents;

    // Time for different parts
    uint64_t nsConfig;
    uint64_t nsBid;
    uint64_t nsAuction;
    uint64_t nsStartBidding;
    uint64_t nsWin;
    uint64_t nsLoss;
    uint64_t nsBidResult;
    uint64_t nsRemoveInFlightAuction;
    uint64_t nsRemoveSubmittedAuction;
    uint64_t nsEraseLossTimeout;
    uint64_t nsEraseAuction;
    uint64_t nsTimeout;
    uint64_t nsSubmitted;
    uint64_t nsImpression;
    uint64_t nsClick;
    uint64_t nsExpireInFlight;
    uint64_t nsExpireSubmitted;
    uint64_t nsExpireFinished;
    uint64_t nsExpireBlacklist;
    uint64_t nsExpireBanker;
    uint64_t nsExpireDebug;

    uint64_t nsOnExpireSubmitted;

    void clear();
        
    void operator += (const DutyCycleEntry & other);
        
    Json::Value toJson() const;
};

// Structure representing bids from one or multiple agents
struct BidMessage {
    std::vector<std::string> agents;

    Id auctionId;
    Bids bids;
    WinCostModel wcm;

    std::string meta;

};

} // namespace RTBKIT

#endif /* __rtb_router__rtb_router_types_h__ */
