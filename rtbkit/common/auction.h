/* auction.h                                                       -*- C++ -*-
   Jeremy Barnes, 6 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Base class for agents to deal with an auction.
*/

#pragma once

#include "soa/types/basic_value_descriptions.h"
#include "rtbkit/common/json_holder.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"
#include "rtbkit/common/augmentation.h"
#include "rtbkit/common/win_cost_model.h"
#include <boost/function.hpp>
#include <boost/enable_shared_from_this.hpp>
#include "soa/jsoncpp/json.h"
#include "soa/types/date.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/exception.h"
#include "jml/utils/compact_vector.h"
#include "jml/db/persistent_fwd.h"

namespace RTBKIT {

struct AgentConfig;
struct ExchangeConnector;
struct AuctionPriceDescription;
struct AuctionResponseDescription;

/*****************************************************************************/
/* AUCTION                                                                   */
/*****************************************************************************/

/** This is the object that represents an in-flight auction to the router.
    Agents modify this object in order to set their bids.
*/

struct Auction : public std::enable_shared_from_this<Auction> {
    
    /** Callback that's called once the auction is finished. */
    typedef boost::function<void (std::shared_ptr<Auction> auction)>
        HandleAuction;

    Auction();
    
    Auction(ExchangeConnector * exchangeConnector,
            HandleAuction handleAuction,
            std::shared_ptr<BidRequest> request,
            const std::string & requestStr,
            const std::string & requestStrFormat,
            Date start,
            Date expiry);
    
    ~Auction();

    bool isZombie;  ///< Auction was externally cancelled

    Date start;
    Date expiry;
    Date lossAssumed;

    // Debug
    Date doneParsing;
    Date inPrepro, outOfPrepro;
    Date doneAugmenting;
    Date inStartBidding;

    Id id;
    std::shared_ptr<BidRequest>  request;
    std::string requestStr;  ///< Stringified version of request
    std::string requestStrFormat;  ///< Format of stringified request
    std::string requestSerialized; ///< Serialized bid request (canonical)

    ///< AugmentationList for each augmentors.
    std::unordered_map<std::string, AugmentationList> augmentations;
    AgentAugmentations agentAugmentations; ///< per agent augmentations.

    /** How much time is still available for the auction (in seconds). */
    double timeAvailable(Date now = Date::now()) const;

    /** How much time has been used by the auction (in seconds). */
    double timeUsed(Date now = Date::now()) const;

    /** If this value is set, then the bid has already been sent of and it's
        too late to modify the object any more.
    */
    bool tooLate();

    struct Price {
        Price(Amount maxPrice = Amount(), float priority = 0.0)
            : maxPrice(maxPrice), priority(priority)
        {
        }

        Amount maxPrice;      ///< Price to bid for this *one* ad
        float  priority;       ///< Bid priority for when multiple campaigns bid

        Json::Value toJson() const;
        std::string toJsonStr() const;
        static Price fromJson(const Json::Value&);

        static void createDescription(AuctionPriceDescription&);
    };

    /** Price to bid if you don't want to bid */
    static const Price NONE;

    /** What happened to the bid at the local and global level? */
    struct WinLoss : public Datacratic::TaggedEnum<WinLoss> {
        enum {
            PENDING,    ///< Bid is pending; unknown if we won or lost
            WIN,        ///< Bid was won
            LOSS,       ///< Bid was lost
            TOOLATE,    ///< Bid was too late and so not accepted
            INVALID,    ///< Bid was invalid and so not accepted
        };

        WinLoss(int value = INVALID) {
            val = value;
        }
    };

    /** Response to a bid. */
    struct Response {
        Response(Price price = NONE,
                 int creativeId = -1,
                 const AccountKey & account = AccountKey(),
                 bool test = true,
                 std::string agent = "",
                 Bids bids = Bids(),
                 std::string meta = "null",
                 std::shared_ptr<const AgentConfig> agentConfig
                     = std::shared_ptr<const AgentConfig>(),
                 const SegmentList& visitChannels = SegmentList(),
                 int agentCreativeIndex = -1,
                 const WinCostModel & wcm = WinCostModel())
            : price(price),
              account(account),
              test(test), agent(agent),
              bidData(bids),
              meta(meta),
              creativeId(creativeId),
              agentConfig(agentConfig),
              visitChannels(visitChannels),
              agentCreativeIndex(agentCreativeIndex),
              wcm(wcm)
        {
        }

        // Information about the actual bid
        Price price;           ///< Price to bid on
        AccountKey account;    ///< Account we are bidding with
        bool test;             ///< Is this a test bid?

        // Information about the agent who did the bidding
        std::string agent;    ///< Agent ID who's bidding

        Bids bidData;   ///< Data that the bidder wants to keep
        Datacratic::UnicodeString meta; ///< Free form agent information about the bid
                               ///< (Passed back to agent with notification)

        int creativeId;           ///< Id of the creative/placement
        Datacratic::UnicodeString creativeName; ///< Name of the creative

        // Information about the status of the bid (what happened to it)
        WinLoss localStatus;   ///< What happened in the local auction?

        /** Configuration of the agent that made the bid

            WARNING: This member will not be serialized and will therefore not
            be available out of process.
        */
        std::shared_ptr<const AgentConfig> agentConfig;

        // List of channels for which we subscribe to post impression visit
        // events.
        SegmentList visitChannels;

        /** Creative index in this agentConfig's creatives array. */
        int agentCreativeIndex;

        /** Win cost model for this auction. */
        WinCostModel wcm;

        static std::string print(WinLoss wl);
        Json::Value toJson() const;
        std::string toJsonStr() const;

        void serialize(ML::DB::Store_Writer & store) const;
        void reconstitute(ML::DB::Store_Reader & store);

        /** Is this a valid response? */
        bool valid() const;

        static void createDescription(AuctionResponseDescription&);
    };

    /** Modify the given response.  The boolean return code says whether or
        not this response was accepted (due to it being the maximum-priority
        response).

        Returns the (local) status of the response.

        Thread safe.
    */
    WinLoss setResponse(int spotNum, Response newResponse);

    /** Merges the given data sources used to make the bidding decision with the
        ones already already present in the auction.

        Thread safe.
     */
    void addDataSources(const std::set<std::string> & sources);
    const std::set<std::string> & getDataSources() const;

    /** Return a status that can be used for debugging. */
    std::string status() const;

    /** Set an error flag on the auction, which will cause the response to
        reflect this error.  Also finishes the auction.
    */
    bool setError(const std::string & error, const std::string & details = "");

    /** Finish the auction.  This will call the auction handler.  The return
        value is true if finish() was actually called, or false if it was
        previously finished by something else.
    */
    bool finish();

    /** How many imp in this auction? */
    size_t numSpots() const
    {
        return request->imp.size();
    }

    /** Return a JSON representation of the response. */
    Json::Value getResponseJson(int spotNum) const;
    
    /** Return all responses as JSON. */
    Json::Value getResponsesJson() const;

    /** Get the wins, winning first and than any losing bids afterwards.  The
        auction should be over before this is called.
    */
    const std::vector<std::vector<Response> > & getResponses() const;

    ExchangeConnector * exchangeConnector; ///< Exchange connector for auction
    HandleAuction handleAuction;   ///< Callback for when auction is finished

    struct Data {
        Data()
            : tooLate(false), oldData(0)
        {
            responses.reserve(8);
        }

        Data(int numSpots)
            : tooLate(false), responses(numSpots), oldData(0)
        {
        }

        bool hasValidResponse(int spotNum) const
        {
            if (spotNum >= responses.size())
                throw ML::Exception("invalid spot number");
            return !responses[spotNum].empty();
        }
        
        bool hasError() const
        {
            return !error.empty();
        }

        const Response & winningResponse(int spotNum) const
        {
            if (spotNum >= responses.size())
                throw ML::Exception("invalid spot number");
            if (responses[spotNum].empty())
                throw ML::Exception("empty responses");
            return responses[spotNum][0];
        }

        bool tooLate;
        std::vector<std::vector<Response> > responses;  ///< Losing responses to track
        std::set<std::string> dataSources; // data sources used to make the bid decissions.
        Data * oldData;  ///< GC list
        std::string error, details;
    };

    const Data * getCurrentData() const
    {
        return data;
    }

private:
    Data * data;

public:
    /// Memory leak tracking
    static long long created;
    static long long destroyed;
};

CREATE_CLASS_DESCRIPTION_NAMED(AuctionPriceDescription,
                               Auction::Price)

CREATE_CLASS_DESCRIPTION_NAMED(AuctionResponseDescription,
                               Auction::Response)

} // namespace RTBKIT

