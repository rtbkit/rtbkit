/* test_agent.h                                                    -*- C++ -*-
   Jeremy Banres, 27 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Test for an agent.
*/

#pragma once

#include "rtbkit/common/account_key.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "jml/arch/futex.h"

namespace RTBKIT {

struct TestAgent : public BiddingAgent {
    TestAgent(std::shared_ptr<RTBKIT::ServiceProxies> proxies,
              const std::string & name = "testAgent",
              const AccountKey & accountKey
              = AccountKey({"testCampaign", "testStrategy"})) noexcept
        : RTBKIT::BiddingAgent(proxies, name)
    {
        setDefaultConfig(accountKey);
        setupCallbacks();
        clear();
    }

    RTBKIT::AgentConfig config;

    void setDefaultConfig(const AccountKey & accountKey)
    {
        RTBKIT::AgentConfig config;
        config.account = accountKey;
        config.maxInFlight = 20000;
        config.creatives.push_back(RTBKIT::Creative::sampleLB);
        config.creatives.push_back(RTBKIT::Creative::sampleWS);
        config.creatives.push_back(RTBKIT::Creative::sampleBB);
        config.externalId = 1;

        this->config = config;
    }

    void start()
    {
        BiddingAgent::start();
        configure();
    }

    void clear()
    {
        numHeartbeats = numBidRequests = numErrors = numGotConfig = 0;
        numWins = numLosses = numNoBudgets = numTooLates = 0;
        numBidsOutstanding = 0;
    }

    bool sleepUntilIdle(double waitTime = 0.0)
    {
        Date timeout;
        if (waitTime > 0.0) timeout = Date::now().plusSeconds(waitTime);

        for (;;) {
            int oldOutstanding = numBidsOutstanding;

            //cerr << "numBidsOutstanding = " << oldOutstanding << endl;

            if (oldOutstanding == 0) return true;
            
            Date now = Date::now();
            if (now >= timeout && waitTime != 0.0) return false;
            
            int res;

            if (waitTime == 0)
                res = ML::futex_wait(numBidsOutstanding, oldOutstanding);
            else res = ML::futex_wait(numBidsOutstanding, oldOutstanding,
                                      now.secondsUntil(timeout));

            //cerr << "res = " << res << endl;

            if (res == 0) return true;
            if (errno == ETIMEDOUT) return false;
            if (errno == EINTR) continue;
            if (errno == EAGAIN) continue;
            
            throw ML::Exception(errno, "futex_wait");
        }
    }

    int numHeartbeats;
    int numBidRequests;
    int numErrors;
    int numGotConfig;
    int numWins;
    int numLosses;
    int numNoBudgets;
    int numTooLates;

    typedef ML::Spinlock Lock;
    typedef boost::lock_guard<Lock> Guard;
    mutable Lock lock;

    std::set<Id> awaitingStatus;
    int numBidsOutstanding;
    
    void defaultError(double timestamp, const std::string & error,
                 const std::vector<std::string> & message)
    {
        using namespace std;
        cerr << "agent got error: " << error << " from message: "
             << message << endl;
        __sync_fetch_and_add(&numErrors, 1);
    }

    void finishBid(int & counter, const RTBKIT::BidResult & args)
    {
        __sync_fetch_and_add(&counter, 1);
        Guard guard(lock);
        if (!awaitingStatus.erase(args.auctionId))
            throw ML::Exception("couldn't find in progress auction");

        numBidsOutstanding = awaitingStatus.size();
        if (numBidsOutstanding == 0)
            ML::futex_wake(numBidsOutstanding);
    }

    void defaultWin(const RTBKIT::BidResult & args)
    {
        finishBid(numWins, args);
        //cerr << args.accountInfo << endl;
    }
                
    void defaultLoss(const RTBKIT::BidResult & args)
    {
        finishBid(numLosses, args);
    }

    void defaultNoBudget(const RTBKIT::BidResult & args)
    {
        finishBid(numNoBudgets, args);
    }

    void defaultTooLate(const RTBKIT::BidResult & args)
    {
        finishBid(numTooLates, args);
    }

    void doBid(const Id & id,
               const Bids & bids,
               const Json::Value & metadata,
               const WinCostModel & wcm,
               bool record = true)
    {
        if (record && bids.size() != 0)
            recordBid(id);
        RTBKIT::BiddingAgent::doBid(id, bids, metadata, wcm);
    }

    void bidNull(double timestamp,
                 const Id & id,
                 std::shared_ptr<RTBKIT::BidRequest> br,
                 const Bids & bids,
                 double timeLeftMs,
                 const Json::Value & augmentations,
                 const WinCostModel & wcm)
    {
        using namespace std;
        //cerr << "got auction " << id << endl;
        doBid(id, bids, Json::Value(), wcm);
        __sync_fetch_and_add(&numBidRequests, 1);
    }

    void bidWithFixedAmount(Amount amount) {
        onBidRequest = [this, amount] (double timestamp,
                                       const Id & id,
                                       std::shared_ptr<BidRequest> br,
                                       Bids bids,
                                       double timeLeftMs,
                                       const Json::Value & augmentations,
                                       const WinCostModel & wcm)
        {
            //std::cerr << amount.toString() << std::endl;
            __sync_fetch_and_add(&numBidRequests, 1);
            Bid & bid = bids[0];
            bid.bid(bid.availableCreatives[0], amount);
            doBid(id, bids, Json::Value(), wcm);
        };
    }

    void recordBid(const Id & id)
    {
        Guard guard(lock);
        if (!awaitingStatus.insert(id).second)
            throw ML::Exception("auction already in progress");
        
        numBidsOutstanding = awaitingStatus.size();
    }

    void setupCallbacks()
    {
        onError
            = boost::bind(&TestAgent::defaultError, this, _1, _2, _3);
        onBidRequest
            = boost::bind(&TestAgent::bidNull, this, _1, _2, _3, _4, _5, _6, _7);
        onWin
            = boost::bind(&TestAgent::defaultWin, this, _1);
        onLoss
            = boost::bind(&TestAgent::defaultLoss, this, _1);
        onNoBudget
            = boost::bind(&TestAgent::defaultNoBudget, this, _1);
        onTooLate
            = boost::bind(&TestAgent::defaultTooLate, this, _1);
    }    

    void configure()
    {
        doConfig(config);
    }

};

} // namespace RTBKIT
