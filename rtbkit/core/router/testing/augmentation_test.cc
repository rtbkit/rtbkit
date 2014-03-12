/* augmentation_test.cc
   Jeremy Barnes, 4 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Test for the augmentation functionality.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/arch/futex.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "rtbkit/plugins/augmentor/augmentor_base.h"
#include <boost/thread/thread.hpp>
#include <ace/Signal.h>


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_augmentation_no_augmentors )
{
    Watchdog watchdog(10.0);

    auto proxies = std::make_shared<ServiceProxies>();

    AugmentationLoop loop(proxies, "augmentation");

    loop.init();
    loop.start();

    proxies->config->dump(cerr);

    auto agentConfig = std::make_shared<AgentConfig>();
    AgentConfig::AugmentationInfo info;
    info.name = "random";
    agentConfig->augmentations.push_back(info);
    
    PotentialBidder bidder;
    bidder.agent = "test";
    bidder.config = agentConfig;

    // Add the augmentor
    AugmentorBase augmentor("random", "random", proxies);
    augmentor.init();

    uint64_t numAugmented = 0;

    augmentor.onRequest = [&] (const AugmentationRequest & request)
        {
            ML::atomic_inc(numAugmented);

            AugmentationList result;
            result[AccountKey()].data[request.augmentor] = "hello";
            augmentor.respond(request, result);
        };

    augmentor.start();

    proxies->config->dump(cerr);

    ML::sleep(1.0);

    cerr << "starting auctions" << endl;

    //augmentor.configureAndWait();


    // Do some auctions as fast as we can manage them
    filter_istream auctions("rtbkit/core/router/testing/20000-datacratic-auctions.xz");
    
    uint64_t numStarted = 0, numFinished = 0;

    for (unsigned i = 0;  i < 100;  ++i) {
        string current;
        getline(auctions, current);
        
        Date start = Date::now();
        Date expiry = start.plusSeconds(0.05);
        
        std::shared_ptr<BidRequest> bidRequest
            (BidRequest::parse("datacratic", current));
        
        auto handleAuction = [&] (std::shared_ptr<Auction> auction)
            {
            };
        
        auto auction = std::make_shared<Auction>(handleAuction,
                                                   bidRequest,
                                                   current,
                                                   "datacratic", 
                                                   start, expiry);

        auto info = std::make_shared<AugmentationInfo>();
        info->auction = auction;
        info->potentialGroups.resize(1);
        info->potentialGroups[0].push_back(bidder);
    
        auto finished = [&] (std::shared_ptr<AugmentationInfo>)
            {
                ML::atomic_inc(numFinished);
            };

        ML::atomic_inc(numStarted);
        loop.augment(info, Date::now().plusSeconds(0.005), finished);
    }

    cerr << "finished injecting auctions" << endl;
    
    loop.sleepUntilIdle();

    ML::sleep(0.1);

    cerr << "loop.numAugmenting() = " << loop.numAugmenting() << endl;

    loop.shutdown();

    cerr << "finished loop shutdown" << endl;

    augmentor.shutdown();

    cerr << "numStarted " << numStarted
         << " numFinished " << numFinished << endl;

    BOOST_CHECK_EQUAL(numStarted, numFinished);
    BOOST_CHECK_EQUAL(numAugmented, numFinished);
}
