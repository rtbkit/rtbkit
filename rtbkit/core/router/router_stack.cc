/* router_stack.cc
   Jeremy Barnes, 21 November 2012
   Copyright (c) 2011 Datacratic.  All rights reserved.

   RTB router code.
*/

#include "router_stack.h"
#include "rtbkit/core/banker/slave_banker.h"

using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* ROUTER STACK                                                              */
/*****************************************************************************/

RouterStack::
RouterStack(std::shared_ptr<ServiceProxies> services,
            const std::string & serviceName,
            double secondsUntilLossAssumed)
    : ServiceBase(serviceName, services),
      router(*this, "router", secondsUntilLossAssumed,
             false /* connect to post auction loop */),
      masterBanker(services, "masterBanker"),
      postAuctionLoop(*this, "postAuction"),
      config(services, "config"),
      monitor(services, "monitor"),
      initialized(false)
{
}

void
RouterStack::
init()
{
    ExcAssert(!initialized);

    namespace p = std::placeholders;
    router.onSubmittedAuction = std::bind(&RouterStack::submitAuction, this,
                                          p::_1, p::_2, p::_3);
    config.init();
    config.bindTcp();
    config.start();

    masterBanker.init(std::make_shared<RedisBankerPersistence>(redis));
    auto bankerAddr = masterBanker.bindTcp().second;
    masterBanker.start();

    budgetController.setApplicationLayer(make_application_layer<ZmqLayer>(getServices()));
    budgetController.start();

    auto makeSlaveBanker = [=] (const std::string & name)
        {
            auto res = make_shared<SlaveBanker>(name);
            res->setApplicationLayer(make_application_layer<ZmqLayer>(getServices()));
            res->start();
            return res;
        };

 
    // getServices()->config->dump(cerr);

    postAuctionLoop.init();
    postAuctionLoop.setBanker(makeSlaveBanker("postAuction"));
    postAuctionLoop.bindTcp();

    router.init();
    router.initFilters();
    router.setBanker(makeSlaveBanker("router"));
    router.bindTcp();

    monitor.init({"router", "postAuction", "masterBanker"});
    monitor.bindTcp();
    monitor.start();

    initialized = true;
}

void
RouterStack::
submitAuction(const std::shared_ptr<Auction> & auction,
              const Id & adSpotId,
              const Auction::Response & response)
{
    const std::string& agentAugmentations = auction->agentAugmentations[response.agent];
    postAuctionLoop.injectSubmittedAuction(auction->id,
                                           adSpotId,
                                           auction->request,
                                           auction->requestStr,
                                           auction->requestStrFormat,
                                           agentAugmentations,
                                           response,
                                           auction->lossAssumed);
}

void
RouterStack::
start(boost::function<void ()> onStop)
{
    if (!initialized)
        init();

    //config.start();
    postAuctionLoop.start();
    router.start(onStop);
}

void
RouterStack::
sleepUntilIdle()
{
    router.sleepUntilIdle();
}
    
void
RouterStack::
shutdown()
{
    router.shutdown();
    postAuctionLoop.shutdown();
    budgetController.shutdown();
    masterBanker.shutdown();
    config.shutdown();
    monitor.shutdown();
}

size_t
RouterStack::
numNonIdle() const
{
    size_t numInFlight, numAwaitingAugmentation;
    {
        numInFlight = router.inFlight.size();
        numAwaitingAugmentation = router.augmentationLoop.numAugmenting();
    }

    cerr << "numInFlight = " << numInFlight << endl;
    cerr << "numAwaitingAugmentation = " << numAwaitingAugmentation << endl;

    size_t result = numInFlight + numAwaitingAugmentation;
    return result;
}

void
RouterStack::
addBudget(const AccountKey & account, CurrencyPool amount)
{
    ExcAssertEqual(account.size(), 1);
    budgetController.addBudgetSync(account[0], amount);
}

void
RouterStack::
setBudget(const AccountKey & account, CurrencyPool amount)
{
    ExcAssertEqual(account.size(), 1);
    budgetController.setBudgetSync(account[0], amount);
}

void
RouterStack::
topupTransfer(const AccountKey & account, CurrencyPool amount)
{
    budgetController.topupTransferSync(account, amount);
}

} // namespace RTBKIT
