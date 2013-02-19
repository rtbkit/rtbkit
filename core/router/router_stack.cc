/* router_stack.cc
   Jeremy Barnes, 21 November 2012
   Copyright (c) 2011 Datacratic.  All rights reserved.

   RTB router code.
*/

#include "router_stack.h"


using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* ROUTER STACK                                                              */
/*****************************************************************************/

RouterStack::
RouterStack(std::shared_ptr<ServiceProxies> services,
            const std::string & serviceName,
            double secondsUntilLossAssumed,
            bool simulationMode)
    : ServiceBase(serviceName, services),
      router(*this, "router", secondsUntilLossAssumed,
             simulationMode, false /* connect to post auction loop */),
      postAuctionLoop(*this, "postAuction"),
      config(services, "config"),
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
 
    getServices()->config->dump(cerr);

    postAuctionLoop.init();
    postAuctionLoop.bindTcp();

    router.init();
    router.bindTcp();

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
    config.shutdown();
}

size_t
RouterStack::
numNonIdle() const
{
    size_t numInFlight, numSubmitted, numAwaitingAugmentation;
    {
        numInFlight = router.inFlight.size();
        numAwaitingAugmentation = router.augmentationLoop.numAugmenting();
        numSubmitted = postAuctionLoop.numAwaitingWinLoss();
    }

    cerr << "numInFlight = " << numInFlight << endl;
    cerr << "numSubmitted = " << numSubmitted << endl;
    cerr << "numAwaitingAugmentation = " << numAwaitingAugmentation << endl;

    size_t result = numInFlight + numSubmitted + numAwaitingAugmentation;
    return result;
}

#if 0
void
RouterStack::
addBudget(const AccountKey & account, CurrencyPool amount)
{
    ExcAssertEqual(account.size(), 1);
    budgetController->addBudgetSync(account[0], amount);
}

void
RouterStack::
setBudget(const AccountKey & account, CurrencyPool amount)
{
    ExcAssertEqual(account.size(), 1);
    budgetController->setBudgetSync(account[0], amount);
}

void
RouterStack::
topupTransfer(const AccountKey & account, CurrencyPool amount)
{
    budgetController->topupTransferSync(account, amount);
}
#endif

} // namespace RTBKIT
