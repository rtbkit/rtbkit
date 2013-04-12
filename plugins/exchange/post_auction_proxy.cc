/* post_auction_proxy.cc
   Jeremy Barnes, 19 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   A class that proxies post auction events to the post auction loop.
*/


#include "post_auction_proxy.h"
#include "rtbkit/core/post_auction/post_auction_loop.h"

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* POST AUCTION PROXY                                                        */
/*****************************************************************************/

PostAuctionProxy::
PostAuctionProxy(std::shared_ptr<zmq::context_t> context)
    : toPostAuctionService(context)
{
}

PostAuctionProxy::
~PostAuctionProxy()
{
}

void
PostAuctionProxy::
init(std::shared_ptr<ConfigurationService> config)
{
    toPostAuctionService.init(config, ZMQ_XREQ);
    toPostAuctionService.connectToServiceClass("rtbPostAuctionService", "events");

#if 0 // later, for when we have multiple

    toPostAuctionServices.init(getServices()->config, name);
    toPostAuctionServices.connectHandler = [=] (const std::string & connectedTo)
        {
            cerr << "PostAuctionProxy is connected to post auction service "
            << connectedTo << endl;
        };
    toPostAuctionServices.connectAllServiceProviders("rtbPostAuctionService",
                                                     "agents");
    toPostAuctionServices.connectToServiceClass();
#endif
}

void
PostAuctionProxy::
start()
{
}

void
PostAuctionProxy::
shutdown()
{
}

void
PostAuctionProxy::
injectWin(const Id & auctionId,
          const Id & adSpotId,
          Amount winPrice,
          Date timestamp,
          const JsonHolder & winMeta,
          const UserIds & ids,
          const AccountKey & account,
          Date bidTimestamp)
{
    PostAuctionEvent event;
    event.type = PAE_WIN;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.winPrice = winPrice;
    event.timestamp = timestamp;
    event.metadata = winMeta;
    event.uids = ids;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("WIN", str);
}

void
PostAuctionProxy::
injectLoss(const Id & auctionId,
           const Id & adSpotId,
           Date timestamp,
           const JsonHolder & lossMeta,
           const AccountKey & account,
           Date bidTimestamp)
{
    PostAuctionEvent event;
    event.type = PAE_LOSS;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.metadata = lossMeta;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("LOSS", str);
}

void
PostAuctionProxy::
injectCampaignEvent(const string & label,
                    const Id & auctionId,
                    const Id & adSpotId,
                    Date timestamp,
                    const JsonHolder & impressionMeta,
                    const UserIds & ids)
{
    PostAuctionEvent event;
    event.type = PAE_CAMPAIGN_EVENT;
    event.label = label;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.uids = ids;
    event.metadata = impressionMeta;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("EVENT", str);
}

} // namespace RTBKIT
