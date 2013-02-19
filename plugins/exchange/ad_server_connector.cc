/* ad_server_connector.cc
   Jeremy Barnes, 19 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Connector for ad server.
*/


#include "ad_server_connector.h"
#include "rtbkit/core/post_auction/post_auction_loop.h"

using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* POST AUCTION PROXY                                                        */
/*****************************************************************************/

PostAuctionProxy::
PostAuctionProxy(std::shared_ptr<zmq::context_t> zmqContext)
    : toPostAuctionService(zmqContext)
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
injectImpression(const Id & auctionId,
                 const Id & adSpotId,
                 Date timestamp,
                 const JsonHolder & impressionMeta,
                 const UserIds & ids)
{
    PostAuctionEvent event;
    event.type = PAE_IMPRESSION;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.uids = ids;
    event.metadata = impressionMeta;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("IMPRESSION", str);
}
    
void
PostAuctionProxy::
injectClick(const Id & auctionId,
            const Id & adSpotId,
            Date timestamp,
            const JsonHolder & clickMeta,
            const UserIds & ids)
{
    PostAuctionEvent event;
    event.type = PAE_IMPRESSION;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.uids = ids;
    event.metadata = clickMeta;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("CLICK", str);
}

void
PostAuctionProxy::
injectVisit(Date timestamp,
            const SegmentList & channels,
            const JsonHolder & visitMeta,
            const UserIds & ids)
{
    PostAuctionEvent event;
    event.type = PAE_IMPRESSION;
    event.timestamp = timestamp;
    event.channels = channels;
    event.metadata = visitMeta;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService.sendMessage("VISIT", str);
}

} // namespace RTBKIT
