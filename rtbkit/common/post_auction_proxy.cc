/** post_aution_proxy.cc                                 -*- C++ -*-
    Rémi Attab, 05 Aug 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Post auction proxy communication thingy.

*/

#include "post_auction_proxy.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/******************************************************************************/
/* POST AUCTION PROXY                                                         */
/******************************************************************************/

PostAuctionProxy::
PostAuctionProxy(shared_ptr<ServiceProxies> proxies) :
    shards(proxies->params.get("postAuctionShards", 1).asInt()),
    proxies(proxies)
{}

void
PostAuctionProxy::
init()
{
    toPostAuction.init(proxies->config);
    toPostAuction.connectAllServiceProviders("rtbPostAuctionService", "events");
}

bool
PostAuctionProxy::
isConnected() const
{
    for (size_t shard = 0; shard < shards; ++shard) {
        if (!toPostAuction.isConnectedToShard(shard)) return false;
    }

    return true;
}

void
PostAuctionProxy::
sendAuction(SubmittedAuctionEvent event)
{
    size_t shard = event.auctionId.hash() % shards;
    string str = ML::DB::serializeToString(event);

    // we intentionally drop the message if the shard isn't up.
    (void) toPostAuction.sendMessageToShard(shard, "AUCTION", move(str));
}

void
PostAuctionProxy::
sendEvent(PostAuctionEvent event)
{
    size_t shard = event.auctionId.hash() % shards;
    string str = ML::DB::serializeToString(event);

    // we intentionally drop the message if the shard isn't up.
    (void) toPostAuction.sendMessageToShard(shard, print(event.type), str);
}


} // namepsace RTBKIT
