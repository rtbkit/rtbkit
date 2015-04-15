/** post_aution_proxy.cc                                 -*- C++ -*-
    Rémi Attab, 05 Aug 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Post auction proxy communication thingy.

*/

#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "rtbkit/core/post_auction/event_forwarder.h"
#include "post_auction_proxy.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

/******************************************************************************/
/* POST AUCTION PROXY                                                         */
/******************************************************************************/

PostAuctionProxy::
PostAuctionProxy(ServiceBase& parent) :
    parent(&parent),
    proxies(parent.getServices())
{}

PostAuctionProxy::
PostAuctionProxy(std::shared_ptr<Datacratic::ServiceProxies> proxies) :
    parent(nullptr),
    proxies(proxies)
{}

void
PostAuctionProxy::
init()
{
    if (proxies->params.isMember("postAuctionURIs"))
        initHTTP();
    else initZMQ();
}

void
PostAuctionProxy::
initZMQ()
{
    shards = proxies->params.get("postAuctionShards", 1).asInt();

    zmq.reset(new Datacratic::ZmqMultipleNamedClientBusProxy);
    zmq->init(proxies->config);
    zmq->connectAllServiceProviders("rtbPostAuctionService", "events");
}

void
PostAuctionProxy::
initHTTP()
{
    const auto& uris = proxies->params["postAuctionURIs"];
    ExcCheckEqual(uris.type(), Json::arrayValue, "invalid postAuctionURIs type");

    shards = uris.size();
    http.resize(uris.size());

    for (size_t i = 0; i < uris.size(); ++i) {
        std::string name = "postAuctionProxy" + std::to_string(i);

        if (parent)
            http[i] = std::make_shared<EventForwarder>(*parent, uris[i].asString(), name);
        else
            http[i] = std::make_shared<EventForwarder>(proxies, uris[i].asString(), name);
    }
}

bool
PostAuctionProxy::
isConnected() const
{
    if (!zmq) return true;

    for (size_t shard = 0; shard < shards; ++shard) {
        if (!zmq->isConnectedToShard(shard)) return false;
    }

    return true;
}

void
PostAuctionProxy::
sendAuction(std::shared_ptr<SubmittedAuctionEvent> event)
{
    size_t shard = event->auctionId.hash() % shards;

    if (!zmq) http[shard]->forwardAuction(event);
    else {
        string str = ML::DB::serializeToString(*event);
        (void) zmq->sendMessageToShard(shard, "AUCTION", move(str));
    }
}

void
PostAuctionProxy::
sendEvent(std::shared_ptr<PostAuctionEvent> event)
{
    size_t shard = event->auctionId.hash() % shards;

    if (!zmq) http[shard]->forwardEvent(event);
    else {
        string str = ML::DB::serializeToString(*event);
        (void) zmq->sendMessageToShard(shard, print(event->type), str);
    }
}


} // namepsace RTBKIT
