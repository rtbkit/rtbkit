/** event_forwarder.h                                 -*- C++ -*-
    Rémi Attab, 04 Dec 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    post auction event forwarding.

*/

#pragma once

#include "soa/service/message_loop.h"
#include "soa/service/service_base.h"
#include "soa/service/http_client.h"
#include "soa/service/typed_message_channel.h"
#include "rtbkit/common/auction_events.h"

namespace RTBKIT {

/******************************************************************************/
/* EVENT FORWARDER                                                            */
/******************************************************************************/

struct EventForwarder :
        public Datacratic::ServiceBase,
        public Datacratic::MessageLoop
{
    enum {
        ConnectionCount = 1 << 7,
        AuctionQueueSize = 1 << 6,
        EventQueueSize = 1 << 8,
    };

    EventForwarder(
            Datacratic::ServiceBase & parent,
            std::string uri, std::string name) :
        ServiceBase(std::move(name), parent),
        client(std::move(uri), ConnectionCount),
        auctionQueue(AuctionQueueSize),
        eventQueue(EventQueueSize)
    {
        init();
    }

    EventForwarder(
            std::shared_ptr<Datacratic::ServiceProxies>& proxies,
            std::string uri, std::string name) :
        ServiceBase(std::move(name), proxies),
        client(std::move(uri), ConnectionCount),
        auctionQueue(AuctionQueueSize),
        eventQueue(EventQueueSize)
    {
        init();
    }


    ~EventForwarder()
    {
        MessageLoop::shutdown();
    }

    void forwardAuction(std::shared_ptr<SubmittedAuctionEvent> auction)
    {
        recordHit("auctions.received");
        if (!auctionQueue.tryPush(auction))
            recordHit("auctions.dropped");
    }

    void forwardEvent(std::shared_ptr<PostAuctionEvent> event)
    {
        recordHit("events.received");
        if (!eventQueue.tryPush(event))
            recordHit("events.dropped");
    }


private:

    void init()
    {
        MessageLoop::init();

        client.sendExpect100Continue(false);

        using std::placeholders::_1;
        auctionQueue.onEvent = std::bind(&EventForwarder::sendAuction, this, _1);
        eventQueue.onEvent = std::bind(&EventForwarder::sendEvent, this, _1);

        addSource("PostAuctionService::EventForwarder::client", client);
        addSource("PostAuctionService::EventForwarder::auctionQueue", auctionQueue);
        addSource("PostAuctionService::EventForwarder::eventQueue", eventQueue);

        MessageLoop::start();
    }

    template<typename T>
    void send(const std::string& endpoint, const T& obj)
    {
        recordHit("%s.send", endpoint);
        Date start = Date::now();

        static auto desc = getDefaultDescriptionShared((T*) 0);

        std::stringstream stream;
        Datacratic::StreamJsonPrintingContext ctx(stream);
        desc->printJson(&obj, ctx);

        HttpRequest::Content body(stream.str(), "application/json");

        auto onDone = [=] (const HttpRequest&, HttpClientError err) {
            if (err != HttpClientError::None) {
                recordHit("%s.error", endpoint);
                return;
            }

            recordHit("%s.success", endpoint);
            recordOutcome(Date::now().secondsSince(start), "%s.latency", endpoint);
        };

        auto cb = std::make_shared<HttpClientCallbacks>(nullptr, nullptr, nullptr, onDone);

        client.post("/v1/" + endpoint, cb, body, RestParams(), RestParams(), 1);
    }

    void sendAuction(std::shared_ptr<SubmittedAuctionEvent> auction)
    {
        send("auctions", *auction);
    }

    void sendEvent(std::shared_ptr<PostAuctionEvent> event)
    {
        send("events", *event);
    }

    HttpClient client;
    TypedMessageSink< std::shared_ptr< SubmittedAuctionEvent> > auctionQueue;
    TypedMessageSink< std::shared_ptr< PostAuctionEvent> > eventQueue;
};


} // namespace RTBKIT
