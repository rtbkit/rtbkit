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

namespace RTBKIT {

/******************************************************************************/
/* EVENT FORWARDER                                                            */
/******************************************************************************/

struct EventForwarder : public ServiceBase, public MessageLoop
{
    EventForwarder(ServiceBase & parent, std::string uri) :
        ServiceBase("forward", parent),
        client(std::move(uri), 1024),
        queue(8)
    {
        MessageLoop::init();

        using std::placeholders::_1;
        queue.onEvent = std::bind(&EventForwarder::sendAuction, this, _1);

        addSource("PostAuctionService::EventForwarder::client", client);
        addSource("PostAuctionService::EventForwarder::queue", queue);

        MessageLoop::start();
    }

    ~EventForwarder()
    {
        MessageLoop::shutdown();
    }

    void forwardAuction(std::shared_ptr<SubmittedAuctionEvent> auction)
    {
        recordHit("auctions");
        if (!queue.tryPush(auction))
            recordHit("dropped");
    }


private:

    void sendAuction(std::shared_ptr<SubmittedAuctionEvent> auction)
    {
        Date start = Date::now();

        HttpRequest::Content body(ML::DB::serializeToString(*auction), "application/json");

        auto onDone = [=] (const HttpRequest&, HttpClientError err) {
            if (err != HttpClientError::None) {
                recordHit("error");
                return;
            }

            recordHit("success");
            recordOutcome(Date::now().secondsSince(start), "latency");
        };

        auto cb = std::make_shared<HttpClientCallbacks>(nullptr, nullptr, nullptr, onDone);

        client.post("/auctions", cb, body, RestParams(), RestParams(), 1000);
    }

    HttpClient client;
    TypedMessageSink< std::shared_ptr< SubmittedAuctionEvent> > queue;
};


} // namespace RTBKIT
