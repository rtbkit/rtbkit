/* nsq_event_handler.cc   
   Mathieu Vadnais, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
*/

#include "soa/service/nsq_event_handler.h"


using namespace Datacratic;

NsqEventHandler::
NsqEventHandler(const std::string & loggerUrl,
                const OnMessageReceived & onMessageReceived)
    :closed_(true)
{
    loop.start();
    auto onClosed = [&] (bool fromPeer,
                         const std::vector<std::string> & msgs) {
        this->onClosed(fromPeer, msgs);
    };
    client.reset(new NsqClient(onClosed, onMessageReceived));
    loop.addSource("client", client);
    init(loggerUrl);
    client->connectSync();
    closed_ = false;
}

NsqEventHandler::
~NsqEventHandler()
{
    auto onClosed = [&] (const NsqFrame & frame) {
        client->requestClose();
    };
    client->cls(onClosed);

    while (!closed_) {
        int old(closed_);
        ML::futex_wait(closed_, old);
    }
    loop.shutdown();
}

void 
NsqEventHandler::
init(const std::string & loggerUrl)
{
    client->init(loggerUrl);
}

void 
NsqEventHandler::
subscribe(const std::string & topic, 
          const std::string & channel) 
{

    client->sub(topic,channel);
}

void 
NsqEventHandler::
consumeMessage(const std::string & messageId)
{
    client->fin(messageId);
}

void 
NsqEventHandler::
publishMessage(const std::string & topic,
               const std::string & message)
{
    client->pub(topic,message);
}

void 
NsqEventHandler::
onClosed(bool fromPeer, 
         const std::vector<std::string> & msgs)
{
    closed_ = true;
    ML::futex_wake(closed_);
}