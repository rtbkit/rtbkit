/* zmq_message_router.h                                            -*- C++ -*-
   Jeremy Barnes, 22 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Router object to hook up zeromq messages (identified with a topic)
   to callback functions.
*/

#pragma once

#include "named_endpoint.h"
#include "message_loop.h"


namespace Datacratic {


/*****************************************************************************/
/* ZMQ MESSAGE ROUTER                                                        */
/*****************************************************************************/

struct ZmqMessageRouter: public ZmqEventSource {

    ZmqMessageRouter(bool routable = true)
        : routable(routable)
    {
    }

    void addRoute(const std::string & topic,
                  AsyncMessageHandler handler)
    {
        messageHandlers[topic] = handler;
    }
    
    void bind(const std::string & topic,
              const std::function<void (const std::vector<std::string> & args)> & handler)
    {
        messageHandlers[topic] = handler;
    }

    virtual void handleMessage(const std::vector<std::string> & message)
    {
        std::string topic = message.at(routable);

        //using namespace std;
        //cerr << "got message " << topic << " " << message << endl;

        auto it = messageHandlers.find(topic);
        if (it == messageHandlers.end())
            defaultHandler(message);
        else it->second(message);
    }

    bool routable;
    std::map<std::string, AsyncMessageHandler> messageHandlers;
    AsyncMessageHandler defaultHandler;
};


} // namespace Datacratic

