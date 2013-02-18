/* flume_endpoint.h                                                -*- C++ -*-
   Jeremy Barnes, 15 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Endpoint that sets up a server that will listen for Flume events.
*/

#ifndef __logger__flume_endpoint_h__
#define __logger__flume_endpoint_h__

#include <boost/function.hpp>
#include <boost/thread.hpp>
#include <boost/shared_ptr.hpp>
#include <string>
#include <map>
#include "soa/service/stats_events.h"


namespace Datacratic {


/*****************************************************************************/
/* FLUME RPC ENDPOINT                                                        */
/*****************************************************************************/

/** This class implements an endpoint for the Flume RPC protocol. */

struct FlumeRpcEndpoint {
    FlumeRpcEndpoint();
    FlumeRpcEndpoint(int port);

    ~FlumeRpcEndpoint();

    /** Initialize to listen on the given port. */
    void init(int port);
    
    void shutdown();

    /** Function called when we get a Flume message. */
    boost::function<void (int64_t timestamp,
                          int priority,
                          const std::string & body,
                          int64_t nanos,
                          const std::string & host,
                          const std::map<std::string, std::string> & meta)>
    onFlumeMessage;

    /** Function to be called when we get a close message. */
    boost::function<void ()> onClose;

    /** Function to record that something has happened. */
    void recordEvent(const std::string & event,
                     EventType type = ET_COUNT,
                     float value = 1.0)
    {
        if (onEvent)
            onEvent(event, type, value);
    }

    std::function<void (std::string, EventType, float)> onEvent;

private:
    struct Handler;
    boost::shared_ptr<Handler> handler;
};

} // namespace Datacratic

#endif /* __logger__flume_endpoint_h__ */
