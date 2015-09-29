/* chunked_http_endpoint.h                                          -*- C++ -*-
   Jeremy Barnes, 26 April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#pragma once

#include "soa/service/http_endpoint.h"
#include "soa/service/stats_events.h"

namespace Datacratic {

class ChunkedHttpEndpoint;


/*****************************************************************************/
/* CHUNKED HTTP HANDLER                                                      */
/*****************************************************************************/

struct ChunkedHttpHandler
    : public HttpConnectionHandler,
      public std::enable_shared_from_this<ChunkedHttpHandler> {

    ChunkedHttpHandler();
    virtual ~ChunkedHttpHandler();

    /** We got our transport. */
    virtual void onGotTransport();

    ChunkedHttpEndpoint * endpoint;

    /** Deal with the chunk.  Default calls onChunk from the handler. */
    virtual void handleHttpChunk(const HttpHeader & httpHeader,
                                 const std::string & chunkHeader,
                                 const std::string & chunk);

    /** Got a disconnection */
    virtual void handleDisconnect();

    virtual std::string status() const;

    /** We got an error.  Send back an HTTP error response. */
    virtual void handleError(const std::string & message);

    void doEvent(const char * eventName,
                 StatEventType type = ET_COUNT,
                 float value = 1.0,
                 const char * units = "");

    std::shared_ptr<ChunkedHttpHandler> makeNewHandlerShared();
    
    static long created;
    static long destroyed;
};



/*****************************************************************************/
/* CHUNKED HTTP ENDPOINT                                                     */
/*****************************************************************************/

struct ChunkedHttpEndpoint: public HttpEndpoint {

    /** Function that will be called to notify of a new auction. */
    typedef boost::function<void (const HttpHeader & header,
                                  const std::string & chunkHeader,
                                  const std::string & chunkData)>
        OnChunk;
    OnChunk onChunk;
    
    ChunkedHttpEndpoint(const std::string & name, OnChunk onChunk);
    virtual ~ChunkedHttpEndpoint();

    /** A function that deals with something happening.  Fields are:
        1.  The name of the event;
        2.  The type of the event;
        3.  The value of the event if it's a measurement;
        4.  The units of the event (currently unused)
    */
    typedef boost::function<void (std::string,
                                  StatEventType,
                                  float,
                                  const char *)> OnEvent;
    OnEvent onEvent;
    
protected:
    void doEvent(const std::string & eventName,
                 StatEventType type = ET_COUNT,
                 float value = 1.0,
                 const char * units = "")
    {
        if (!onEvent) return;
        onEvent(eventName, type, value, units);
    }
    
    virtual std::shared_ptr<ConnectionHandler> makeNewHandler();
    virtual std::shared_ptr<ChunkedHttpHandler> makeNewHandlerShared();

private:
    friend class ChunkedHttpHandler;
};


} // namespace Datacratic
