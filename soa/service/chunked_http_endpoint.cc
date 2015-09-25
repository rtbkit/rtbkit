/* chunked_http_endpoint.cc
   Jeremy Barnes, 7 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Endpoint for chunked http requests.
*/

#include "soa/service/chunked_http_endpoint.h"
#include <boost/make_shared.hpp>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* CHUNKED HTTP HANDLER                                                      */
/*****************************************************************************/

long ChunkedHttpHandler::created = 0;
long ChunkedHttpHandler::destroyed = 0;

ChunkedHttpHandler::
ChunkedHttpHandler()
{
    atomic_add(created, 1);
}

ChunkedHttpHandler::
~ChunkedHttpHandler()
{
    atomic_add(destroyed, 1);
}

void
ChunkedHttpHandler::
onGotTransport()
{
    this->endpoint = dynamic_cast<ChunkedHttpEndpoint *>(get_endpoint());

    if (!this->endpoint)
        throw Exception("ChunkedHttpHandler needs to be owned by an "
                        "ChunkedHttpEndpoint");

    HttpConnectionHandler::onGotTransport();
    
    startReading();
}

void
ChunkedHttpHandler::
handleDisconnect()
{
    doEvent("disconnection");
    closeWhenHandlerFinished();
}

void
ChunkedHttpHandler::
handleHttpChunk(const HttpHeader & header,
                const std::string & chunkHeader,
                const std::string & chunk)
{
    endpoint->onChunk(header, chunkHeader, chunk);
}

void
ChunkedHttpHandler::
handleError(const std::string & message)
{
    doEvent("error");

    send(ML::format("HTTP/1.1 400 Bad Request\r\n"
                    "Content-Type: text/plain\r\n"
                    "Content-Length: %zd\r\n"
                    "Connection: Close\r\n"
                    "\r\n"
                    "%s", message.size(), message.c_str()),
         NEXT_CLOSE);
}

std::string
ChunkedHttpHandler::
status() const
{
    return "ChunkedHttpHandler";
}

void
ChunkedHttpHandler::
doEvent(const char * eventName,
        StatEventType type,
        float value,
        const char * units)
{
    endpoint->doEvent(eventName, type, value, units);
}

std::shared_ptr<ChunkedHttpHandler>
ChunkedHttpHandler::
makeNewHandlerShared()
{
    return endpoint->makeNewHandlerShared();
}


/*****************************************************************************/
/* CHUNKED HTTP ENDPOINT                                                     */
/*****************************************************************************/

ChunkedHttpEndpoint::
ChunkedHttpEndpoint(const std::string & name, OnChunk onChunk)
    : HttpEndpoint("name"),
      onChunk(onChunk)
{
    // Link up events
    onTransportOpen = [=] (TransportBase *)
        {
            this->doEvent("newConnection");
        };

    onTransportClose = [=] (TransportBase *)
        {
            this->doEvent("closedConnection");
        };
}

ChunkedHttpEndpoint::
~ChunkedHttpEndpoint()
{
}

std::shared_ptr<ConnectionHandler>
ChunkedHttpEndpoint::
makeNewHandler()
{
    return makeNewHandlerShared();
}

std::shared_ptr<ChunkedHttpHandler>
ChunkedHttpEndpoint::
makeNewHandlerShared()
{
    return std::make_shared<ChunkedHttpHandler>();
}

} // namespace Datacratic
