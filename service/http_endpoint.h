/* http_endpoint.h                                                 -*- C++ -*-
   Jeremy Barnes, 18 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Endpoint for generic HTTP connections.
*/

#pragma once

#include "soa/service/passive_endpoint.h"
#include "soa/types/date.h"
#include "http_header.h"
#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>

namespace Datacratic {


class HttpEndpoint;


/*****************************************************************************/
/* HTTP RESPONSE                                                             */
/*****************************************************************************/

/** Structure used to return an HTTP response.
    TODO: make use of the the HttpHeader class
*/

struct HttpResponse {
    HttpResponse(int responseCode,
                 std::string contentType,
                 std::string body,
                 std::vector<std::pair<std::string, std::string> > extraHeaders
                     = std::vector<std::pair<std::string, std::string> >())
        : responseCode(responseCode),
          responseStatus(getResponseReasonPhrase(responseCode)),
          contentType(contentType),
          body(body),
          extraHeaders(extraHeaders),
          sendBody(true)
    {
    }

    /** Construct an HTTP response header only, with no body.  No content-
        length will be inferred. */

    HttpResponse(int responseCode,
                 std::string contentType,
                 std::vector<std::pair<std::string, std::string> > extraHeaders
                     = std::vector<std::pair<std::string, std::string> >())
        : responseCode(responseCode),
          responseStatus(getResponseReasonPhrase(responseCode)),
          contentType(contentType),
          extraHeaders(extraHeaders),
          sendBody(false)
    {
    }

    HttpResponse(int responseCode,
                 Json::Value body,
                 std::vector<std::pair<std::string, std::string> > extraHeaders
                     = std::vector<std::pair<std::string, std::string> >())
        : responseCode(responseCode),
          responseStatus(getResponseReasonPhrase(responseCode)),
          contentType("application/json"),
          body(boost::trim_copy(body.toString())),
          extraHeaders(extraHeaders),
          sendBody(true)
    {
    }

    int responseCode;
    std::string responseStatus;
    std::string contentType;
    std::string body;
    std::vector<std::pair<std::string, std::string> > extraHeaders;
    bool sendBody;
};


/*****************************************************************************/
/* HTTP CONNECTION HANDLER                                                   */
/*****************************************************************************/

/** A connection handler that deals with HTTP connections.

    This will handle parsing the header, but will forward the data off
    to another slave handler.  It's the HTTP Endpoint's responsibility to
    generate a slave handler once a header is received.
*/

struct HttpConnectionHandler : public PassiveConnectionHandler {

    HttpConnectionHandler();

    enum ReadState {
        INVALID,
        HEADER,
        PAYLOAD,       // non chunk only
        CHUNK_HEADER,  // chunk only
        CHUNK_BODY,    // chunk only
        DONE
    } readState;

    /** Accumulated text for the header. */
    std::string headerText;

    /** The actual header */
    HttpHeader header;

    /** The payload we're accumulating. */
    std::string payload;

    /** The chunk header we're accumulating. */
    std::string chunkHeader;
    size_t chunkSize;
    std::string chunkBody;

    /** When we first got data. */
    Date firstData;

    HttpEndpoint * httpEndpoint;

    virtual void onGotTransport();

    /** Create a new connection handler.  Delegates to the endpoint.  This
        is used after a response is sent to set the connection up for a
        new request.
    */
    std::shared_ptr<ConnectionHandler> makeNewHandlerShared();

    //virtual void handleNewConnection();
    virtual void handleData(const std::string & data);
    virtual void handleError(const std::string & message);
    virtual void onCleanup();

    /** Called when the HTTP header comes through.  Default will pass it
        back to the endpoint to do something with it.
    */
    virtual void handleHttpHeader(const HttpHeader & header);

    /** Called for each packet of data that comes through.  Default
        concatenates them together into a payload and calls
        handleHttpPayload once done.
    */
    virtual void handleHttpData(const std::string & data);

    /** Called once the entire payload has come through.  Default will
        throw.  Will be called multiple times for chunked encoding.
    */
    virtual void handleHttpPayload(const HttpHeader & header,
                                   const std::string & payload);

    /** Called when a chunk comes through.  Default will call
        handleHttpPayload.
    */
    virtual void handleHttpChunk(const HttpHeader & header,
                                 const std::string & chunkHeader,
                                 const std::string & chunk);

    /** Send an HTTP chunk with the appropriate headers back down the
        wire. */
    void sendHttpChunk(const std::string & chunk,
                       NextAction next = NEXT_CONTINUE,
                       OnWriteFinished onWriteFinished = OnWriteFinished());

    /** Handle sending an HTTP response.

        Calls the given callback once done.
    */
    virtual void putResponseOnWire(HttpResponse response,
                                   std::function<void ()> onSendFinished
                                   = std::function<void ()>(),
                                   NextAction next = NEXT_CONTINUE);

};


/*****************************************************************************/
/* HTTP ENDPOINT                                                             */
/*****************************************************************************/

/** An endpoint that deals with HTTP. */

struct HttpEndpoint: public PassiveEndpointT<SocketTransport> {

    HttpEndpoint(const std::string & name);

    virtual ~HttpEndpoint();

    typedef std::function<std::shared_ptr<ConnectionHandler> ()>
    HandlerFactory;

    HandlerFactory handlerFactory;

    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler()
    {
        if (handlerFactory)
            return handlerFactory();
        return std::make_shared<HttpConnectionHandler>();
    }
};

} // namespace Datacratic
