/* json_endpoint.h                                                 -*- C++ -*-
   Jeremy Barnes, 22 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Endpoint that does JSON.
*/

#pragma once


#include "soa/service//http_endpoint.h"
#include "soa/jsoncpp/json.h"


namespace Datacratic {


/*****************************************************************************/
/* JSON CONNECTION HANDLER                                                   */
/*****************************************************************************/

/** A connection handler that deals with JSON connections.

    This will handle parsing the header, but will forward the data off
    to another slave handler.  It's the HTTP Endpoint's responsibility to
    generate a slave handler once a header is received.
*/

struct JsonConnectionHandler : public HttpConnectionHandler {

    JsonConnectionHandler();

    virtual void handleHttpHeader(const HttpHeader & header);

    virtual void handleUnknownHeader(const HttpHeader& header);

    virtual void handleHttpPayload(const HttpHeader & header,
                                   const std::string & payload);


    /** Allow it to work with chunked headers too */
    virtual void handleHttpChunk(const HttpHeader & header,
                                 const std::string & chunkHeader,
                                 const std::string & chunk);

    /** Handler that deals with the actual JSON. */
    virtual void handleJson(const HttpHeader & header,
                            const Json::Value & json,
                            const std::string & jsonStr) = 0;
};


/*****************************************************************************/
/* ADHOC JSON CONNECTION HANDLER                                             */
/*****************************************************************************/

/** A JSON connection handler that allows an ad-hoc handler to be used. */

struct AdHocJsonConnectionHandler : public JsonConnectionHandler {

    typedef std::function<void (const HttpHeader & header,
                                const Json::Value & json,
                                const std::string & jsonStr,
                                AdHocJsonConnectionHandler * connection)>
    OnJson;

    AdHocJsonConnectionHandler()
    {
    }

    AdHocJsonConnectionHandler(OnJson onJson)
        : onJson(onJson)
    {
    }

    OnJson onJson;

    virtual void handleJson(const HttpHeader & header,
                            const Json::Value & json,
                            const std::string & jsonStr)
    {
        onJson(header, json, jsonStr, this);
    }
};

} // namespace Datacratic
