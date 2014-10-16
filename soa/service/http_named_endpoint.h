/* http_named_endpoint.h                                           -*- C++ -*-
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
*/

#pragma once

#include "soa/service/http_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "named_endpoint.h"
#include "http_rest_proxy.h"
#include <boost/make_shared.hpp>


namespace Datacratic {


/*****************************************************************************/
/* HTTP NAMED ENDPOINT                                                       */
/*****************************************************************************/

/** A message loop-compatible endpoint for http connections. */

struct HttpNamedEndpoint : public NamedEndpoint, public HttpEndpoint {

    HttpNamedEndpoint();

    /** Set the Access-Control-Allow-Origin: * HTTP header */
    void allowAllOrigins();

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName);

    /** Bid into a given address.  Address is host:port.

        If no port is given (and no colon), than use any port number.
        If port is a number and then "+", then scan for any port higher than
        the given number.
        If host is empty or "*", then use all interfaces.
    */
    std::string
    bindTcpAddress(const std::string & address);

    /** Bind into a specific tcp port.  If the port is not available, it will
        throw an exception.

        Returns the uri to connect to.
    */
    std::string
    bindTcpFixed(std::string host, int port);

    /** Bind into a tcp port.  If the preferred port is not available, it will
        scan until it finds one that is.

        Returns the uri to connect to.
    */
    std::string
    bindTcp(PortRange const & portRange, std::string host = "");

    struct RestConnectionHandler: public HttpConnectionHandler {
        RestConnectionHandler(HttpNamedEndpoint * endpoint);

        HttpNamedEndpoint * endpoint;
        std::weak_ptr<RestConnectionHandler> sharedThis;

        virtual void
        handleHttpPayload(const HttpHeader & header,
                          const std::string & payload);

        /** Called when the other end disconnects from us.  We set the
            zombie flag and stop anything else from happening on the
            socket once we're done.
        */
        virtual void handleDisconnect();

        void sendErrorResponse(int code, const std::string & error);

        void sendErrorResponse(int code, const Json::Value & error);

        void sendResponse(int code,
                          const Json::Value & response,
                          const std::string & contentType = "application/json",
                          RestParams headers = RestParams());

        void sendResponse(int code,
                          const std::string & body,
                          const std::string & contentType,
                          RestParams headers = RestParams());

        void sendResponseHeader(int code,
                                const std::string & contentType,
                                RestParams headers = RestParams());

        /** Send an HTTP chunk with the appropriate headers back down the
            wire. */
        void sendHttpChunk(const std::string & chunk,
                           NextAction next = NEXT_CONTINUE,
                           OnWriteFinished onWriteFinished = OnWriteFinished());

        /** Send the entire HTTP payload.  Its length must match that of
            the response header.
        */
        void sendHttpPayload(const std::string & str);

        mutable std::mutex mutex;

    public:
        /// If this is true, the connection has no transport
        std::atomic<bool> isZombie;
    };

    typedef std::function<void (std::shared_ptr<RestConnectionHandler> connection,
                                const HttpHeader & header,
                                const std::string & payload)> OnRequest;

    OnRequest onRequest;

    std::vector<std::pair<std::string, std::string> > extraHeaders;

    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler();
};


/*****************************************************************************/
/* HTTP NAMED REST PROXY                                                     */
/*****************************************************************************/

/** Proxy to connect to a named http-based service. */

struct HttpNamedRestProxy: public HttpRestProxy {

    void init(std::shared_ptr<ConfigurationService> config);

    bool connectToServiceClass(const std::string & serviceClass,
                               const std::string & endpointName,
                               bool local = true);

    bool connect(const std::string & endpointName);

    /** Called back when one of our endpoints either changes or disappears. */
    bool onConfigChange(ConfigurationService::ChangeType change,
                        const std::string & key,
                        const Json::Value & newValue);


private:
    std::shared_ptr<ConfigurationService> config;

    bool connected;
    std::string serviceClass;
    std::string endpointName;
};

} // namespace Datacratic

