/* json_service_endpoint.h                                         -*- C++ -*-
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#ifndef __service__zmq_json_endpoint_h__
#define __service__zmq_json_endpoint_h__

#include "zmq_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "http_named_endpoint.h"


namespace Datacratic {


/*****************************************************************************/
/* REST REQUEST                                                              */
/*****************************************************************************/

struct RestRequest {
    RestRequest()
    {
    }

    RestRequest(const HttpHeader & header,
                const std::string & payload)
        : header(header),
          verb(header.verb),
          resource(header.resource),
          params(header.queryParams),
          payload(payload)
    {
    }

    RestRequest(const std::string & verb,
                const std::string & resource,
                const RestParams & params,
                const std::string & payload)
        : verb(verb), resource(resource), params(params), payload(payload)
    {
    }

    HttpHeader header;
    std::string verb;
    std::string resource;
    RestParams params;
    std::string payload;
};

std::ostream & operator << (std::ostream & stream, const RestRequest & request);


/*****************************************************************************/
/* REST SERVICE ENDPOINT                                                     */
/*****************************************************************************/

/** This class exposes an API for a given service via:
    - zeromq
    - http

    It allows both synchronous and asynchronous responses.

    The location of the endpoints are published via a configuration service
    (usually Zookeeper).
*/

struct RestServiceEndpoint: public MessageLoop {

    /** Start the service with the given parameters.  If the ports are given,
        then the service will bind to those specific ports for the given
        endpoints, and so no service discovery will need to be done.
    */
    RestServiceEndpoint(std::shared_ptr<zmq::context_t> context);

    virtual ~RestServiceEndpoint();

    void shutdown();

    /** Defines a connection: either a zeromq connection (identified by its
        zeromq identifier) or an http connection (identified by its
        connection handler object).
    */
    struct ConnectionId {
        /// Don't initialize for now
        ConnectionId()
        {
        }
        
        /// Initialize for zeromq
        ConnectionId(const std::string & zmqAddress,
                     const std::string & requestId,
                     RestServiceEndpoint * endpoint)
            : itl(new Itl(zmqAddress, requestId, endpoint))
        {
        }

        /// Initialize for http
        ConnectionId(std::shared_ptr<HttpNamedEndpoint::RestConnectionHandler> http,
                     const std::string & requestId,
                     RestServiceEndpoint * endpoint)
            : itl(new Itl(http, requestId, endpoint))
        {
        }

        struct Itl {
            Itl(std::shared_ptr<HttpNamedEndpoint::RestConnectionHandler> http,
                const std::string & requestId,
                RestServiceEndpoint * endpoint)
                : requestId(requestId),
                  http(http),
                  endpoint(endpoint),
                  responseSent(false),
                  startDate(Date::now()),
                  chunkedEncoding(false),
                  keepAlive(true)
            {
            }

            Itl(const std::string & zmqAddress,
                const std::string & requestId,
                RestServiceEndpoint * endpoint)
                : zmqAddress(zmqAddress),
                  requestId(requestId),
                  http(0),
                  endpoint(endpoint),
                  responseSent(false),
                  startDate(Date::now()),
                  chunkedEncoding(false),
                  keepAlive(true)
            {
            }

            ~Itl()
            {
                if (!responseSent)
                    throw ML::Exception("no response sent on connection");
            }

            std::string zmqAddress;
            std::string requestId;
            std::shared_ptr<HttpNamedEndpoint::RestConnectionHandler> http;
            RestServiceEndpoint * endpoint;
            bool responseSent;
            Date startDate;
            bool chunkedEncoding;
            bool keepAlive;

            /** Data that is maintained with the connection.  This is where control
                data required for asynchronous or long-running connections can be
                put.
            */
            std::vector<std::shared_ptr<void> > piggyBack;
        };

        std::shared_ptr<Itl> itl;

        void sendResponse(int responseCode,
                          const char * response,
                          const std::string & contentType) const
        {
            return sendResponse(responseCode, std::string(response),
                                contentType);
        }

        /** Send the given response back on the connection. */
        void sendResponse(int responseCode,
                          const std::string & response,
                          const std::string & contentType) const;

        /** Send the given response back on the connection. */
        void sendResponse(int responseCode,
                          const Json::Value & response,
                          const std::string & contentType = "application/json") const;

        void sendResponse(int responseCode) const
        {
            return sendResponse(responseCode, "", "");
        }

        void sendRedirect(int responseCode, const std::string & location) const;

        /** Send an HTTP-only response with the given headers.  If it's not
            an HTTP connection, this will fail.
        */
        void sendHttpResponse(int responseCode,
                              const std::string & response,
                              const std::string & contentType,
                              const RestParams & headers) const;

        enum {
            UNKNOWN_CONTENT_LENGTH = -1,
            CHUNKED_ENCODING = -2
        };

        /** Send an HTTP-only response header.  This will not close the connection.  A
            contentLength of -1 means don't send it (for when the content length is
            not known ahead of time).  A contentLength of -2 means used HTTP chunked
            transfer encoding
        */
        void sendHttpResponseHeader(int responseCode,
                                    const std::string & contentType,
                                    ssize_t contentLength,
                                    const RestParams & headers = RestParams()) const;

        /** Send a payload (or a chunk of a payload) for an HTTP connection. */
        void sendPayload(const std::string & payload);

        /** Finish the response, recycling or closing the connection. */
        void finishResponse();

        /** Send the given error string back on the connection. */
        void sendErrorResponse(int responseCode,
                               const std::string & error,
                               const std::string & contentType) const;

        void sendErrorResponse(int responseCode, const char * error,
                               const std::string & contentType) const
        {
            sendErrorResponse(responseCode, std::string(error), "application/json");
        }

        void sendErrorResponse(int responseCode, const Json::Value & error) const;

        bool responseSent() const
        {
            return itl->responseSent;
        }

        bool isConnected() const
        {
            if (itl->http)
                return !itl->http->isZombie;  // NOTE: race condition
            else return true;  // zmq is always "connected"
        }
    };

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName,
              double maxAddedLatency = 0.005,
              int numThreads = 1);

    /** Bind to TCP/IP ports.  There is one for zeromq and one for
        http.
    */
    std::pair<std::string, std::string>
    bindTcp(PortRange const & zmqRange = PortRange(),
            PortRange const & httpRange = PortRange(),
            std::string host = "");

    /** Bind to a fixed URI for the HTTP endpoint.  This will throw an
        exception if it can't bind.

        example address: "*:4444", "localhost:8888"
    */
    std::string bindFixedHttpAddress(std::string host, int port)
    {
        return httpEndpoint.bindTcpFixed(host, port);
    }

    std::string bindFixedHttpAddress(std::string address)
    {
        return httpEndpoint.bindTcpAddress(address);
    }

    /// Request handler function type
    typedef std::function<void (ConnectionId connection,
                                RestRequest request)> OnHandleRequest;

    OnHandleRequest onHandleRequest;

    /** Handle a request.  Default implementation defers to onHandleRequest.
        Otherwise this method should be overridden.
    */
    virtual void handleRequest(const ConnectionId & connection,
                               const RestRequest & request) const;

    ZmqNamedEndpoint zmqEndpoint;
    HttpNamedEndpoint httpEndpoint;

    std::function<void (const ConnectionId & conn, const RestRequest & req) > logRequest;
    std::function<void (const ConnectionId & conn,
                        int code,
                        const std::string & resp,
                        const std::string & contentType) > logResponse;

    void doHandleRequest(const ConnectionId & connection,
                         const RestRequest & request)
    {
        if (logRequest)
            logRequest(connection, request);

        handleRequest(connection, request);
    }
    
    // Create a random request ID for an HTTP request
    std::string getHttpRequestId() const;

    /** Log all requests and responses to the given stream.  This function
        will overwrite logRequest and logResponse with new handlers.
    */
    void logToStream(std::ostream & stream);
};

} // namespace Datacratic

#endif /* __service__zmq_json_endpoint_h__ */
