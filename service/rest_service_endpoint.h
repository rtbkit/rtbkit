/* json_service_endpoint.h                                         -*- C++ -*-
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#ifndef __service__zmq_json_endpoint_h__
#define __service__zmq_json_endpoint_h__

#include "zmq_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "http_named_endpoint.h"
#include "city.h"


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
        : verb(header.verb),
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
    RestServiceEndpoint(std::shared_ptr<zmq::context_t> context)
        : zmqEndpoint(context)
    {
    }

    virtual ~RestServiceEndpoint()
    {
        shutdown();
    }

    void shutdown()
    {
        // 1.  Shut down the http endpoint, since it needs our threads to
        //     complete its shutdown
        httpEndpoint.shutdown();

        // 2.  Shut down the message loop
        MessageLoop::shutdown();

        // 3.  Shut down the zmq endpoint now we know that the message loop is not using
        //     it.
        zmqEndpoint.shutdown();
    }

    /** Defines a connection: either a zeromq connection (identified by its
        zeromq identifier) or an http connection (identified by its
        connection handler object).
    */
    struct ConnectionId {
        /// Initialize for zeromq
        ConnectionId(const std::string & zmqAddress,
                     const std::string & requestId,
                     RestServiceEndpoint * endpoint)
            : itl(new Itl(zmqAddress, requestId, endpoint))
        {
        }

        /// Initialize for http
        ConnectionId(HttpNamedEndpoint::RestConnectionHandler * http,
                     const std::string & requestId,
                     RestServiceEndpoint * endpoint)
            : itl(new Itl(http, requestId, endpoint))
        {
        }

        struct Itl {
            Itl(HttpNamedEndpoint::RestConnectionHandler * http,
                const std::string & requestId,
                RestServiceEndpoint * endpoint)
                : requestId(requestId),
                  http(http),
                  endpoint(endpoint),
                  responseSent(false),
                  startDate(Date::now())
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
                  startDate(Date::now())
            {
            }

            ~Itl()
            {
                if (!responseSent)
                    throw ML::Exception("no response sent on connection");
            }

            std::string zmqAddress;
            std::string requestId;
            HttpNamedEndpoint::RestConnectionHandler * http;
            RestServiceEndpoint * endpoint;
            bool responseSent;
            Date startDate;
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
                          const std::string & contentType) const
        {
            if (itl->responseSent)
                throw ML::Exception("response already sent");

            if (itl->endpoint->logResponse)
                itl->endpoint->logResponse(*this, responseCode, response,
                                      contentType);

            if (itl->http)
                itl->http->sendResponse(responseCode, response, contentType);
            else {
                std::vector<std::string> message;
                message.push_back(itl->zmqAddress);
                message.push_back(itl->requestId);
                message.push_back(std::to_string(responseCode));
                message.push_back(response);

                //std::cerr << "sending response to " << itl->requestId
                //          << std::endl;
                itl->endpoint->zmqEndpoint.sendMessage(message);
            }

            itl->responseSent = true;
        }

        /** Send the given response back on the connection. */
        void sendResponse(int responseCode,
                          const Json::Value & response,
                          const std::string & contentType = "application/json") const
        {
            using namespace std;
            //cerr << "sent response " << responseCode << " " << response
            //     << endl;

            if (itl->responseSent)
                throw ML::Exception("response already sent");

            if (itl->endpoint->logResponse)
                itl->endpoint->logResponse(*this, responseCode, response.toString(),
                                      contentType);

            if (itl->http)
                itl->http->sendResponse(responseCode, response, contentType);
            else {
                std::vector<std::string> message;
                message.push_back(itl->zmqAddress);
                message.push_back(itl->requestId);
                message.push_back(std::to_string(responseCode));
                message.push_back(response.toString());
                itl->endpoint->zmqEndpoint.sendMessage(message);
            }

            itl->responseSent = true;
        }

        void sendResponse(int responseCode) const
        {
            return sendResponse(responseCode, "", "");
        }

        /** Send the given error string back on the connection. */
        void sendErrorResponse(int responseCode,
                               const std::string & error,
                               const std::string & contentType) const
        {
            using namespace std;
            cerr << "sent error response " << responseCode << " " << error
                 << endl;

            if (itl->responseSent)
                throw ML::Exception("response already sent");


            if (itl->endpoint->logResponse)
                itl->endpoint->logResponse(*this, responseCode, error,
                                      contentType);
            
            if (itl->http)
                itl->http->sendResponse(responseCode, error);
            else {
                std::vector<std::string> message;
                message.push_back(itl->zmqAddress);
                message.push_back(itl->requestId);
                message.push_back(std::to_string(responseCode));
                message.push_back(error);
                itl->endpoint->zmqEndpoint.sendMessage(message);
            }

            itl->responseSent = true;
        }

        void sendErrorResponse(int responseCode, const char * error,
                               const std::string & contentType) const
        {
            sendErrorResponse(responseCode, std::string(error), "application/json");
        }

        void sendErrorResponse(int responseCode, const Json::Value & error) const
        {
            using namespace std;
            cerr << "sent error response " << responseCode << " " << error
                 << endl;

            if (itl->responseSent)
                throw ML::Exception("response already sent");

            if (itl->endpoint->logResponse)
                itl->endpoint->logResponse(*this, responseCode, error.toString(),
                                           "application/json");

            if (itl->http)
                itl->http->sendResponse(responseCode, error);
            else {
                std::vector<std::string> message;
                message.push_back(itl->zmqAddress);
                message.push_back(itl->requestId);
                message.push_back(std::to_string(responseCode));
                message.push_back(error.toString());
                itl->endpoint->zmqEndpoint.sendMessage(message);
            }

            itl->responseSent = true;
        }
    };

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName,
              double maxAddedLatency = 0.005,
              int numThreads = 1)
    {
        MessageLoop::init(numThreads, maxAddedLatency);
        zmqEndpoint.init(config, ZMQ_XREP, endpointName + "/zeromq");
        httpEndpoint.init(config, endpointName + "/http");

        auto zmqHandler = [=] (std::vector<std::string> && message)
            {
                using namespace std;
                //cerr << "got REST message at " << this << " " << message << endl;
                this->doHandleRequest(ConnectionId(message.at(0),
                                                   message.at(1),
                                                   this),
                                      RestRequest(message.at(2),
                                                  message.at(3),
                                                  RestParams::fromBinary(message.at(4)),
                                                  message.at(5)));
            };
        
        zmqEndpoint.messageHandler = zmqHandler;
        
        httpEndpoint.onRequest
            = [=] (HttpNamedEndpoint::RestConnectionHandler * connection,
                   const HttpHeader & header,
                   const std::string & payload)
            {
                std::string requestId = this->getHttpRequestId();
                this->doHandleRequest(ConnectionId(connection, requestId, this),
                                      RestRequest(header, payload));
            };
        
        addSource("RestServiceEndpoint::zmqEndpoint", zmqEndpoint);
        addSource("RestServiceEndpoint::httpEndpoint", httpEndpoint);

    }

    /** Bind to TCP/IP ports.  There is one for zeromq and one for
        http.  -1 means scan to find one.
    */
    std::pair<std::string, std::string>
    bindTcp(PortRange const & zmqRange = PortRange(), PortRange const & httpRange = PortRange(), std::string host = "")
    {
        std::string zmqAddr = zmqEndpoint.bindTcp(zmqRange, host);
        std::string httpAddr = httpEndpoint.bindTcp(httpRange, host);
        return std::make_pair(zmqAddr, httpAddr);
    }

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
                               const RestRequest & request) const
    {
        using namespace std;

        //cerr << "got request " << request << endl;
        if (onHandleRequest) {
            onHandleRequest(connection, request);
        }
        else {
            throw ML::Exception("need to override handleRequest or assign to "
                                "onHandleRequest");
        }
    }

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
    std::string getHttpRequestId() const
    {
        std::string s = Date::now().print(9) + ML::format("%d", random());
        uint64_t jobId = CityHash64(s.c_str(), s.size());
        return ML::format("%016llx", jobId);
    }
    
};

} // namespace Datacratic

#endif /* __service__zmq_json_endpoint_h__ */
