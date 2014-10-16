/* rest_service_endpoint.cc
   Jeremy Barnes, 11 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Endpoint to talk with a REST service.
*/

#include "rest_service_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "city.h"

using namespace std;

namespace Datacratic {


/*****************************************************************************/
/* REST REQUEST                                                              */
/*****************************************************************************/

std::ostream & operator << (std::ostream & stream, const RestRequest & request)
{
    return stream << request.verb << " " << request.resource << endl
                  << request.params << endl
                  << request.payload;
}



/*****************************************************************************/
/* REST SERVICE ENDPOINT CONNECTION ID                                       */
/*****************************************************************************/

void
RestServiceEndpoint::ConnectionId::
sendResponse(int responseCode,
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

void
RestServiceEndpoint::ConnectionId::
sendResponse(int responseCode,
                  const Json::Value & response,
                  const std::string & contentType) const
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

void
RestServiceEndpoint::ConnectionId::
sendErrorResponse(int responseCode,
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

void
RestServiceEndpoint::ConnectionId::
sendErrorResponse(int responseCode, const Json::Value & error) const
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

void
RestServiceEndpoint::ConnectionId::
sendRedirect(int responseCode, const std::string & location) const
{
    if (itl->responseSent)
        throw ML::Exception("response already sent");

    if (itl->endpoint->logResponse)
        itl->endpoint->logResponse(*this, responseCode, location,
                                   "REDIRECT");

    if (itl->http)
        itl->http->sendResponse(responseCode, string(""), "",
                                { { "Location", location } });
    else {
        throw ML::Exception("zeromq redirects not done yet");
    }

    itl->responseSent = true;
}

void
RestServiceEndpoint::ConnectionId::
sendHttpResponse(int responseCode,
                 const std::string & response,
                 const std::string & contentType,
                 const RestParams & headers) const
{
    if (itl->responseSent)
        throw ML::Exception("response already sent");

    if (itl->endpoint->logResponse)
        itl->endpoint->logResponse(*this, responseCode, response,
                                   contentType);

    if (itl->http)
        itl->http->sendResponse(responseCode, response, contentType,
                                headers);
    else {
        throw ML::Exception("zeromq redirects not done yet");
    }

    itl->responseSent = true;
}

void
RestServiceEndpoint::ConnectionId::
sendHttpResponseHeader(int responseCode,
                       const std::string & contentType,
                       ssize_t contentLength,
                       const RestParams & headers_) const
{
    if (itl->responseSent)
        throw ML::Exception("response already sent");

    if (!itl->http)
        throw ML::Exception("sendHttpResponseHeader only works on HTTP connections");

    if (itl->endpoint->logResponse)
        itl->endpoint->logResponse(*this, responseCode, "", contentType);

    RestParams headers = headers_;
    if (contentLength == CHUNKED_ENCODING) {
        itl->chunkedEncoding = true;
        headers.push_back({"Transfer-Encoding", "chunked"});
    }
    else if (contentLength >= 0) {
        headers.push_back({"Content-Length", to_string(contentLength) });
    }
    else {
        itl->keepAlive = false;
    }

    itl->http->sendResponseHeader(responseCode, contentType, headers);
}

void
RestServiceEndpoint::ConnectionId::
sendPayload(const std::string & payload)
{
    if (itl->chunkedEncoding) {
        if (payload.empty()) {
            throw ML::Exception("Can't send empty chunk over a chunked connection");
        }
        itl->http->sendHttpChunk(payload, HttpConnectionHandler::NEXT_CONTINUE);
    }
    else itl->http->sendHttpPayload(payload);
}

void
RestServiceEndpoint::ConnectionId::
finishResponse()
{
    if (itl->chunkedEncoding) {
        itl->http->sendHttpChunk("", HttpConnectionHandler::NEXT_CLOSE);
    }
    else if (!itl->keepAlive) {
        itl->http->send("", HttpConnectionHandler::NEXT_CLOSE);
    } else {
        itl->http->send("", HttpConnectionHandler::NEXT_RECYCLE);
    }

    itl->responseSent = true;
}


/*****************************************************************************/
/* REST SERVICE ENDPOINT                                                     */
/*****************************************************************************/

RestServiceEndpoint::
RestServiceEndpoint(std::shared_ptr<zmq::context_t> context)
    : zmqEndpoint(context)
{
}

RestServiceEndpoint::
~RestServiceEndpoint()
{
    shutdown();
}

void 
RestServiceEndpoint::
shutdown()
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

void
RestServiceEndpoint::
init(std::shared_ptr<ConfigurationService> config,
     const std::string & endpointName,
     double maxAddedLatency,
     int numThreads)
{
    MessageLoop::init(numThreads, maxAddedLatency);
    zmqEndpoint.init(config, ZMQ_XREP, endpointName + "/zeromq");
    httpEndpoint.init(config, endpointName + "/http");

    auto zmqHandler = [=] (std::vector<std::string> && message)
        {
            using namespace std;

            if (message.size() < 6) {
                cerr << "ignored message with invalid number of members:"
                     << message.size()
                     << endl;
                return;
            }
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
        = [=] (std::shared_ptr<HttpNamedEndpoint::RestConnectionHandler> connection,
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

std::pair<std::string, std::string>
RestServiceEndpoint::
bindTcp(PortRange const & zmqRange, PortRange const & httpRange,
        std::string host)
{
    std::string httpAddr = httpEndpoint.bindTcp(httpRange, host);
    std::string zmqAddr = zmqEndpoint.bindTcp(zmqRange, host);
    return std::make_pair(zmqAddr, httpAddr);
}

void
RestServiceEndpoint::
handleRequest(const ConnectionId & connection,
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

std::string
RestServiceEndpoint::
getHttpRequestId() const
{
    std::string s = Date::now().print(9) + ML::format("%d", random());
    uint64_t jobId = CityHash64(s.c_str(), s.size());
    return ML::format("%016llx", jobId);
}

void
RestServiceEndpoint::
logToStream(std::ostream & stream)
{
    auto logLock = std::make_shared<std::mutex>();

    logRequest = [=,&stream] (const ConnectionId & conn, const RestRequest & req)
        {
            std::unique_lock<std::mutex> guard(*logLock);

            stream << "--> ------------------------- new request "
            << conn.itl->requestId
            << " at " << conn.itl->startDate.print(9) << endl;
            stream << req << endl << endl;
        };

    logResponse = [=,&stream] (const ConnectionId & conn,
                       int code,
                       const std::string & resp,
                       const std::string & contentType)
        {
            std::unique_lock<std::mutex> guard(*logLock);

            Date now = Date::now();

            stream << "<-- ========================= finished request "
            << conn.itl->requestId
            << " at " << now.print(9)
            << ML::format(" (%.3fms)", conn.itl->startDate.secondsUntil(now) * 1000)
            << endl;
            stream << code << " " << contentType << " " << resp.length() << " bytes" << endl;;
                
            if (resp.size() <= 16384)
                stream << resp;
            else {
                stream << string(resp, 0, 16386) << "...";
            }
            stream << endl << endl;
        };
}


} // namespace Datacratic
