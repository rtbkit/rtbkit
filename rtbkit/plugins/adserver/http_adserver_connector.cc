/* http_adserver_connector.cc                                       -*- C++ -*-
   Wolfgang Sourdeau, April 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/


#include <memory>

#include "http_adserver_connector.h"


using namespace std;
using namespace RTBKIT;


/****************************************************************************/
/* HTTPADSERVERCONNECTIONHANDLER                                            */
/****************************************************************************/

HttpAdServerConnectionHandler::
HttpAdServerConnectionHandler(HttpAdServerHttpEndpoint & endpoint,
                              const HttpAdServerRequestCb & requestCb)
    : endpoint_(endpoint), requestCb_(requestCb)
{
}

void
HttpAdServerConnectionHandler::
handleUnknownHeader(const HttpHeader& header) {
    if (header.resource == "/ready") {
        putResponseOnWire(HttpResponse(200, "text/plain", "1"));
        return;
    }

    throw ML::Exception("Unknown resource '" + header.resource + "'");
}

void
HttpAdServerConnectionHandler::
handleJson(const HttpHeader & header, const Json::Value & json,
           const string & jsonStr)
{
    string resultMsg;

    transport().assertLockedByThisThread();

    auto onSendFinished = [&] () {
        Date endRq = Date::now();
        double timeElapsedMs = endRq.secondsSince(this->firstData) * 1000;
        endpoint_.doEvent("rqTimeMs", ET_OUTCOME, timeElapsedMs);

        this->transport().associateWhenHandlerFinished
        (endpoint_.makeNewHandler(), "rqFinished");
    };

    try {
        HttpAdServerResponse returnValue = requestCb_(header, json, jsonStr);
        if(returnValue.valid) {
            resultMsg = ("HTTP/1.1 200 OK\r\n"
                     "Content-Type: none\r\n"
                     "Content-Length: 0\r\n"
                     "\r\n");
        }
        else {
            endpoint_.doEvent("error.rqParsingError");
            resultMsg = sendErrorResponse(returnValue.error, returnValue.details, json);
        }
    }
    catch (const exception & exc) {
        cerr << "error parsing adserver request " << json << ": "
             << exc.what() << endl;
        endpoint_.doEvent("error.rqParsingError");
        resultMsg = sendErrorResponse("error parsing AdServer message", exc.what(), json);
    }

    send(resultMsg,
         NEXT_CONTINUE,
         onSendFinished);
}


std::string
HttpAdServerConnectionHandler::
sendErrorResponse(const std::string & error, const std::string & details, const Json::Value & json)
{
    std::string resultMsg;
    Json::Value responseJson;
    
    responseJson["error"] =error;
    responseJson["message"] = json;
    responseJson["details"] = details;

    string response = responseJson.toString();
    resultMsg = ML::format("HTTP/1.1 400 Bad Request\r\n"
                               "Content-Type: text/json\r\n"
                               "Content-Length: %zd\r\n"
                               "\r\n%s",
                               response.size(), response.c_str());
    return resultMsg;
}

/****************************************************************************/
/* HTTPADSERVERHTTPENDPOINT                                                 */
/****************************************************************************/

HttpAdServerHttpEndpoint::
HttpAdServerHttpEndpoint(int port, const HttpAdServerRequestCb & requestCb)
    : HttpEndpoint("adserver-ep-" + to_string(port)),
      port_(port), requestCb_(requestCb)
{
}

HttpAdServerHttpEndpoint::
HttpAdServerHttpEndpoint(HttpAdServerHttpEndpoint && otherEndpoint)
: HttpEndpoint("adserver-ep-" + to_string(otherEndpoint.port_))
{
    port_ = otherEndpoint.port_;
    requestCb_ = otherEndpoint.requestCb_;
}

HttpAdServerHttpEndpoint::
~HttpAdServerHttpEndpoint()
{
    shutdown();
}

HttpAdServerHttpEndpoint &
HttpAdServerHttpEndpoint::
operator = (const HttpAdServerHttpEndpoint& other)
{
    if (this != &other) {
        port_ = other.port_;
        requestCb_ = other.requestCb_;
    }

    return *this;
}

int
HttpAdServerHttpEndpoint::
getPort()
    const
{
    return port_;
}

shared_ptr<ConnectionHandler>
HttpAdServerHttpEndpoint::
makeNewHandler()
{
    return std::make_shared<HttpAdServerConnectionHandler>(*this, requestCb_);
}


/****************************************************************************/
/* HTTPADSERVERCONNECTOR                                                    */
/****************************************************************************/

HttpAdServerConnector::
HttpAdServerConnector(const string & serviceName,
                      const shared_ptr<Datacratic::ServiceProxies> & proxy)
    : AdServerConnector(serviceName, proxy)
{
}

void
HttpAdServerConnector::
registerEndpoint(int port, const HttpAdServerRequestCb & requestCb)
{
    endpoints_.emplace_back(port, requestCb);
}

void
HttpAdServerConnector::
init(const shared_ptr<ConfigurationService> & config)
{
    AdServerConnector::init(config);
    for (HttpAdServerHttpEndpoint & endpoint: endpoints_) {
        auto onEvent = bind(&ServiceBase::recordEvent, this,
                            placeholders::_1, placeholders::_2,
                            placeholders::_3, placeholders::_4);
        endpoint.onEvent = onEvent;
    }
}

void
HttpAdServerConnector::
shutdown()
{
    for (HttpAdServerHttpEndpoint & endpoint: endpoints_) {
        endpoint.shutdown();
    }
    AdServerConnector::shutdown();
}

void
HttpAdServerConnector::
bindTcp()
{
    for (HttpAdServerHttpEndpoint & endpoint: endpoints_) {
        endpoint.init(endpoint.getPort(), "0.0.0.0", 4);
    }
}

void
HttpAdServerConnector::
start()
{
    for (HttpAdServerHttpEndpoint & endpoint: endpoints_) {
        endpoint.makeRealTime(10);
    }
    AdServerConnector::start();
}

std::vector<int>
HttpAdServerConnector::
ports() const
{
    vector<int> ports;

    for (const auto & ep: endpoints_)
        ports.push_back(ep.port());

    return ports;
}
