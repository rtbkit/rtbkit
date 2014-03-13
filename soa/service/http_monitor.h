/** Http_monitor.h                                 -*- C++ -*-
    Rémi Attab, 03 Aug 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Simple http monitoring service for a service.

    \todo Arg passing is pretty damn awkward right now.

    Essentially you pass a type through template arg in HttpMonitor and set
    the value in start. That value will be passed to the constructor of the
    handler. There's no real indication of this fancy mechanism which makes the
    whole thing fairly odd. Unfortunately there's no real way around this so...?

*/

#pragma once

#include "soa/service//http_endpoint.h"
#include "jml/arch/format.h"
#include "soa/jsoncpp/value.h"

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>


namespace Datacratic {


/******************************************************************************/
/* HTTP MONITOR                                                               */
/******************************************************************************/

/** Simple Http server that works with HttpMonitorHandler to provide a simple
    Http querying or monitering service.

*/
template<typename Handler, typename Arg = void*>
struct HttpMonitor : public HttpEndpoint
{
    HttpMonitor(const std::string& name)
        : HttpEndpoint(name)
    {}

    /** Starts the server on the given port. The given argument will be made
        available to the handlers whenever a connection is created.
    */
    void start(int port, Arg a = Arg())
    {
        arg = a;
        init(port, "0.0.0.0");
    }

private:

    Arg arg;

    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler()
    {
        return std::make_shared<Handler>(name(), arg);
    }
};


/******************************************************************************/
/* HTTP MONITOR HANDLER                                                       */
/******************************************************************************/

/** Connection handler for an HttpMonitor.

    Subclass and overwrite either the doGet() and doPost() function or only the
    handleHttpPayload() if you want access to the header.

    This template uses the curiosly recurring templat pattern so Handler should
    be the subclass.

*/
template<typename Handler, typename Arg = void*>
struct HttpMonitorHandler :
    public HttpConnectionHandler
{
    HttpMonitorHandler(const std::string& name, Arg arg);

    /** Override to access the full header. */
    virtual void
    handleHttpPayload(const HttpHeader & header, const std::string & payload);

    /** Handle a GET message on the given resource.

        Should call either sendResponse() or sendErrorResponse().
    */
    virtual void
    doGet(const std::string& resource) {}

    /** Handle a POST message on the given resource and payload.

        Should call either sendResponse() or sendErrorResponse().
    */
    virtual void
    doPost(const std::string& resource, const std::string& payload) {}

    /** Handle a DELETE message on the given resource and payload.

        Should call either sendResponse() or sendErrorResponse().
    */
    virtual void
    doDelete(const std::string& resource, const std::string& payload) {}

    void sendResponse(const Json::Value & response);
    void sendErrorResponse(int code, const std::string & error);
    void sendErrorResponse(int code, const Json::Value & error);

    Arg getArg() { return arg; }

    const std::string & name() const { return name_; }

private:
    Arg arg;
    std::string name_;
};


/******************************************************************************/
/* HTTP MONITOR HANDLER IMPL                                                  */
/******************************************************************************/

template<typename Handler, typename Arg>
HttpMonitorHandler<Handler, Arg>::
HttpMonitorHandler(
        const std::string& name, Arg arg) :
    HttpConnectionHandler(),
    arg(arg),
    name_(name)
{}

template<typename Handler, typename arg>
void
HttpMonitorHandler<Handler, arg>::
handleHttpPayload(const HttpHeader & header, const std::string & payload)
{
    try {
        if (header.verb == "GET")
            doGet(header.resource);

        else if (header.verb == "POST")
            doPost(header.resource, payload);

        else if (header.verb == "DELETE")
            doDelete(header.resource, payload);

        else sendErrorResponse(404, "bad verb " + header.verb);
    }
    catch(const std::exception& ex) {
        Json::Value response;
        response["error"] =
            "exception processing request "
            + header.verb + " " + header.resource;

        response["exception"] = ex.what();
        sendErrorResponse(400, response);
    }
    catch(...) {
        Json::Value response;
        response["error"] =
            "exception processing request "
            + header.verb + " " + header.resource;

        sendErrorResponse(400, response);
    }
}


template<typename Handler, typename arg>
void
HttpMonitorHandler<Handler, arg>::
sendErrorResponse(int code, const std::string & error)
{
    Json::Value val;
    val["error"] = error;
    sendErrorResponse(code, val);
}

template<typename Handler, typename arg>
void
HttpMonitorHandler<Handler, arg>::
sendErrorResponse(int code, const Json::Value & error)
{
    std::string encodedError = error.toString();
    send(ML::format("HTTP/1.1 %d Pants are on fire\r\n"
                    "Content-Type: application/json\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Content-Length: %zd\r\n"
                    "\r\n"
                    "%s",
                    code,
                    encodedError.length(),
                    encodedError.c_str()),
            NEXT_CLOSE);
}

template<typename Handler, typename Arg>
void
HttpMonitorHandler<Handler, Arg>::
sendResponse(const Json::Value & response)
{
    std::string body = response.toStyledString();

    auto onSendFinished = [=] {
        this->transport().associateWhenHandlerFinished
        (std::make_shared<Handler>(this->name(), this->arg),
         "sendResponse");
    };

    send(ML::format("HTTP/1.1 200 OK\r\n"
                    "Content-Type: application/json\r\n"
                    "Access-Control-Allow-Origin: *\r\n"
                    "Content-Length: %zd\r\n"
                    "Connection: Keep-Alive\r\n"
                    "\r\n"
                    "%s",
                    body.length(),
                    body.c_str()),
            NEXT_CONTINUE,
            onSendFinished);
}

} // Datacratic
