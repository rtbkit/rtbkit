/* http_adserver_connector.h                                       -*- C++ -*-
   Wolfgang Sourdeau, April 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/


#pragma once

#include <string>
#include <vector>
#include <initializer_list>

#include "soa/service/http_endpoint.h"
#include "soa/service/json_endpoint.h"

#include "adserver_connector.h"


namespace RTBKIT {


struct HttpAdServerResponse 
{
    HttpAdServerResponse() : valid(true) {
    }

    bool valid;
    std::string error;
    std::string details;
};

/****************************************************************************/
/* HTTPADSERVERCONNECTIONHANDLER                                            */
/****************************************************************************/

class HttpAdServerHttpEndpoint;

typedef std::function<HttpAdServerResponse (const HttpHeader & header,
                            const Json::Value & json,
                            const std::string & jsonStr)>
    HttpAdServerRequestCb;

struct HttpAdServerConnectionHandler
    : public Datacratic::JsonConnectionHandler {
    HttpAdServerConnectionHandler(HttpAdServerHttpEndpoint & endpoint,
                                  const HttpAdServerRequestCb & requestCb);

    virtual void handleUnknownHeader(const HttpHeader& header);

    virtual void handleJson(const HttpHeader & header,
                            const Json::Value & json,
                            const std::string & jsonStr);

private:
    std::string sendErrorResponse(const std::string & error, const std::string & details, const Json::Value & json);
    
    HttpAdServerHttpEndpoint & endpoint_;
    const HttpAdServerRequestCb & requestCb_;
};


/****************************************************************************/
/* HTTPADSERVERHTTPENDPOINT                                                 */
/****************************************************************************/

struct HttpAdServerHttpEndpoint : public Datacratic::HttpEndpoint {
    HttpAdServerHttpEndpoint(int port,
                             const HttpAdServerRequestCb & requestCb);
    HttpAdServerHttpEndpoint(HttpAdServerHttpEndpoint && otherEndpoint);

    ~HttpAdServerHttpEndpoint();

    HttpAdServerHttpEndpoint & operator =
        (const HttpAdServerHttpEndpoint& other);

    int getPort() const;

    /* carbon logging */
    typedef std::function<void (const char * eventName,
                                StatEventType,
                                float,
                                std::initializer_list<int>)> OnEvent;
    OnEvent onEvent;

    void doEvent(const char * eventName, StatEventType type = ET_COUNT,
                 float value = 1.0, const char * units = "",
                 std::initializer_list<int> extra = DefaultOutcomePercentiles)
      const
    {
        if (onEvent) {
            std::string prefixedName(name() + "." + eventName);
            onEvent(prefixedName.c_str(), type, value, extra);
        }
    }

    virtual std::shared_ptr<ConnectionHandler> makeNewHandler();

private:
    int port_;
    HttpAdServerRequestCb requestCb_;
};
        
/****************************************************************************/
/* HTTPADSERVERCONNECTOR                                                    */
/****************************************************************************/

struct HttpAdServerConnector : public AdServerConnector {
    HttpAdServerConnector(const std::string & serviceName,
                          const std::shared_ptr<Datacratic::ServiceProxies>
                          & proxy = std::make_shared<Datacratic::ServiceProxies>());
    ~HttpAdServerConnector() {
        shutdown();
    }

    void registerEndpoint(int port, const HttpAdServerRequestCb & requestCb);

    void init(const std::shared_ptr<ConfigurationService> & config);
    void shutdown();

    void bindTcp();

    void start();

    std::vector<int> ports() const;

private:
    std::vector<HttpAdServerHttpEndpoint> endpoints_;
};

} //namespace RTBKIT
