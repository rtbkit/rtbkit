/* http_named_endpoint.h                                           -*- C++ -*-
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
*/

#pragma once

#include "soa/service/http_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "named_endpoint.h"
#include <boost/make_shared.hpp>


namespace Datacratic {


/*****************************************************************************/
/* HTTP NAMED ENDPOINT                                                       */
/*****************************************************************************/

/** A message loop-compatible endpoint for http connections. */

struct HttpNamedEndpoint : public NamedEndpoint, public HttpEndpoint {

    HttpNamedEndpoint()
        : HttpEndpoint("HttpNamedEndpoint")
    {
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName)
    {
        NamedEndpoint::init(config, endpointName);
    }

    /** Bid into a given address.  Address is host:port.

        If no port is given (and no colon), than use any port number.
        If port is a number and then "+", then scan for any port higher than
        the given number.
        If host is empty or "*", then use all interfaces.
    */
    std::string
    bindTcpAddress(const std::string & address)
    {
        using namespace std;
        auto pos = address.find(':');
        if (pos == string::npos) {
            // No port specification; take any port
            return bindTcp(PortRange(12000, 12999), address);
        }
        string hostPart(address, 0, pos);
        string portPart(address, pos + 1);

        if (portPart.empty())
            throw ML::Exception("invalid port " + portPart + " in address "
                                + address);

        if (portPart[portPart.size() - 1] == '+') {
            int port = boost::lexical_cast<int>(string(portPart, 0, portPart.size() - 1));
            return bindTcp(PortRange(port, port + 999),
                           hostPart);
        }
        else {
            return bindTcp(boost::lexical_cast<int>(portPart),
                           hostPart);
        }
    }

    /** Bind into a specific tcp port.  If the port is not available, it will
        throw an exception.

        Returns the uri to connect to.
    */
    std::string
    bindTcpFixed(std::string host, int port)
    {
        return bindTcp(port, host);
    }

    /** Bind into a tcp port.  If the preferred port is not available, it will
        scan until it finds one that is.

        Returns the uri to connect to.
    */
    std::string
    bindTcp(PortRange const & portRange, std::string host = "")
    {
        using namespace std;

        // TODO: generalize this...
        if (host == "" || host == "*")
            host = "0.0.0.0";

        // TODO: really scan ports
        int port = HttpEndpoint::listen(portRange, host, false /* name lookup */);

        cerr << "bound tcp for http port " << port << endl;

        auto getUri = [&] (const std::string & host)
            {
                return "http://" + host + ":" + to_string(port);
            };

        Json::Value config;

        auto addEntry = [&] (const std::string & addr,
                             const std::string & hostScope,
                             const std::string & uri)
            {
                Json::Value & entry = config[config.size()];
                entry["httpUri"] = uri;

                Json::Value & transports = entry["transports"];
                transports[0]["name"] = "tcp";
                transports[0]["addr"] = addr;
                transports[0]["hostScope"] = hostScope;
                transports[0]["port"] = port;
                transports[1]["name"] = "http";
                transports[1]["uri"] = uri;
            };

        if (host == "0.0.0.0") {
            auto interfaces = getInterfaces({AF_INET});
            for (unsigned i = 0;  i < interfaces.size();  ++i) {
                addEntry(interfaces[i].addr,
                         interfaces[i].hostScope,
                         getUri(interfaces[i].addr));
            }
            publishAddress("tcp", config);
            return getUri(host);
        }
        else {
            string host2 = addrToIp(host);
            string uri = getUri(host2);
            // TODO: compute host scope
            addEntry(host2, "*", uri);
            publishAddress("tcp", config);
            return uri;
        }
    }

    struct RestConnectionHandler: public HttpConnectionHandler {
        RestConnectionHandler(HttpNamedEndpoint * endpoint)
            : endpoint(endpoint)
        {
        }

        HttpNamedEndpoint * endpoint;

        virtual void
        handleHttpPayload(const HttpHeader & header,
                          const std::string & payload)
        {
            try {
                endpoint->onRequest(this, header, payload);
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

        void sendErrorResponse(int code, const std::string & error)
        {
            Json::Value val;
            val["error"] = error;
            sendErrorResponse(code, val);
        }

        void sendErrorResponse(int code, const Json::Value & error)
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

        void sendResponse(int code,
                          const Json::Value & response,
                          const std::string & contentType = "application/json")
        {
            std::string body = response.toStyledString();
            return sendResponse(code, body, contentType);
        }

        void sendResponse(int code,
                          const std::string & body,
                          const std::string & contentType)
        {
            auto onSendFinished = [=] {
                this->transport().associateWhenHandlerFinished
                (std::make_shared<RestConnectionHandler>(endpoint),
                 "sendResponse");
            };

            send(ML::format("HTTP/1.1 %d %s\r\n"
                            "Content-Type: %s\r\n"
                            "Access-Control-Allow-Origin: *\r\n"
                            "Content-Length: %zd\r\n"
                            "Connection: Keep-Alive\r\n"
                            "\r\n"
                            "%s",
                            code,
                            getResponseReasonPhrase(code).c_str(),
                            contentType.c_str(),
                            body.length(),
                            body.c_str()),
                 NEXT_CONTINUE,
                 onSendFinished);
        }

    };

    typedef std::function<void (RestConnectionHandler * connection,
                                const HttpHeader & header,
                                const std::string & payload)> OnRequest;

    OnRequest onRequest;


    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler()
    {
        return std::make_shared<RestConnectionHandler>(this);
    }
};

/*****************************************************************************/
/* HTTP REST PROXY                                                           */
/*****************************************************************************/

struct HttpRestProxy {

    /** The response of a request.  Has a return code and a body. */
    struct Response {
        Response()
            : code_(0)
        {
        }

        int code() const {
            return code_;
        }

        std::string body() const
        {
            if (code_ < 200 || code_ >= 300)
                throw ML::Exception("invalid http code returned");
            return body_;
        }

        std::string getHeader(const std::string & name) const
        {
            auto it = header_.headers.find(name);
            if (it == header_.headers.end())
                throw ML::Exception("required header " + name + " not found");
            return it->second;
        }

        long code_;
        std::string body_;
        HttpHeader header_;
    };

    struct Content {
        Content()
            : data(0), size(0), hasContent(false)
        {
        }

        Content(const std::string & str,
                const std::string & contentType = "")
            : str(str), data(str.c_str()), size(str.size()),
              hasContent(true), contentType(contentType)
        {
        }

        Content(const char * data, uint64_t size,
                const std::string & contentType = "",
                const std::string & contentMd5 = "")
            : data(data), size(size), hasContent(true),
              contentType(contentType), contentMd5(contentMd5)
        {
        }

        std::string str;

        const char * data;
        uint64_t size;
        bool hasContent;

        std::string contentType;
        std::string contentMd5;
    };

    struct Request {

        Request()
        {
        }

        std::string uri;
        std::string verb;
        std::string bucket;

        std::string contentType;
        Content content;

        RestParams headers;
        RestParams queryParams;
    };

    /** Perform a POST request from end to end. */
    Response post(const std::string & resource,
                  const Content & content = Content(),
                  const RestParams & queryParams = RestParams(),
                  const RestParams & headers = RestParams(),
                  int timeout = -1) const;

    std::string serviceUri;
};


/*****************************************************************************/
/* HTTP NAMED REST PROXY                                                     */
/*****************************************************************************/

/** Proxy to connect to a named http-based service. */

struct HttpNamedRestProxy: public HttpRestProxy {

    HttpNamedRestProxy()
    {
    }

    void init(std::shared_ptr<ConfigurationService> config)
    {
        this->config = config;
    }

    bool connectToServiceClass(const std::string & serviceClass,
                               const std::string & endpointName)
    {
        this->serviceClass = serviceClass;
        this->endpointName = endpointName;

        std::vector<std::string> children
            = config->getChildren("serviceClass/" + serviceClass);

        for (auto c : children) {
            std::string key = "serviceClass/" + serviceClass + "/" + c;
            //cerr << "getting " << key << endl;
            Json::Value value = config->getJson(key);
            std::string name = value["serviceName"].asString();
            std::string path = value["servicePath"].asString();

            //cerr << "name = " << name << " path = " << path << endl;
            if (connect(path + "/" + endpointName))
                break;
        }

        return connected;
    }

    bool connect(const std::string & endpointName)
    {
        using namespace std;

        auto onChange = std::bind(&HttpNamedRestProxy::onConfigChange, this,
                                  std::placeholders::_1,
                                  std::placeholders::_2,
                                  std::placeholders::_3);

        connected = false;

        // 2.  Iterate over all of the connection possibilities until we
        //     find one that works.
        auto onConnection = [&] (const std::string & key,
                                 const Json::Value & epConfig) -> bool
            {
                if (connected)
                    return false;
                //cerr << "epConfig for " << key << " is " << epConfig
                //<< endl;
                
                for (auto & entry: epConfig) {

                    //cerr << "entry is " << entry << endl;

                    if (!entry.isMember("httpUri"))
                        return true;

                    string uri = entry["httpUri"].asString();

                    cerr << "uri = " << uri << endl;

                    auto hs = entry["transports"][0]["hostScope"];
                    if (!hs)
                        continue;

                    // TODO: allow localhost connections on localhost
                    string hostScope = hs.asString();
                    if (hs != "*") {
                        utsname name;
                        if (uname(&name))
                            throw ML::Exception(errno, "uname");
                        if (hostScope != name.nodename)
                            continue;  // wrong host scope
                    }

                    serviceUri = uri;

                    cerr << "connected to " << uri << endl;
                    connected = true;

                    // Continue the connection in the onConfigChange function
                    onConfigChange(ConfigurationService::VALUE_CHANGED,
                                   key,
                                   epConfig);
                    return false;
                }

                return false;
            };

        config->forEachEntry(onConnection, endpointName);
        return connected;
    }

    /** Called back when one of our endpoints either changes or disappears. */
    bool onConfigChange(ConfigurationService::ChangeType change,
                        const std::string & key,
                        const Json::Value & newValue)
    {
        using namespace std;

        cerr << "config for " << key << " has changed" << endl;

#if 0
        // 3.  Find an appropriate entry to connect to
        for (auto & entries: newValue) {
            // TODO: connect
            cerr << "got entries " << entries << endl;
        }
#endif

        return true;
    }


private:
    std::shared_ptr<ConfigurationService> config;

    bool connected;
    std::string serviceClass;
    std::string endpointName;
};

} // namespace Datacratic

