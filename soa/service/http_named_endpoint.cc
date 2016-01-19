/* http_named_endpoint.cc
   Jeremy Barnes, 11 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Named endpoint for http connections.
*/

#include "http_named_endpoint.h"

using namespace std;

namespace Datacratic {


/*****************************************************************************/
/* HTTP NAMED ENDPOINT                                                       */
/*****************************************************************************/

HttpNamedEndpoint::
HttpNamedEndpoint()
    : HttpEndpoint("HttpNamedEndpoint")
{
}

void
HttpNamedEndpoint::
allowAllOrigins()
{
    extraHeaders.push_back({ "Access-Control-Allow-Origin", "*" });
}

void
HttpNamedEndpoint::
init(std::shared_ptr<ConfigurationService> config,
          const std::string & endpointName)
{
    NamedEndpoint::init(config, endpointName);
}

std::string
HttpNamedEndpoint::
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
        unsigned port = boost::lexical_cast<unsigned>(string(portPart, 0, portPart.size() - 1));
        if(port < 65536) {
            unsigned last = port + 999;
            return bindTcp(PortRange(port, last), hostPart);
        }

        throw ML::Exception("invalid port %u", port);
    }

    return bindTcp(boost::lexical_cast<int>(portPart), hostPart);
}

std::string
HttpNamedEndpoint::
bindTcpFixed(std::string host, int port)
{
    return bindTcp(port, host);
}

std::string
HttpNamedEndpoint::
bindTcp(PortRange const & portRange, std::string host)
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

std::shared_ptr<ConnectionHandler>
HttpNamedEndpoint::
makeNewHandler()
{
    auto res = std::make_shared<RestConnectionHandler>(this);

    // Allow it to get a shared pointer to itself
    res->sharedThis = res;
    return res;
}


/*****************************************************************************/
/* HTTP NAMED ENDPOINT REST CONNECTIO HANDLER                                */
/*****************************************************************************/

HttpNamedEndpoint::RestConnectionHandler::
RestConnectionHandler(HttpNamedEndpoint * endpoint)
    : endpoint(endpoint), isZombie(false)
{
}

void
HttpNamedEndpoint::RestConnectionHandler::
handleHttpPayload(const HttpHeader & header,
                  const std::string & payload)
{
    // We don't lock here, since sending the response will take the lock,
    // and whatever called us must know it's a valid connection

    try {
        auto th = sharedThis.lock();
        ExcAssert(th);
        endpoint->onRequest(th, header, payload);
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

void
HttpNamedEndpoint::RestConnectionHandler::
handleDisconnect()
{
    std::unique_lock<std::mutex> guard(mutex);
    //cerr << "*** Got handle disconnect for rest connection handler" << endl;
    isZombie = true;
    HttpConnectionHandler::handleDisconnect();
}

void
HttpNamedEndpoint::RestConnectionHandler::
sendErrorResponse(int code, const std::string & error)
{
    if (isZombie)
        return;

    Json::Value val;
    val["error"] = error;

    sendErrorResponse(code, val);
}

void
HttpNamedEndpoint::RestConnectionHandler::
sendErrorResponse(int code, const Json::Value & error)
{
    if (isZombie)
        return;

    std::string encodedError = error.toString();

    std::unique_lock<std::mutex> guard(mutex);
    if (isZombie)
        return;
    putResponseOnWire(HttpResponse(code, "application/json",
                                   encodedError, endpoint->extraHeaders),
                      nullptr, NEXT_CLOSE);
}

void
HttpNamedEndpoint::RestConnectionHandler::
sendResponse(int code,
             const Json::Value & response,
             const std::string & contentType,
             RestParams headers)
{
    std::string body = response.toStyledString();
    return sendResponse(code, body, contentType, std::move(headers));
}

        

void
HttpNamedEndpoint::RestConnectionHandler::
sendResponse(int code,
             const std::string & body,
             const std::string & contentType,
             RestParams headers)
{
    // Recycle back to a new handler once done so that the next connection can be
    // handled.
    auto onSendFinished = [=] {
        this->transport().associateWhenHandlerFinished
        (endpoint->makeNewHandler(), "sendResponse");
    };
    
    for (auto & h: endpoint->extraHeaders)
        headers.push_back(h);

    std::unique_lock<std::mutex> guard(mutex);
    if (isZombie)
        return;
    putResponseOnWire(HttpResponse(code, contentType, body, headers),
                      onSendFinished);
}

void
HttpNamedEndpoint::RestConnectionHandler::
sendResponseHeader(int code,
                   const std::string & contentType,
                   RestParams headers)
{
    auto onSendFinished = [=] {
        // Do nothing once we've finished sending the response, so that
        // the connection isn't closed
    };
    
    for (auto & h: endpoint->extraHeaders)
        headers.push_back(h);

    std::unique_lock<std::mutex> guard(mutex);
    if (isZombie)
        return;
    putResponseOnWire(HttpResponse(code, contentType, headers),
                      onSendFinished);
}

void
HttpNamedEndpoint::RestConnectionHandler::
sendHttpChunk(const std::string & chunk,
              NextAction next,
              OnWriteFinished onWriteFinished)
{
    std::unique_lock<std::mutex> guard(mutex);
    if (isZombie)
        return;
    HttpConnectionHandler::sendHttpChunk(chunk, next, onWriteFinished);
}


void
HttpNamedEndpoint::RestConnectionHandler::
sendHttpPayload(const std::string & str)
{
    std::unique_lock<std::mutex> guard(mutex);
    if (isZombie)
        return;
    send(str);
}


/*****************************************************************************/
/* HTTP NAMED REST PROXY                                                     */
/*****************************************************************************/

void
HttpNamedRestProxy::
init(std::shared_ptr<ConfigurationService> config)
{
    this->config = config;
}

bool
HttpNamedRestProxy::
connectToServiceClass(const std::string & serviceClass,
                      const std::string & endpointName,
                      bool local)
{
    this->serviceClass = serviceClass;
    this->endpointName = endpointName;

    std::vector<std::string> children
        = config->getChildren("serviceClass/" + serviceClass);

    for (auto c : children) {
        std::string key = "serviceClass/" + serviceClass + "/" + c;

        Json::Value value = config->getJson(key);
        std::string name = value["serviceName"].asString();
        std::string path = value["servicePath"].asString();

        std::string location = value["serviceLocation"].asString();
        if (local && location != config->currentLocation)
            continue;

        //cerr << "name = " << name << " path = " << path << endl;
        if (connect(path + "/" + endpointName))
            break;
    }

    return connected;
}

bool
HttpNamedRestProxy::
connect(const std::string & endpointName)
{
    using namespace std;

    // auto onChange = std::bind(&HttpNamedRestProxy::onConfigChange, this,
    //                           std::placeholders::_1,
    //                           std::placeholders::_2,
    //                           std::placeholders::_3);

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
bool
HttpNamedRestProxy::
onConfigChange(ConfigurationService::ChangeType change,
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


} // namespace Datacratic
