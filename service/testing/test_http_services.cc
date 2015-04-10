#include "test_http_services.h"

using namespace std;
using namespace Datacratic;


HttpService::
HttpService(const shared_ptr<ServiceProxies> & proxies)
    : ServiceBase("http-test-service", proxies),
      HttpEndpoint("http-test-service-ep"),
      portToUse(0), numReqs(0)
{
}

HttpService::
~HttpService()
{
    shutdown();
}

void
HttpService::
start(const string & address, int numThreads)
{
    init(portToUse, address, numThreads);
    waitListening();
}

shared_ptr<ConnectionHandler>
HttpService::
makeNewHandler()
{
    return make_shared<HttpTestConnHandler>();
}

void
HttpTestConnHandler::
handleHttpPayload(const HttpHeader & header,
                  const string & payload)
{
    HttpService *svc = (HttpService *) httpEndpoint;
    svc->handleHttpPayload(*this, header, payload);
}

void
HttpTestConnHandler::
sendResponse(int code, const string & body, const string & type)
{
    putResponseOnWire(HttpResponse(code, type, body));
}

HttpGetService::
HttpGetService(const shared_ptr<ServiceProxies> & proxies)
    : HttpService(proxies)
{}

void
HttpGetService::
handleHttpPayload(HttpTestConnHandler & handler,
                  const HttpHeader & header,
                  const string & payload)
{
    numReqs++;
    string key = header.verb + ":" + header.resource;
    if (header.resource == "/timeout") {
        sleep(3);
        handler.sendResponse(200, "Will time out", "text/plain");
    }
    else if (header.resource == "/counter") {
        handler.sendResponse(200, to_string(numReqs), "text/plain");
    }
    else if (header.resource == "/headers") {
        string headersBody("{\n");
        bool first(true);
        for (const auto & it: header.headers) {
            if (first) {
                first = false;
            }
            else {
                headersBody += ",\n";
            }
            headersBody += "  \"" + it.first + "\": \"" + it.second + "\"\n";
        }
        headersBody += "}\n";
        handler.sendResponse(200, headersBody, "application/json");
    }
    else if (header.resource == "/query-params") {
        string body = header.queryParams.uriEscaped();
        handler.sendResponse(200, body, "text/plain");
    }
    else if (header.resource == "/connection-close") {
        handler.send("HTTP/1.1 204 No contents\r\nConnection: close\r\n\r\n",
                     PassiveConnectionHandler::NextAction::NEXT_CLOSE);
    }
    else if (header.resource == "/quiet-connection-close") {
        handler.send("HTTP/1.1 204 No contents\r\n\r\n",
                     PassiveConnectionHandler::NextAction::NEXT_CLOSE);
    }
    else if (header.resource == "/abrupt-connection-close") {
        handler.send("",
                     PassiveConnectionHandler::NextAction::NEXT_CLOSE);
    }
    else {
        const auto & it = responses_.find(key);
        if (it == responses_.end()) {
            handler.sendResponse(404, "Not found", "text/plain");
        }
        else {
            const TestResponse & resp = it->second;
            handler.sendResponse(resp.code_, resp.body_, "text/plain");
        }
    }
}

void
HttpGetService::
addResponse(const string & verb, const string & resource,
            int code, const string & body)
{
    string key = verb + ":" + resource;
    responses_[key] = TestResponse(code, body);
}

HttpUploadService::
HttpUploadService(const shared_ptr<ServiceProxies> & proxies)
    : HttpService(proxies)
{}

void
HttpUploadService::
handleHttpPayload(HttpTestConnHandler & handler,
                  const HttpHeader & header,
                  const string & payload)
{
    Json::Value response;

    string cType = header.contentType;
    if (cType.empty()) {
        cType = header.tryGetHeader("Content-Type");
    }
    response["verb"] = header.verb;
    response["type"] = cType;
    response["payload"] = payload;
    Json::Value & jsonHeaders = response["headers"];
    for (const auto & it: header.headers) {
        jsonHeaders[it.first] = it.second;
    }

    handler.sendResponse(200, response.toString(), "application/json");
}
