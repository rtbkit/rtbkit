#include "test_http_services.h"

using namespace std;
using namespace Datacratic;


HttpService::
HttpService(const shared_ptr<ServiceProxies> & proxies)
    : ServiceBase("http-test-service", proxies),
      HttpEndpoint("http-test-service-ep"),
      portToUse(0)
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
    : HttpService(proxies), numReqs(0)
{}

void
HttpGetService::
handleHttpPayload(HttpTestConnHandler & handler,
                  const HttpHeader & header,
                  const string & payload)
{
    numReqs++;
    // int localRq = numReqs;
    // if ((localRq % 100) == 0) {
    //     ::fprintf(stderr, "srv reqs: %d\n", localRq);
    // }
    string key = header.verb + ":" + header.resource;
    if (header.resource == "/timeout") {
        sleep(3);
        handler.sendResponse(200, "Will time out", "text/plain");
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
        
    handler.sendResponse(200, response.toString(), "application/json");
}
