#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <string>

#include "soa/service/http_endpoint.h"
#include "soa/service/named_endpoint.h"
#include "soa/service/service_base.h"


namespace Datacratic {

struct HttpTestConnHandler;

struct HttpService : public ServiceBase, public HttpEndpoint
{
    HttpService(const std::shared_ptr<ServiceProxies> & proxies);

    ~HttpService();

    void start(const std::string & address = "127.0.0.1", int numThreads = 1);

    virtual std::shared_ptr<ConnectionHandler> makeNewHandler();
    virtual void handleHttpPayload(HttpTestConnHandler & handler,
                                   const HttpHeader & header,
                                   const std::string & payload) = 0;

    int portToUse;
    std::atomic<size_t> numReqs;
};

struct HttpTestConnHandler : HttpConnectionHandler
{
    virtual void handleHttpPayload(const HttpHeader & header,
                                   const std::string & payload);
    void sendResponse(int code, const std::string & body, const std::string & type);
};

struct HttpGetService : public HttpService
{
    HttpGetService(const std::shared_ptr<ServiceProxies> & proxies);

    struct TestResponse {
        TestResponse(int code = 0, const std::string & body = "")
            : code_(code), body_(body)
        {}

        int code_;
        std::string body_;
    };

    void handleHttpPayload(HttpTestConnHandler & handler,
                           const HttpHeader & header,
                           const std::string & payload);
    void addResponse(const std::string & verb, const std::string & resource,
                     int code, const std::string & body);

    std::map<std::string, TestResponse> responses_;
};

struct HttpUploadService : public HttpService
{
    HttpUploadService(const std::shared_ptr<ServiceProxies> & proxies);

    void handleHttpPayload(HttpTestConnHandler & handler,
                           const HttpHeader & header,
                           const std::string & payload);
};

} // namespace Datacratic
