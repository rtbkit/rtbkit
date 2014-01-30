#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <string>
#include <boost/test/unit_test.hpp>

#include "jml/utils/testing/watchdog.h"
#include "soa/service/http_endpoint.h"
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/http_client.h"

using namespace std;
using namespace Datacratic;


struct HttpTestConnHandler;

struct HttpService : public ServiceBase, public HttpEndpoint {
    HttpService(const shared_ptr<ServiceProxies> & proxies)
        : ServiceBase("http-test-service", proxies),
          HttpEndpoint("http-test-service-ep")
    {
    }

    ~HttpService()
    {
        shutdown();
    }

    void start()
    {
        init(0, "127.0.0.1", 1);
    }

    virtual shared_ptr<ConnectionHandler> makeNewHandler();
    virtual void handleHttpPayload(HttpTestConnHandler & handler,
                                   const HttpHeader & header,
                                   const string & payload) = 0;
};

struct HttpTestConnHandler : HttpConnectionHandler {
    virtual void handleHttpPayload(const HttpHeader & header,
                                   const string & payload) {
        HttpService *svc = (HttpService *) httpEndpoint;
        svc->handleHttpPayload(*this, header, payload);
    }

    void sendResponse(int code, const string & body, const string & type)
    {
        putResponseOnWire(HttpResponse(code, type, body));
    }
};

shared_ptr<ConnectionHandler>
HttpService::
makeNewHandler()
{
    return make_shared<HttpTestConnHandler>();
}

struct HttpGetService : public HttpService {
    HttpGetService(const shared_ptr<ServiceProxies> & proxies)
        : HttpService(proxies)
    {}

    struct TestResponse {
        TestResponse(int code = 0, const string & body = "")
            : code_(code), body_(body)
        {}

        int code_;
        string body_;
    };

    void handleHttpPayload(HttpTestConnHandler & handler,
                           const HttpHeader & header,
                           const string & payload)
    {
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

    void addResponse(const string & verb, const string & resource,
                     int code, const string & body)
    {
        string key = verb + ":" + resource;
        responses_[key] = TestResponse(code, body);
    }

    map<string, TestResponse> responses_;
};

struct HttpUploadService : public HttpService {
    HttpUploadService(const shared_ptr<ServiceProxies> & proxies)
        : HttpService(proxies)
    {}

    void handleHttpPayload(HttpTestConnHandler & handler,
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
};

/* sync request helpers */
HttpClientResponse
doGetRequest(MessageLoop & loop,
             const string & baseUrl, const string & resource,
             const RestParams & headers = RestParams(),
             int timeout = -1)
{
    HttpClientResponse response;

    int done(false);
    auto onResponse = [&] (const HttpClientResponse & resp,
                           const HttpRequest & req) {
        response = resp;
        done = true;
        ML::futex_wake(done);
    };

    auto client = make_shared<HttpClient>(baseUrl);
    loop.addSource("httpClient", client);
    if (timeout == -1) {
        client->get(resource, onResponse, RestParams(), headers);
    }
    else {
        client->get(resource, onResponse, RestParams(), headers,
                    timeout);
    }

    while (!done) {
        int oldDone = done;
        ML::futex_wait(done, oldDone);
    }

    loop.removeSource(client.get());
    client->waitConnectionState(AsyncEventSource::DISCONNECTED);

    return response;
}

HttpClientResponse
doUploadRequest(MessageLoop & loop,
                bool isPut,
                const string & baseUrl, const string & resource,
                const string & body, const string & type)
{
    HttpClientResponse response;

    int done(false);
    auto onResponse = [&] (const HttpClientResponse & resp,
                           const HttpRequest & req) {
        response = resp;
        done = true;
        ML::futex_wake(done);
    };

    auto client = make_shared<HttpClient>(baseUrl);
    loop.addSource("httpClient", client);

    HttpRequest::Content content(body, type);
    if (isPut) {
        client->put(resource, onResponse, content);
    }
    else {
        client->post(resource, onResponse, content);
    }

    while (!done) {
        int oldDone = done;
        ML::futex_wait(done, oldDone);
    }
    
    loop.removeSource(client.get());
    client->waitConnectionState(AsyncEventSource::DISCONNECTED);

    return response;
}

BOOST_AUTO_TEST_CASE( test_http_client_get )
{
    ML::Watchdog watchdog(10);
    auto proxies = make_shared<ServiceProxies>();
    HttpGetService service(proxies);

    service.addResponse("GET", "/coucou", 200, "coucou");
    service.start();

    MessageLoop loop;
    loop.start();

    /* request to bad ip */
    {
        string baseUrl("http://123.234.12.23");
        auto resp = doGetRequest(loop, baseUrl, "/");
        BOOST_CHECK_EQUAL(resp.errorCode_,
                          HttpClientResponse::Error::COULD_NOT_CONNECT);
        BOOST_CHECK_EQUAL(resp.header_.code(), 0);
    }

    /* request to bad hostname */
    {
        string baseUrl("http://somewhere.lost");
        auto resp = doGetRequest(loop, baseUrl, "/");
        BOOST_CHECK_EQUAL(resp.errorCode_,
                          HttpClientResponse::Error::HOST_NOT_FOUND);
        BOOST_CHECK_EQUAL(resp.header_.code(), 0);
    }

    /* request with timeout */
    {
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/timeout", {}, 1);
        BOOST_CHECK_EQUAL(resp.errorCode_,
                          HttpClientResponse::Error::TIMEOUT);
        BOOST_CHECK_EQUAL(resp.header_.code(), 0);
    }

    /* request to /nothing -> 404 */
    {
        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/nothing");
        BOOST_CHECK_EQUAL(resp.errorCode_, HttpClientResponse::Error::NONE);
        BOOST_CHECK_EQUAL(resp.header_.code(), 404);
    }

    /* request to /coucou -> 200 + "coucou" */
    {
        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/coucou");
        BOOST_CHECK_EQUAL(resp.errorCode_, HttpClientResponse::Error::NONE);
        BOOST_CHECK_EQUAL(resp.header_.code(), 200);
        BOOST_CHECK_EQUAL(resp.body(), "coucou");
    }

    /* headers and cookies */
    {
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/headers",
                                 {{"someheader", "somevalue"}});
        Json::Value expBody;
        expBody["accept"] = "*/*";
        expBody["host"] = baseUrl.substr(7);
        expBody["someheader"] = "somevalue";
        BOOST_CHECK_EQUAL(resp.jsonBody(), expBody);
    }
}

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_post )
{
    ML::Watchdog watchdog(10);
    auto proxies = make_shared<ServiceProxies>();
    HttpUploadService service(proxies);
    service.start();

    MessageLoop loop;
    loop.start();

    /* request to /coucou -> 200 + "coucou" */
    {
        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));
        auto resp = doUploadRequest(loop, false, baseUrl, "/post-test",
                                    "post body", "application/x-nothing");
        BOOST_CHECK_EQUAL(resp.errorCode_, HttpClientResponse::Error::NONE);
        BOOST_CHECK_EQUAL(resp.header_.code(), 200);
        Json::Value jsonBody = Json::parse(resp.body());
        BOOST_CHECK_EQUAL(jsonBody["verb"], "POST");
        BOOST_CHECK_EQUAL(jsonBody["payload"], "post body");
        BOOST_CHECK_EQUAL(jsonBody["type"], "application/x-nothing");
    }
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_put )
{
    ML::Watchdog watchdog(10);
    auto proxies = make_shared<ServiceProxies>();
    HttpUploadService service(proxies);
    service.start();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));
    string bigBody;
    for (int i = 0; i < 65535; i++) {
        bigBody += "this is one big body,";
    }
    auto resp = doUploadRequest(loop, true, baseUrl, "/put-test",
                                bigBody, "application/x-nothing");
    BOOST_CHECK_EQUAL(resp.errorCode_, HttpClientResponse::Error::NONE);
    BOOST_CHECK_EQUAL(resp.header_.code(), 200);
    Json::Value jsonBody = Json::parse(resp.body());
    BOOST_CHECK_EQUAL(jsonBody["verb"], "PUT");
    BOOST_CHECK_EQUAL(jsonBody["payload"], bigBody);
    BOOST_CHECK_EQUAL(jsonBody["type"], "application/x-nothing");
}
#endif

#if 1
/* Ensures that all requests are correctly performed under load.
   Not a performance test. */
BOOST_AUTO_TEST_CASE( test_http_client_stress_test )
{
    ML::Watchdog watchdog(10);
    auto proxies = make_shared<ServiceProxies>();
    HttpGetService service(proxies);

    service.addResponse("GET", "/", 200, "coucou");
    service.start();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));

    auto client = make_shared<HttpClient>(baseUrl);
    loop.addSource("httpClient", client);

    int maxReqs(10000), numReqs(0);
    int numResponses(0);

    auto onResponse = [&] (const HttpClientResponse & resp,
                           const HttpRequest & req) {
        // cerr << ("* onResponse " + to_string(numResponses)
        //          + ": " + to_string(resp.header_.code())
        //          + "\n\n\n");
        // cerr << "    body =\n/" + resp.body() + "/\n";
        numResponses++;
        if (numResponses == numReqs) {
            ML::futex_wake(numResponses);
        }
    };

    while (numReqs < maxReqs) {
        if (client->get("/", onResponse)) {
            numReqs++;
        }
    }

    while (numResponses < numReqs) {
        int old(numResponses);
        ML::futex_wait(numResponses, old);
    }
}
#endif
