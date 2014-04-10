#include <atomic>
#include <memory>
#include <string>
#include <thread>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "jml/arch/exception.h"
#include "soa/types/date.h"
#include "soa/service/http_client.h"
#include "soa/service/http_endpoint.h"
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/rest_service_endpoint.h"

using namespace std;
using namespace Datacratic;


/* http service */

struct HttpTestConnHandler;

struct HttpService : public ServiceBase, public HttpEndpoint {
    HttpService(const shared_ptr<ServiceProxies> & proxies)
        : ServiceBase("http-test-service", proxies),
          HttpEndpoint("http-test-service-ep"),
          portToUse(0)
    {
    }

    ~HttpService()
    {
        shutdown();
    }

    void start()
    {
        init(portToUse, "127.0.0.1", 1);
    }

    virtual shared_ptr<ConnectionHandler> makeNewHandler();
    virtual void handleHttpPayload(HttpTestConnHandler & handler,
                                   const HttpHeader & header,
                                   const string & payload) = 0;

    int portToUse;
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


/* bench methods */

void
AsyncModelBench(const string & baseUrl, int maxReqs, int concurrency)
{
    int numReqs, numResponses(0);
    MessageLoop loop;

    loop.start();

    auto client = make_shared<HttpClient>(baseUrl, concurrency);
    loop.addSource("httpClient", client);

    auto onDone = [&] (const HttpRequest & rq, HttpClientError errorCode_) {
        numResponses++;
        if (numResponses == numReqs) {
            ML::futex_wake(numResponses);
        }
    };
    auto cbs = make_shared<HttpClientCallbacks>(nullptr, nullptr, nullptr, onDone);

    auto & clientRef = *client.get();
    for (numReqs = 0; numReqs < maxReqs;) {
        if (clientRef.get("/", cbs)) {
            numReqs++;
        }
    }

    while (numResponses < numReqs) {
        int old(numResponses);
        ML::futex_wait(numResponses, old);
    }
}

void
ThreadedModelBench(const string & baseUrl, int maxReqs, int concurrency)
{
    vector<thread> threads;

    auto threadFn = [&] (int num, int nReqs) {
        int i;
        HttpRestProxy client(baseUrl);
        for (i = 0; i < nReqs; i++) {
            auto response = client.get("/");
        }
    };

    int slice(maxReqs / concurrency);
    for (int i = 0; i < concurrency; i++) {
        threads.emplace_back(threadFn, i, slice);
    }
    for (int i = 0; i < concurrency; i++) {
        threads[i].join();
    }
}

int main(int argc, char *argv[])
{
    using namespace boost::program_options;

    int concurrency(-1);
    int model(0);
    int maxReqs(-1);
    size_t payloadSize(0);

    options_description all_opt;
    all_opt.add_options()
        ("concurrency,C", value(&concurrency),
         "Number of concurrent requests")
        ("model,m", value(&model),
         "Type of concurrency model (1 for async, 2 for threaded))")
        ("requests,r", value(&maxReqs),
         "total of number of requests to perform")
        ("payload-size,s", value(&payloadSize),
         "size of the response body (*8)")
        ("help,H", "show help");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        return 1;
    }

    if (maxReqs == -1) {
        throw ML::Exception("'max-reqs' must be specified");
    }
    if (concurrency == -1) {
        throw ML::Exception("'concurrency' must be specified");
    }
    if (payloadSize == 0) {
        throw ML::Exception("'payload-size' must be specified");
    }

    /* service setup */
    auto proxies = make_shared<ServiceProxies>();
    HttpGetService service(proxies);
    service.portToUse = 20000;

    string payload;
    while (payload.size() < payloadSize) {
        payload += "aaaaaaaa";
    }
    
    service.addResponse("GET", "/", 200, payload);
    service.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));

    Date start = Date::now();
    if (model == 1) {
        AsyncModelBench(baseUrl, maxReqs,concurrency);
    }
    else if (model == 2) {
        ThreadedModelBench(baseUrl, maxReqs,concurrency);
    }
    else {
        throw ML::Exception("invalid 'model'");
    }
    Date end = Date::now();
    double delta = end - start;
    double qps = maxReqs / delta;
    ::fprintf(stderr, "%d requests performed in %f secs => %f qps\n",
              maxReqs, delta, qps);

    return 0;
}
