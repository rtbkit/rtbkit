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
#include "soa/types/value_description.h"
#include "soa/service/http_client.h"
#include "soa/service/http_endpoint.h"
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/runner.h"

#include "test_http_services.h"

using namespace std;
using namespace Datacratic;


/* bench methods */

void
AsyncModelBench(const string & baseUrl, int maxReqs, int concurrency,
                Date & start, Date & end)
{
    int numReqs, numResponses(0), numMissed(0);
    MessageLoop loop(1, 0, -1);

    loop.start();

    auto client = make_shared<HttpClient>(baseUrl, concurrency);
    loop.addSource("httpClient", client);

    auto onResponse = [&] (const HttpRequest & rq, HttpClientError errorCode_,
                           int status, string && headers, string && body) {
        numResponses++;
        // if (numResponses % 1000) {
            // cerr << "resps: "  + to_string(numResponses) + "\n";
        // }
        if (numResponses == maxReqs) {
            // cerr << "received all responses\n";
            ML::futex_wake(numResponses);
        }
    };
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);

    auto & clientRef = *client.get();
    string url("/");
    start = Date::now();
    for (numReqs = 0; numReqs < maxReqs;) {
        if (clientRef.get(url, cbs)) {
            numReqs++;
            // if (numReqs % 1000) {
            //     cerr << "reqs: "  + to_string(numReqs) + "\n";
            // }
        }
        else {
            numMissed++;
        }
    }

    while (numResponses < maxReqs) {
        // cerr << (" num Responses: " + to_string(numResponses)
        //          + "; max reqs: " + to_string(maxReqs)
        //          + "\n");
        int old(numResponses);
        ML::futex_wait(numResponses, old);
    }
    end = Date::now();

    loop.removeSource(client.get());
    client->waitConnectionState(AsyncEventSource::DISCONNECTED);

    cerr << "num misses: "  + to_string(numMissed) + "\n";
}

void
ThreadedModelBench(const string & baseUrl, int maxReqs, int concurrency,
                   Date & start, Date & end)
{
    vector<thread> threads;

    auto threadFn = [&] (int num, int nReqs) {
        int i;
        HttpRestProxy client(baseUrl);
        for (i = 0; i < nReqs; i++) {
            auto response = client.get("/");
        }
    };

    start = Date::now();
    int slice(maxReqs / concurrency);
    for (int i = 0; i < concurrency; i++) {
        // cerr << "doing slice: "  + to_string(slice) + "\n";
        threads.emplace_back(threadFn, i, slice);
    }
    for (int i = 0; i < concurrency; i++) {
        threads[i].join();
    }
    end = Date::now();
}

int main(int argc, char *argv[])
{
    using namespace boost::program_options;

    size_t concurrency(0);
    int model(0);
    size_t maxReqs(0);
    size_t payloadSize(0);

    string serveriface("127.0.0.1");
    string clientiface(serveriface);

    options_description all_opt;
    all_opt.add_options()
        ("client-iface,C", value(&clientiface),
         "address:port to connect to (\"none\" for no client)")
        ("concurrency,c", value(&concurrency),
         "Number of concurrent requests")
        ("model,m", value(&model),
         "Type of concurrency model (1 for async, 2 for threaded))")
        ("requests,r", value(&maxReqs),
         "total of number of requests to perform")
        ("payload-size,s", value(&payloadSize),
         "size of the response body")
        ("server-iface,S", value(&serveriface),
         "server address (\"none\" for no server)")
        ("help,H", "show help");

    if (argc == 1) {
        return 0;
    }

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

    /* service setup */
    auto proxies = make_shared<ServiceProxies>();

    HttpGetService service(proxies);

    if (concurrency == 0) {
        throw ML::Exception("'concurrency' must be specified");
    }

    if (payloadSize == 0) {
        throw ML::Exception("'payload-size' must be specified");
    }

    string payload;
    while (payload.size() < payloadSize) {
        payload += "aaaaaaaa";
    }

    if (serveriface != "none") {
        cerr << "launching server\n";
        service.portToUse = 20000;

        service.addResponse("GET", "/", 200, payload);
        service.start(serveriface, concurrency);
    }

    if (clientiface != "none") {
        cerr << "launching client\n";
        if (maxReqs == 0) {
            throw ML::Exception("'max-reqs' must be specified");
        }

        string baseUrl;
        if (serveriface != "none") {
            baseUrl = ("http://" + serveriface
                       + ":" + to_string(service.port()));
        }
        else {
            baseUrl = "http://" + clientiface;
        }

        ::printf("model\tconc.\treqs\tsize\ttime_secs\tqps\n");

        Date start, end;
        if (model == 1) {
            AsyncModelBench(baseUrl, maxReqs, concurrency, start, end);
        }
        else if (model == 2) {
            ThreadedModelBench(baseUrl, maxReqs, concurrency, start, end);
        }
        else {
            throw ML::Exception("invalid 'model'");
        }
        double delta = end - start;
        double qps = maxReqs / delta;
        ::printf("%d\t%lu\t%lu\t%lu\t%f\t%f\n",
                 model, concurrency, maxReqs, payloadSize, delta, qps);
    }
    else {
        while (1) {
            sleep(100);
        }
    }

    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    Json::Value jsonUsage = jsonEncode(usage);
    cerr << "rusage:\n" << jsonUsage.toStyledString() << endl;

    return 0;
}
