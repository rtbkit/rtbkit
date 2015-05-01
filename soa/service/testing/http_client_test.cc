#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <string>
#include <tuple>
#include <boost/test/unit_test.hpp>

#include "jml/arch/futex.h"
#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/http_client.h"
#include "soa/utils/print_utils.h"

#include "test_http_services.h"


using namespace std;
using namespace Datacratic;


/* helpers functions used in tests */
namespace {

typedef tuple<HttpClientError, int, string> ClientResponse;

#define CALL_MEMBER_FN(object, pointer)  (object.*(pointer))

/* sync request helpers */
template<typename Func>
ClientResponse
doRequest(MessageLoop & loop,
          const string & baseUrl, const string & resource,
          Func func,
          const RestParams & queryParams, const RestParams & headers,
          int timeout = -1)
{
    ClientResponse response;

    auto client = make_shared<HttpClient>(baseUrl, 4);
    loop.addSource("client", client);
    client->waitConnectionState(AsyncEventSource::CONNECTED);

    int done(false);
    auto onResponse = [&] (const HttpRequest & rq,
                           HttpClientError error,
                           int status,
                           string && headers,
                           string && body) {
        int & code = get<1>(response);
        code = status;
        string & body_ = get<2>(response);
        body_ = move(body);
        HttpClientError & errorCode = get<0>(response);
        errorCode = error;
        done = true;
        ML::futex_wake(done);
    };
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);

    CALL_MEMBER_FN(*client, func)(resource, cbs, queryParams, headers,
                                  timeout);

    while (!done) {
        int oldDone = done;
        ML::futex_wait(done, oldDone);
    }

    loop.removeSource(client.get());
    client->waitConnectionState(AsyncEventSource::DISCONNECTED);

    return response;
}

ClientResponse
doGetRequest(MessageLoop & loop,
             const string & baseUrl, const string & resource,
             const RestParams & queryParams = RestParams(),
             const RestParams & headers = RestParams(),
             int timeout = -1)
{
    return doRequest(loop, baseUrl, resource, &HttpClient::get,
                     queryParams, headers, timeout);
}

ClientResponse
doDeleteRequest(MessageLoop & loop,
                const string & baseUrl, const string & resource,
                const RestParams & queryParams = RestParams(),
                const RestParams & headers = RestParams(),
                int timeout = -1)
{
    return doRequest(loop, baseUrl, resource, &HttpClient::del,
                     queryParams, headers, timeout);
}

ClientResponse
doUploadRequest(MessageLoop & loop,
                bool isPut,
                const string & baseUrl, const string & resource,
                const string & body, const string & type)
{
    ClientResponse response;

    auto client = make_shared<HttpClient>(baseUrl, 4);
    loop.addSource("client", client);
    client->waitConnectionState(AsyncEventSource::CONNECTED);

    int done(false);
    auto onResponse = [&] (const HttpRequest & rq,
                           HttpClientError error,
                           int status,
                           string && headers,
                           string && body) {
        int & code = get<1>(response);
        code = status;
        string & body_ = get<2>(response);
        body_ = move(body);
        HttpClientError & errorCode = get<0>(response);
        errorCode = error;
        done = true;
        ML::futex_wake(done);
    };

    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    HttpRequest::Content content(body, type);
    if (isPut) {
        client->put(resource, cbs, content);
    }
    else {
        client->post(resource, cbs, content);
    }

    while (!done) {
        int oldDone = done;
        ML::futex_wait(done, oldDone);
    }

    loop.removeSource(client.get());
    client->waitConnectionState(AsyncEventSource::DISCONNECTED);

    return response;
}

}

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_get )
{
    cerr << "client_get\n";
    ML::Watchdog watchdog(10);
    auto proxies = make_shared<ServiceProxies>();
    HttpGetService service(proxies);

    service.addResponse("GET", "/coucou", 200, "coucou");
    service.start();

    MessageLoop loop;
    loop.start();

    service.waitListening();

#if 0
    /* request to bad ip
       Note: if the ip resolution timeout is very high on the router, the
       Watchdog timeout might trigger first */
    {
        ::fprintf(stderr, "request to bad ip\n");
        string baseUrl("http://123.234.12.23");
        auto resp = doGetRequest(loop, baseUrl, "/");
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::CouldNotConnect);
        BOOST_CHECK_EQUAL(get<1>(resp), 0);
    }
#endif

#if 0
    /* request to bad hostname
       Note: will fail when the name service returns a "default" value for all
       non resolved hosts */
    {
        ::fprintf(stderr, "request to bad hostname\n");
        string baseUrl("http://somewhere.lost");
        auto resp = doGetRequest(loop, baseUrl, "/");
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::HostNotFound);
        BOOST_CHECK_EQUAL(get<1>(resp), 0);
    }
#endif

    /* request with timeout */
    {
        ::fprintf(stderr, "request with timeout\n");
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/timeout", {}, {}, 1);
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::Timeout);
        BOOST_CHECK_EQUAL(get<1>(resp), 0);
    }

    /* request connection close */
    {
        ::fprintf(stderr, "testing behaviour with connection: close\n");
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/connection-close");
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
        BOOST_CHECK_EQUAL(get<1>(resp), 204);
    }

    /* request to /nothing -> 404 */
    {
        ::fprintf(stderr, "request with 404\n");
        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/nothing");
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
        BOOST_CHECK_EQUAL(get<1>(resp), 404);
    }

    /* request to /coucou -> 200 + "coucou" */
    {
        ::fprintf(stderr, "request with 200\n");
        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/coucou");
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
        BOOST_CHECK_EQUAL(get<1>(resp), 200);
        BOOST_CHECK_EQUAL(get<2>(resp), "coucou");
    }

    /* headers and cookies */
    {
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/headers", {},
                                 {{"someheader", "somevalue"}});
        Json::Value expBody;
        expBody["accept"] = "*/*";
        expBody["host"] = baseUrl.substr(7);
        expBody["someheader"] = "somevalue";
        Json::Value jsonBody = Json::parse(get<2>(resp));
        BOOST_CHECK_EQUAL(jsonBody, expBody);
    }

    /* query-params */
    {
        string baseUrl("http://127.0.0.1:" + to_string(service.port()));
        auto resp = doGetRequest(loop, baseUrl, "/query-params",
                                 {{"value", "hello"}});
        string body = get<2>(resp);
        BOOST_CHECK_EQUAL(body, "?value=hello");
    }

    service.shutdown();
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_post )
{
    cerr << "client_post\n";
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
        BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
        BOOST_CHECK_EQUAL(get<1>(resp), 200);
        Json::Value jsonBody = Json::parse(get<2>(resp));
        BOOST_CHECK_EQUAL(jsonBody["verb"], "POST");
        BOOST_CHECK_EQUAL(jsonBody["payload"], "post body");
        BOOST_CHECK_EQUAL(jsonBody["type"], "application/x-nothing");
    }

    service.shutdown();
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_put )
{
    cerr << "client_put\n";
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
    BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
    BOOST_CHECK_EQUAL(get<1>(resp), 200);
    Json::Value jsonBody = Json::parse(get<2>(resp));
    BOOST_CHECK_EQUAL(jsonBody["verb"], "PUT");
    BOOST_CHECK_EQUAL(jsonBody["payload"], bigBody);
    BOOST_CHECK_EQUAL(jsonBody["type"], "application/x-nothing");

    service.shutdown();
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( http_test_client_delete )
{
    cerr << "client_delete" << endl;
    ML::Watchdog watchdog(10);

    auto proxies = make_shared<ServiceProxies>();
    HttpGetService service(proxies);

    service.addResponse("DELETE", "/deleteMe", 200, "Deleted");
    service.start();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:" + to_string(service.port()));
    auto resp = doDeleteRequest(loop, baseUrl, "/deleteMe", {}, {}, 1);

    BOOST_CHECK_EQUAL(get<0>(resp), HttpClientError::None);
    BOOST_CHECK_EQUAL(get<1>(resp), 200);

    service.shutdown();
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_http_client_put_multi )
{
    cerr << "client_put_multi\n";
    auto proxies = make_shared<ServiceProxies>();
    HttpUploadService service(proxies);
    service.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));

    MessageLoop loop;
    loop.start();

    auto client = make_shared<HttpClient>(baseUrl);
    loop.addSource("client", client);
    client->waitConnectionState(AsyncEventSource::CONNECTED);

    size_t maxRequests(500);
    int done(0);

    auto makeBody = [&] (size_t i) {
        int multiplier = (i < maxRequests / 2) ? -2 : 2;
        size_t bodySize = 2000 + multiplier * i;
        string body = ML::format("%.4x", bodySize);
        size_t rndSize = bodySize - body.size();
        body += randomString(rndSize);

        return body;
    };

    for (size_t i = 0; i < maxRequests; i++) {
        auto sendBody = makeBody(i);
        auto onResponse = [&, sendBody] (const HttpRequest & rq,
                                         HttpClientError error,
                                         int status,
                                         string && headers,
                                         string && body) {
            BOOST_CHECK_EQUAL(error, HttpClientError::None);
            BOOST_CHECK_EQUAL(status, 200);
            Json::Value jsonBody = Json::parse(body);
            BOOST_CHECK_EQUAL(jsonBody["verb"], "PUT");
            BOOST_CHECK_EQUAL(jsonBody["payload"], sendBody);
            BOOST_CHECK_EQUAL(jsonBody["type"], "text/plain");
            done++;
            if (done == maxRequests) {
                ML::futex_wake(done);
            }
        };

        auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
        HttpRequest::Content content(sendBody, "text/plain");
        while (!client->put("/", cbs, content)) {
            ML::sleep(0.2);
        }
    };

    while (done < maxRequests) {
        int oldDone = done;
        ML::futex_wait(done, oldDone);
    }

    loop.shutdown();
    service.shutdown();
}
#endif

#if 1
/* Ensures that all requests are correctly performed under load, including
   when "Connection: close" is encountered once in a while.
   Not a performance test. */
BOOST_AUTO_TEST_CASE( test_http_client_stress_test )
{
    cerr << "stress_test\n";
    // const int mask = 0x3ff; /* mask to use for displaying counts */
    // ML::Watchdog watchdog(300);
    auto proxies = make_shared<ServiceProxies>();
    auto doStressTest = [&] (int numParallel) {
        ::fprintf(stderr, "stress test with %d parallel connections\n",
                  numParallel);

        HttpGetService service(proxies);
        service.start();
        service.waitListening();

        string baseUrl("http://127.0.0.1:"
                       + to_string(service.port()));

        MessageLoop loop;
        loop.start();

        auto client = make_shared<HttpClient>(baseUrl, numParallel);
        loop.addSource("client", client);
        client->waitConnectionState(AsyncEventSource::CONNECTED);

        int maxReqs(30000), numReqs(0), missedReqs(0);
        int numResponses(0);

        auto onDone = [&] (const HttpRequest & rq,
                           HttpClientError errorCode, int status,
                           string && headers, string && body) {
            numResponses++;

            BOOST_CHECK_EQUAL(errorCode, HttpClientError::None);
            BOOST_CHECK_EQUAL(status, 200);

            int bodyNbr;
            try {
                bodyNbr = stoi(body);
            }
            catch (...) {
                ::fprintf(stderr, "exception when parsing body: %s\n",
                          body.c_str());
                throw;
            }

            if (numResponses == numReqs) {
                ML::futex_wake(numResponses);
            }
        };

        while (numReqs < maxReqs) {
            const char * url = "/counter";
            auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
            if (client->get(url, cbs)) {
                numReqs++;
                // if ((numReqs & mask) == 0 || numReqs == maxReqs) {
                //     ::fprintf(stderr, "performed %d requests\n", numReqs);
                // }
            }
            else {
                missedReqs++;
            }
        }

        ::fprintf(stderr, "all requests performed, awaiting responses...\n");
        while (numResponses < maxReqs) {
            int old(numResponses);
            ML::futex_wait(numResponses, old);
        }
        ::fprintf(stderr, "performed %d requests; missed: %d\n",
                  maxReqs, missedReqs);

        loop.shutdown();
        service.shutdown();
    };

    doStressTest(1);
    doStressTest(8);
    doStressTest(128);
}
#endif

#if 1
/* Ensure that the move constructor and assignment operator behave
   reasonably well. */
BOOST_AUTO_TEST_CASE( test_http_client_move_constructor )
{
    cerr << "move_constructor\n";
    ML::Watchdog watchdog(30);
    auto proxies = make_shared<ServiceProxies>();

    HttpGetService service(proxies);
    service.addResponse("GET", "/", 200, "coucou");
    service.start();
    service.waitListening();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));

    auto doGet = [&] (HttpClient & getClient) {
        loop.addSource("client", getClient);
        getClient.waitConnectionState(AsyncEventSource::CONNECTED);

        int done(false);

        auto onDone = [&] (const HttpRequest & rq,
                           HttpClientError errorCode, int status,
                           string && headers, string && body) {
            done = true;
            ML::futex_wake(done);
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);

        getClient.get("/", cbs);
        while (!done) {
            int old = done;
            ML::futex_wait(done, old);
        }

        loop.removeSource(&getClient);
        getClient.waitConnectionState(AsyncEventSource::DISCONNECTED);
    };

    /* move constructor */
    cerr << "testing move constructor\n";
    auto makeClient = [&] () {
        return HttpClient(baseUrl, 1);
    };
    HttpClient client1(move(makeClient()));
    doGet(client1);

    /* move assignment operator */
    cerr << "testing move assignment op.\n";
    HttpClient client2("http://nowhere", 1);
    client2 = move(client1);
    doGet(client2);

    service.shutdown();
}
#endif

#if 1
/* Ensure that an infinite number of requests can be queued when queue size is
 * 0, even from within callbacks. */
BOOST_AUTO_TEST_CASE( test_http_client_unlimited_queue )
{
    static const int maxLevel(4);

    ML::Watchdog watchdog(30);
    auto proxies = make_shared<ServiceProxies>();

    HttpGetService service(proxies);
    service.addResponse("GET", "/", 200, "coucou");
    service.start();
    service.waitListening();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:"
                   + to_string(service.port()));

    auto client = make_shared<HttpClient>(baseUrl, 4, 0);
    loop.addSource("client", client);
    client->waitConnectionState(AsyncEventSource::CONNECTED);

    atomic<int> pending(0);
    int done(0);

    function<void(int)> doGet = [&] (int level) {
        pending++;
        auto onDone = [&,level] (const HttpRequest & rq,
                                 HttpClientError errorCode, int status,
                                 string && headers, string && body) {
            if (level < maxLevel) {
                for (int i = 0; i < 10; i++) {
                    doGet(level+1);
                }
            }
            pending--;
            done++;
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
        client->get("/", cbs);
    };

    doGet(0);

    while (pending > 0) {
        ML::sleep(1);
        cerr << "requests done: " + to_string(done) + "\n";
    }

    loop.shutdown();
    service.shutdown();
}
#endif

#if 1
/* Test connection restoration after a timeout occurs. */
BOOST_AUTO_TEST_CASE( test_http_client_connection_timeout )
{
    ML::Watchdog watchdog(30);
    auto proxies = make_shared<ServiceProxies>();

    HttpGetService service(proxies);
    service.addResponse("GET", "/", 200, "coucou");
    service.start();
    service.waitListening();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:" + to_string(service.port()));

    auto client = make_shared<HttpClient>(baseUrl, 1);
    client->enableDebug(true);
    loop.addSource("client", client);
    client->waitConnectionState(AsyncEventSource::CONNECTED);

    int done(0);
    auto onDone = [&] (const HttpRequest & rq,
                       HttpClientError errorCode, int status,
                       string && headers, string && body) {
        done++;
        ML::futex_wake(done);
    };
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
    client->get("/timeout", cbs, {}, {}, 1);
    client->get("/", cbs, {}, {}, 1);

    while (done < 2) {
        ML::futex_wait(done, done);
    }

    loop.shutdown();
    service.shutdown();
}
#endif

#if 1
/* Test connection restoration after the server closes the connection, under
 * various circumstances. */
BOOST_AUTO_TEST_CASE( test_http_client_connection_closed )
{
    ML::Watchdog watchdog(30);
    auto proxies = make_shared<ServiceProxies>();

    HttpGetService service(proxies);
    service.portToUse = 8080;
    service.addResponse("GET", "/", 200, "coucou");
    service.start();
    service.waitListening();

    MessageLoop loop;
    loop.start();

    string baseUrl("http://127.0.0.1:" + to_string(service.port()));

    /* response sent, "Connection: close" header */
    {
        cerr << "* connection-close\n";
        auto client = make_shared<HttpClient>(baseUrl, 1);
        loop.addSource("client", client);
        client->waitConnectionState(AsyncEventSource::CONNECTED);

        int done(0);
        auto onDone = [&] (const HttpRequest & rq,
                           HttpClientError errorCode, int status,
                           string && headers, string && body) {
            done++;
            ML::futex_wake(done);
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
        client->get("/connection-close", cbs);
        client->get("/", cbs);

        while (done < 2) {
            ML::futex_wait(done, done);
        }

        loop.removeSourceSync(client.get());
    }

    /* response sent, no "Connection: close" header */
    {
        cerr << "* no connection-close\n";
        auto client = make_shared<HttpClient>(baseUrl, 1);
        loop.addSource("client", client);
        client->waitConnectionState(AsyncEventSource::CONNECTED);

        int done(0);
        auto onDone = [&] (const HttpRequest & rq,
                           HttpClientError errorCode, int status,
                           string && headers, string && body) {
            done++;
            ML::futex_wake(done);
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
        client->get("/quiet-connection-close", cbs);
        client->get("/", cbs);

        while (done < 2) {
            ML::futex_wait(done, done);
        }

        loop.removeSourceSync(client.get());
    }

    /* response not sent */
    {
        cerr << "* no response at all\n";
        auto client = make_shared<HttpClient>(baseUrl, 1);
        loop.addSource("client", client);
        client->waitConnectionState(AsyncEventSource::CONNECTED);

        int done(0);
        auto onDone = [&] (const HttpRequest & rq,
                           HttpClientError errorCode, int status,
                           string && headers, string && body) {
            done++;
            ML::futex_wake(done);
        };
        auto cbs = make_shared<HttpClientSimpleCallbacks>(onDone);
        client->get("/abrupt-connection-close", cbs);
        client->get("/", cbs);

        while (done < 2) {
            ML::futex_wait(done, done);
        }

        loop.removeSourceSync(client.get());
    }

    loop.shutdown();
    service.shutdown();
}
#endif
