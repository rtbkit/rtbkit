/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/

#include <string>

#include "analytics.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"

using namespace std;
using namespace Datacratic;

AnalyticsClient::
AnalyticsClient(int port, const string & address) : MessageLoop(), client(string(address + ":" + to_string(port)), 1)
{
    string baseUrl = address + ":" + to_string(port);
    client = HttpClient(baseUrl, 1);

    start();
    addSource("analyticsClient", client);
}

void
AnalyticsClient::
sendWin(const string & body)
{
    using namespace std;
    using namespace Datacratic;

    auto onResponse = [] (const HttpRequest & rq,
            HttpClientError error,
            int status,
            string && headers,
            string && body) {
        std::cout << "status: " << status << std::endl
            << "error: " << error << std::endl;
    };
    HttpRequest::Content content(body, "text/plain");
    client.waitConnectionState(AsyncEventSource::CONNECTED);
    string ressource("/win");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client.post(ressource, cbs, content);
    client.waitConnectionState(AsyncEventSource::DISCONNECTED);
}

AnalyticsRestEndpoint::
AnalyticsRestEndpoint(shared_ptr<ServiceProxies> proxies) : RestServiceEndpoint(proxies->zmqContext)
{
    httpEndpoint.allowAllOrigins();
}

void
AnalyticsRestEndpoint::
init()
{

}

