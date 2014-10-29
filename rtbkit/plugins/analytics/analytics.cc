/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/

#include <iostream>

#include "analytics.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/rest_request_binding.h"

using namespace std;
using namespace Datacratic;

AnalyticsClient::
AnalyticsClient(const string & baseUrl) : baseUrl(baseUrl) 
{
}

AnalyticsClient::
AnalyticsClient(int port, const string & address) : baseUrl(address + ":" + to_string(port))
{
}

void
AnalyticsClient::
init()
{
    cout << "url: " << baseUrl << endl;
    client = make_shared<HttpClient>(baseUrl, 1);
    client->sendExpect100Continue(false);
    addSource("analyticsClient", client);
}

void
AnalyticsClient::
start()
{
    MessageLoop::start();
}

void
AnalyticsClient::
shutdown()
{
    MessageLoop::shutdown();
}

void
AnalyticsClient::
sendEvent(const string & type, const string & event)
{
    auto onResponse = [] (const HttpRequest & rq,
            HttpClientError error,
            int status,
            string && headers,
            string && body) {
        std::cout << "status: " << status << std::endl
                  << "headers: " << headers << std::endl
                  << "body: " << body << std::endl
                  << "error: " << error << std::endl;
    };
    string ressource("/v1/event");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client->post(ressource, cbs, {}, { { "type", type },
                                      { "event", event } });
}



AnalyticsRestEndpoint::
AnalyticsRestEndpoint(shared_ptr<ServiceProxies> proxies,
                      const std::string & serviceName)
    : ServiceBase(serviceName, proxies),
      RestServiceEndpoint(proxies->zmqContext)
{
    httpEndpoint.allowAllOrigins();
}

void
AnalyticsRestEndpoint::
init()
{
    RestServiceEndpoint::init(getServices()->config, serviceName());
    onHandleRequest = router.requestHandler();
    registerServiceProvider(serviceName(), { "analytics" });
    router.description = "Analytics REST API";

    router.addHelpRoute("/", "GET");

    RestRequestRouter::OnProcessRequest pingRoute
        = [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context) {
            recordHit("ping");
            connection.sendResponse(200, "pong");
            return RestRequestRouter::MR_YES;
        };

    router.addRoute("/ping", "GET", "availability request", pingRoute,
            Json::Value());

    auto & versionNode = router.addSubRouter("/v1", "version 1 of API");

    addRouteSyncReturn(versionNode,
                       "/event",
                       {"POST","PUT"},
                       "Add a win to the logs.",
                       "Returns a success notice.",
                       [] (const string & r) {
                            Json::Value response(Json::stringValue);
                            response = r;
                            return response;
                       },
                       &AnalyticsRestEndpoint::addEvent,
                       this,
                       RestParamDefault<string>("type", "event type to add to list"),
                       RestParamDefault<string>("event", "win event to add to list", "")
                       );

}

string
AnalyticsRestEndpoint::
addEvent(const string & type, const string & event)
{
    cout << "type: " << type << ", event: " << event << endl;
    return "success";
}

pair<string, string>
AnalyticsRestEndpoint::
bindTcp()
{
    return RestServiceEndpoint::bindTcp(PortRange(), getServices()->ports->getRange("analytics"));
}

void
AnalyticsRestEndpoint::
start()
{
    RestServiceEndpoint::start();
}

void
AnalyticsRestEndpoint::
shutdown()
{
    RestServiceEndpoint::shutdown();
}

