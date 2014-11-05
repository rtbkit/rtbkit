/** analytics.h                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics plugin used to  
*/

#include <iostream>
#include <functional>

#include "analytics.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/rest_request_binding.h"
#include "jml/arch/timers.h"

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
    client = make_shared<HttpClient>(baseUrl, 1);
    client->sendExpect100Continue(false);
    addSource("analytics::client", client);
    cout << "analytics client is initialized" << endl;

    auto heartbeat = [&] (uint64_t wakeups) {
        checkHeartbeat();
    };
    addPeriodic("analytics::heartbeat", 1.0, heartbeat);
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
sendEvent(const string type, const string event)
{
    auto onResponse = [] (const HttpRequest & rq,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        if (status != 200) {
            cout << "status: " << status << endl
                 << "error: " << error << endl;
        }
    };
    string ressource("/v1/event");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client->post(ressource, cbs, {}, { { "type",  type },
                                       { "event", event } });
}

void
AnalyticsClient::
checkHeartbeat()
{
    auto onResponse = [&] (const HttpRequest & rq,
                          HttpClientError error,
                          int status,
                          string && headers,
                          string && body)
    {
        if (status == 200)
           live = true;
        else
           live = false;
    };
    string ressource("/heartbeat");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client->get(ressource, cbs);
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
init(bool test)
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
            recordHit("beat");
            connection.sendResponse(200, "beat");
            return RestRequestRouter::MR_YES;
        };

    router.addRoute("/heartbeat", "GET", "availability request", pingRoute,
            Json::Value());

    auto & versionNode = router.addSubRouter("/v1", "version 1 of API");

    if (test) {
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
                       &AnalyticsRestEndpoint::testEvent,
                       this,
                       RestParamDefault<string>("type", "event type to add to list"),
                       RestParamDefault<string>("event", "win event to add to list", "")
                );
    } else {
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
}

string
AnalyticsRestEndpoint::
addEvent(const string & type, const string & event)
{
    cout << type << " " << event << endl;
    return "success";
}

string
AnalyticsRestEndpoint::
testEvent(const string & type, const string & event)
{
    if (type != "" && event != "")
        return "success";
    else
        return "error";
}

pair<string, string>
AnalyticsRestEndpoint::
bindTcp(int port)
{
    pair<string, string> location;
    if (port)
        location = RestServiceEndpoint::bindTcp(PortRange(), PortRange(port));
    else
        location = RestServiceEndpoint::bindTcp(PortRange(), getServices()->ports->getRange("analytics"));
    
    cout << "Analytics listening on http port: " << location.second << endl;
    return location;
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

