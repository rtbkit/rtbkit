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
#include "soa/jsoncpp/reader.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace Datacratic;

/********************************************************************************/
/* ANALYTICS CLIENT                                                             */
/********************************************************************************/

void
AnalyticsClient::
init(const string & baseUrl)
{
    client = make_shared<HttpClient>(baseUrl, 1);
    client->sendExpect100Continue(false);
    addSource("analytics::client", client);
    cout << "analytics client is initialized" << endl;
    
    auto heartbeat = [&] (uint64_t wakeups) {
        checkHeartbeat();
    };
    addPeriodic("analytics::heartbeat", 1.0, heartbeat);
    
    auto syncFilters = [&] (uint64_t wakeups) {
        syncChannelFilters();
    };
    addPeriodic("analytics::syncFilters", 10.0, syncFilters);
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
sendEvent(const string & channel, const string & event)
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
    Json::Value payload(Json::objectValue);
    payload["channel"] = channel;
    payload["event"] = event;
    client->post(ressource, cbs, payload);
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
        if (status == 200) {
            if (!live) {
                live = true;
                syncChannelFilters();
            }
        }
        else live = false;
    };
    string ressource("/heartbeat");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client->get(ressource, cbs);
}

void
AnalyticsClient::
syncChannelFilters()
{
    auto onResponse = [&] (const HttpRequest & rq,
                           HttpClientError error,
                           int status,
                           string && headers,
                           string && body)
    {
        if (status != 200) return;
        
        Json::Value filters = Json::parse(body);
        if (filters.isObject()) {
            for ( auto it = filters.begin(); it != filters.end(); ++it) {
                channelFilter[it.memberName()] = (*it).asBool();
            }
        }
    };
    if (!live) return;
    string ressource("/v1/channels");
    auto cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    client->get(ressource, cbs);
}


/********************************************************************************/
/* ANALYTICS REST ENDPOINT                                                      */
/********************************************************************************/

AnalyticsRestEndpoint::
AnalyticsRestEndpoint(shared_ptr<ServiceProxies> proxies,
                      const std::string & serviceName)
    : ServiceBase(serviceName, proxies),
      RestServiceEndpoint(proxies->zmqContext),
      enableAll(false)
{
    httpEndpoint.allowAllOrigins();
}

void
AnalyticsRestEndpoint::
initChannels(unordered_map<string, bool> & channels)
{
    channelFilter = channels;
}

void
AnalyticsRestEndpoint::
init()
{
    // last param in init is threads increase it accordingly to needs.
    RestServiceEndpoint::init(getServices()->config, serviceName(), 0.005, 1);
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

    addRouteSyncReturn(versionNode,
                    "/event",
                    {"POST","PUT"},
                    "Add an event to the logs.",
                    "Returns a success notice.",
                    [] (const string & r) {
                        Json::Value response(Json::stringValue);
                        response = r;
                        return response;
                    },
                    &AnalyticsRestEndpoint::addEvent,
                    this,
                    JsonParam<string>("channel", "channel to use for event"),
                    JsonParam<string>("event", "event to publish")
            );

    addRouteSyncReturn(versionNode,
                    "/channels",
                    {"GET"},
                    "Gets the list of available channels",
                    "Returns a list of channels and their status",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::listChannels,
                    this
            );

    addRouteSyncReturn(versionNode,
                    "/enable",
                    {"POST", "PUT"},
                    "Start logging a certain channel of event.",
                    "Returns a success notice.",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::enableChannel,
                    this,
                    RestParamDefault<string>("channel", "event channel to enable", "")
            );
 
    addRouteSyncReturn(versionNode,
                    "/disable",
                    {"POST", "PUT"},
                    "Stop logging a certain channel of event.",
                    "Returns a success notice.",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::disableChannel,
                    this,
                    RestParamDefault<string>("channel", "event channel to disable", "")
            ); 
}

string
AnalyticsRestEndpoint::
print(const string & channel, const string & event)
{
    cout << channel << " " << event << endl;
    return "success";
}

string
AnalyticsRestEndpoint::
addEvent(const string & channel, const string & event)
{
    if (enableAll)
        return print(channel, event);

    if (channelFilter.find(channel) != channelFilter.end()
            && channelFilter[channel]) {
        return print(channel, event);
    } else {
        return "channel not found or not enabled";
    }
}

Json::Value
AnalyticsRestEndpoint::
listChannels()
{
    Json::Value response(Json::objectValue);
    for (const auto & channel : channelFilter)
        response[channel.first] = channel.second;
    return response;
}

Json::Value
AnalyticsRestEndpoint::
enableChannel(const string & channel)
{
    if (channel == "ALL"){
        enableAllChannels();
    }
    else if (!channel.empty() && !channelFilter[channel])
        channelFilter[channel] = true;
    return listChannels();
}

void
AnalyticsRestEndpoint::
enableAllChannels()
{
    enableAll = true;
    for (auto & channel : channelFilter)
        channel.second = true;
}

void
AnalyticsRestEndpoint::
disableAllChannels()
{
    enableAll = false;
    for (auto & channel : channelFilter)
        channel.second = false;
}

Json::Value
AnalyticsRestEndpoint::
disableChannel(const string & channel)
{
    if (channel == "ALL") {
        disableAllChannels();
    }
    else if (!channel.empty() && channelFilter[channel])
        channelFilter[channel] = false;
    return listChannels();
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

