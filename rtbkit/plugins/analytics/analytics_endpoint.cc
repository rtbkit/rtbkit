/** analytics_endpoint.cc                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics endpoint used to log events on different channels.
*/

#include <iostream>
#include <functional>

#include "analytics_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/http_client.h"
#include "soa/service/rest_request_binding.h"
#include "soa/jsoncpp/reader.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace Datacratic;

/********************************************************************************/
/* ANALYTICS REST ENDPOINT                                                      */
/********************************************************************************/

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
                    "Returns a list of channels and their status.",
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
                    "Returns a list of channels.",
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
                    "Returns a list of channels.",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::disableChannel,
                    this,
                    RestParamDefault<string>("channel", "event channel to disable", "")
            );

    addRouteSyncReturn(versionNode,
                    "/enableAll",
                    {"POST", "PUT"},
                    "Start logging on all known channels of events.",
                    "Returns a list of channels.",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::enableAllChannels,
                    this
            );

    addRouteSyncReturn(versionNode,
                    "/disableAll",
                    {"POST", "PUT"},
                    "Stop logging on all channels of events.",
                    "Returns a list of channels.",
                    [] (const Json::Value & lst) {
                        return lst;
                    },
                    &AnalyticsRestEndpoint::disableAllChannels,
                    this
            ); 

}

string
AnalyticsRestEndpoint::
print(const string & channel, const string & event) const
{
    recordHit("channel." + channel);
    cout << channel << " " << event << endl;
    return "success";
}

string
AnalyticsRestEndpoint::
addEvent(const string & channel, const string & event) const
{
    boost::shared_lock<boost::shared_mutex> lock(access);
    auto it = channelFilter.find(channel);
    if (it == channelFilter.end() ||  !it->second) 
        return "channel not found or not enabled";
 
    return print(channel, event);
}

Json::Value
AnalyticsRestEndpoint::
listChannels() const
{
    boost::shared_lock<boost::shared_mutex> lock(access);
    Json::Value response(Json::objectValue);
    for (const auto & channel : channelFilter) {
        response[channel.first] = channel.second;
        //cout << channel.first << " " << channel.second << endl;
    }
    return response;
}

Json::Value
AnalyticsRestEndpoint::
enableChannel(const string & channel)
{
    {
        boost::lock_guard<boost::shared_mutex> guard(access);
        if (!channel.empty() && !channelFilter[channel])
            channelFilter[channel] = true;
    }
    return listChannels();
}

Json::Value
AnalyticsRestEndpoint::
enableAllChannels()
{
    {
        boost::lock_guard<boost::shared_mutex> guard(access);
        for (auto & channel : channelFilter)
            channel.second = true;
    }
    return listChannels();
}

Json::Value
AnalyticsRestEndpoint::
disableAllChannels()
{
    {
        boost::lock_guard<boost::shared_mutex> guard(access);
        for (auto & channel : channelFilter)
            channel.second = false;
    }
    return listChannels();
}

Json::Value
AnalyticsRestEndpoint::
disableChannel(const string & channel)
{
    {
        boost::lock_guard<boost::shared_mutex> guard(access);
        if (!channel.empty() && channelFilter[channel])
            channelFilter[channel] = false;
    }
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

