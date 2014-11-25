/** analytics_publisher.cc                                     -*- C++ -*-
      Michael Burkat, 22 Oct 2014
        Copyright (c) 2014 Datacratic.  All rights reserved.

        Analytics publisher used to send data to an analytics endpoint.
*/

#include <iostream>
#include <functional>

#include "analytics_publisher.h"
#include "soa/jsoncpp/value.h"
#include "soa/jsoncpp/reader.h"

using namespace std;
using namespace Datacratic;

/********************************************************************************/
/* ANALYTICS PUBLISHER                                                          */
/********************************************************************************/

void
AnalyticsPublisher::
init(const string & baseUrl, const int numConnections)
{
    client = make_shared<HttpClient>(baseUrl, numConnections);
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

    initialized = true;
}

void
AnalyticsPublisher::
start()
{
    MessageLoop::start();
}

void
AnalyticsPublisher::
shutdown()
{
    MessageLoop::shutdown();
}

void
AnalyticsPublisher::
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
    auto const & cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::objectValue);
    payload["channel"] = channel;
    payload["event"] = event;
    client->post(ressource, cbs, payload);
}

void
AnalyticsPublisher::
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
AnalyticsPublisher::
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
            std::lock_guard<std::mutex> lock(mu);
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

