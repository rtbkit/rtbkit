/* http_augmentor_interface.cc
   Mathieu Stefani, 21 january 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
*/

#include "http_augmentor_interface.h"
#include "soa/service/http_client.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/openrtb/openrtb_parsing.h"

using namespace Datacratic;

namespace RTBKIT {

namespace {
    DefaultDescription<OpenRTB::BidRequest> desc;
}

HttpAugmentorInterface::HttpAugmentorInterface(
    std::shared_ptr<ServiceProxies> proxies,
    const std::string& serviceName,
    const Json::Value& config)
  : AugmentorInterface(std::move(proxies), serviceName)
{
    /*
    {
        "type":"http",
        "frequencyCap": {
            "instances": [
                {
                    "address": "http://127.0.0.1:9985",
                    "name": "rtb1.useast1b.freqCap",
                    "maxInFlight": 100
                },
                {
                    "address": "http://10.0.0.2:9985",
                    "name": "rtb2.useast1b.freqCap",
                    "maxInFlight": 20
                }
            ]
        }
    }
    */

    /* @Validation */
    for (auto it = config.begin(), end = config.end(); it != end; ++it) {
        if(it.memberName() == "type")
            continue;
        const std::string augmentorName = it.memberName();
        auto val = *it;
        auto instances = val["instances"];
        Instances info;
        for (const auto& instance: instances) {
            auto address = instance["address"].asString();
            auto path = instance.get("path", "/").asString();
            auto name = instance["name"].asString();
            auto maxInFlight = instance.get("maxInFlight", 100).asInt();

            auto client = std::make_shared<HttpClient>(address);
            info.push_back(std::make_shared<HttpAugmentorInstanceInfo>(maxInFlight, client, path, name));
        }
        augmentors[augmentorName] = info;
    }
}

void
HttpAugmentorInterface::init()
{
    for (auto& augmentor: augmentors) {
        std::string augmentorName = std::move(augmentor.first);

        for (auto& instance: augmentor.second) {
            addSource(ML::format("HttpAugmentorInterface::%s.client", instance->name), instance->client);
            onConnection(std::move(augmentorName), std::move(instance));
        }
    }

    augmentors.clear();
}

void
HttpAugmentorInterface::bindTcp(const PortRange& range)
{
}

void
HttpAugmentorInterface::sendAugmentMessage(
    const std::shared_ptr<AugmentorInstanceInfo>& instance,
    const std::string& name,
    const std::shared_ptr<Auction>& auction,
    const std::set<std::string>& agents,
    Date date)
{
    auto httpInstance = std::static_pointer_cast<HttpAugmentorInstanceInfo>(instance);
    auto request = auction->request;


    std::string requestStr;
    StructuredJsonPrintingContext context;

    auto openrtbVersion = request->protocolVersion;
    if (openrtbVersion.empty()) openrtbVersion = "2.1";
    auto parser = OpenRTBBidRequestParser::openRTBBidRequestParserFactory(openrtbVersion);
    OpenRTB::BidRequest openrtbRequest  = parser->toBidRequest(*request);

    auto &augExt = openrtbRequest.ext["augmentation"];
    Json::Value agentsExt(Json::arrayValue);
    for (const auto& agent: agents)
        agentsExt.append(agent);

    augExt["agents"] = std::move(agentsExt);
    desc.printJson(&openrtbRequest, context);
    requestStr = context.output.toString();

    RestParams headers { { "X-Openrtb-Version", openrtbVersion } };

    auto callback = std::make_shared<HttpClientSimpleCallbacks>(
            [=](const HttpRequest &, HttpClientError errorCode,
                int statusCode, const std::string &, std::string &&body)
            {
                /* To be continued */
            }
    );


    HttpRequest::Content reqContent { requestStr, "application/json" };
    httpInstance->client->post(httpInstance->path, callback, reqContent, { }, headers);
}

} // namespace RTBKIT

namespace {

struct AtInit {
    AtInit()
    {
      PluginInterface<AugmentorInterface>::registerPlugin("http",
          [](std::string const &serviceName,
             std::shared_ptr<ServiceProxies> const &proxies,
             Json::Value const &json)
          {
              return new HttpAugmentorInterface(proxies, serviceName, json);
          });
    }
} atInit;

}
