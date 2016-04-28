/* bid_request_pipeline.cc
   Mathieu Stefani, 12 février 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.

   The Bid Request Pipeline
*/

#include "bid_request_pipeline.h"

namespace RTBKIT {

BidRequestPipeline::BidRequestPipeline(
    std::shared_ptr<Datacratic::ServiceProxies> proxies, std::string serviceName)
        : ServiceBase(std::move(serviceName), std::move(proxies))
{ }

std::shared_ptr<BidRequestPipeline>
BidRequestPipeline::create(
        std::string serviceName,
        std::shared_ptr<ServiceProxies> proxies,
        const Json::Value& json)
{
    auto type = json.get("type", "null").asString();
    auto factory = PluginInterface<BidRequestPipeline>::getPlugin(type);

    if(serviceName.empty()) {
        serviceName = json.get("serviceName", "pipeline").asString();
    }

    return std::shared_ptr<BidRequestPipeline>(factory(std::move(serviceName), std::move(proxies), json));
}

} // namespace RTBKIT
