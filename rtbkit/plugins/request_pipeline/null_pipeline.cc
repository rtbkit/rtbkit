/* null_pipeline.cc
   Mathieu Stefani, 12 février 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Implementation of the Null Pipeline
*/

#include "null_pipeline.h"

using namespace Datacratic;

namespace RTBKIT {

NullBidRequestPipeline::NullBidRequestPipeline(
        std::shared_ptr<ServiceProxies> proxies, std::string serviceName,
        const Json::Value& json)
    : BidRequestPipeline(std::move(proxies), std::move(serviceName))
{ }

PipelineStatus
NullBidRequestPipeline::preBidRequest(
        const ExchangeConnector* exchange,
        const HttpHeader& header,
        const std::string& payload) {
    return PipelineStatus::Continue;
}

PipelineStatus
NullBidRequestPipeline::postBidRequest(
        const ExchangeConnector* exchange,
        const std::shared_ptr<Auction>& auction) {
    return PipelineStatus::Continue;
}

namespace {

struct AtInit {
    AtInit()
    {
      PluginInterface<BidRequestPipeline>::registerPlugin("null",
          [](std::string serviceName,
             std::shared_ptr<ServiceProxies> proxies,
             Json::Value const &json)
          {
              return new NullBidRequestPipeline(std::move(proxies), std::move(serviceName), json);
          });
    }
} atInit;

}

} // namespace RTBKIT

