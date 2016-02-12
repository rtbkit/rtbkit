/* null_pipeline.h
   Mathieu Stefani, 12 février 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   A Bid Request Pipeline that does nothing
*/

#pragma once

#include "rtbkit/common/bid_request_pipeline.h"

namespace RTBKIT {

class NullBidRequestPipeline : public BidRequestPipeline {
public:

    NullBidRequestPipeline(
            std::shared_ptr<Datacratic::ServiceProxies> proxies, std::string serviceName,
            const Json::Value& json);
    
    PipelineStatus
    preBidRequest(
            const ExchangeConnector* exchange,
            const HttpHeader& header,
            const std::string& payload);

    PipelineStatus
    postBidRequest(
            const ExchangeConnector* exchange,
            const std::shared_ptr<Auction>& auction);

};

} // namespace RTBKIT
