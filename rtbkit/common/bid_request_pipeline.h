/* bid_request_pipeline.h
   Mathieu Stefani, 12 février 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
 
*/


#pragma once

#include "soa/service/service_base.h"
#include "soa/service/http_header.h"
#include "soa/jsoncpp/json.h"
#include "rtbkit/common/exchange_connector.h"
#include "rtbkit/common/bid_request.h"
#include <string>
#include <memory>

namespace RTBKIT {

enum class PipelineStatus {
    Continue,
    Stop
};

class BidRequestPipeline : public Datacratic::ServiceBase {
public:

    BidRequestPipeline(
            std::shared_ptr<Datacratic::ServiceProxies> proxies, std::string serviceName);

    typedef std::function<BidRequestPipeline *(
            std::string serviceName,
            const std::shared_ptr<Datacratic::ServiceProxies>& proxies,
            const Json::Value& json)> Factory;

    static std::string libNameSufix() { return "pipeline"; }

    static std::shared_ptr<BidRequestPipeline>
    create(
            std::string serviceName,
            std::shared_ptr<Datacratic::ServiceProxies> proxies,
            const Json::Value& json);

    virtual PipelineStatus
    preBidRequest(
            const ExchangeConnector* exchange,
            const Datacratic::HttpHeader& header,
            const std::string& payload) = 0;

    virtual PipelineStatus
    postBidRequest(
            const ExchangeConnector* exchange,
            const std::shared_ptr<Auction>& auction) = 0;

};

} // namespace RTBKIT

