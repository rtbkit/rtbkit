/* zmq_augmentor_interface.h
   Mathieu Stefani, 20 january 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   An augmentor interface that uses zeromq to communicate with the augmentor
*/

#pragma once

#include "rtbkit/common/augmentor_interface.h"
#include "soa/service/zmq_endpoint.h"

namespace RTBKIT {

struct ZmqAugmentorInstanceInfo : public AugmentorInstanceInfo {
    ZmqAugmentorInstanceInfo(const std::string& addr, int maxInFlight)
        : AugmentorInstanceInfo(maxInFlight)
        , addr(addr)
    { }

    std::string addr;

    std::string address() const {
        return addr;
    }

};

struct ZmqAugmentorInterface : public AugmentorInterface {

    ZmqAugmentorInterface(Datacratic::ServiceBase& parent,
            const std::string& serviceName = "augmentationLoop",
            const Json::Value& config = Json::Value());

    ZmqAugmentorInterface(std::shared_ptr<Datacratic::ServiceProxies> proxies,
            const std::string& serviceName = "augmentationLoop",
            const Json::Value& config = Json::Value());

    void init();
    void bindTcp(const Datacratic::PortRange& range);

protected:

    void doSendAugmentMessage(
                const std::shared_ptr<AugmentorInstanceInfo>& instance,
                const std::string& augmentorName,
                const std::shared_ptr<Auction>& auction,
                const std::set<std::string>& agents,
                Datacratic::Date date);

private:
    /// Connection to all of our augmentors
    Datacratic::ZmqNamedClientBus toAugmentors;

    void handleAugmentorMessage(const std::vector<std::string> & message);
    void doConfig(const std::vector<std::string>& message);
    void doResponse(const std::vector<std::string>& message);
};

} // namespace RTBKIT
