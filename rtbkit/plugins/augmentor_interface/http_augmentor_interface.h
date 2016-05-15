/* http_augmentor_interface.h
   Mathieu Stefani, 21 january 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   An augmentor interface that uses http to communicate with the augmentor
*/

#pragma once

#include "rtbkit/common/augmentor_interface.h"
#include "soa/service/http_client.h"
#include <unordered_map>

namespace RTBKIT {

struct HttpAugmentorInstanceInfo : public AugmentorInstanceInfo {
    HttpAugmentorInstanceInfo(
        int maxInFlight,
        const std::shared_ptr<HttpClient>& client,
        const std::string& path,
        const std::string& name)

      : AugmentorInstanceInfo(maxInFlight)
      , client(client)
      , path(path)
      , name(name)
    { }

    std::string address() const {
        return name;
    }

    std::shared_ptr<HttpClient> client;
    std::string path;
    std::string name;

};

struct HttpAugmentorInterface : public AugmentorInterface {

    HttpAugmentorInterface(std::shared_ptr<Datacratic::ServiceProxies> proxies,
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
    // We keep a list of augmentors ourself so that we can add the http clients
    // to our own MessageLoop and call onConnection later on (we can not call it
    // directly from the constructor otherwise the AugmentationLoop won't be notified,
    // so instead, we defer the call).
    typedef std::vector<std::shared_ptr<HttpAugmentorInstanceInfo>> Instances;
    std::unordered_map<std::string, Instances> augmentors;

};

} // namespace RTBKIT
