/* augmentor_interface.cc
   Mathieu Stefani, 20 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
*/

#include "augmentor_interface.h"

using namespace Datacratic;

namespace RTBKIT {

AugmentorInterface::AugmentorInterface(
    ServiceBase& parent, const std::string& serviceName)
  : ServiceBase(serviceName, parent)
{ }

AugmentorInterface::AugmentorInterface(
    std::shared_ptr<ServiceProxies> proxies,
    const std::string& serviceName)
  : ServiceBase(serviceName, std::move(proxies))
{ }

void
AugmentorInterface::start() {
    MessageLoop::start();
}

void AugmentorInterface::shutdown() {
    MessageLoop::shutdown();
}

std::shared_ptr<AugmentorInterface> AugmentorInterface::create(
        std::string serviceName,
        std::shared_ptr<ServiceProxies> const & proxies,
        Json::Value const & json) {

    std::string type = "http";
    if(json == Json::Value::null)
        type = "zmq";
    else
        type = json.get("type", "zmq").asString();

    auto factory = PluginInterface<AugmentorInterface>::getPlugin(type);

    if(serviceName.empty()) {
        serviceName = json.get("serviceName", "augmentor").asString();
    }

    return std::shared_ptr<AugmentorInterface>(factory(serviceName, proxies, json));
}

} // namespace RTBKIT

