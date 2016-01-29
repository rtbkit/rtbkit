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

} // namespace RTBKIT


