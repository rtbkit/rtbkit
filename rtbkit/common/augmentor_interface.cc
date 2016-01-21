/* augmentor_interface.cc
   Mathieu Stefani, 20 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
*/

#include "augmentor_interface.h"

using namespace Datacratic;

namespace RTBKIT {

AugmentorInterface::AugmentorInterface(
    ServiceBase& parent, const std::string& serviceName)
  : ServiceBase(parent, serviceName)
{ }

AugmentorInterface::AugmentorInterface(
    std::shared_ptr<ServiceProxies> proxies,
    const std::string& serviceName)
  : ServiceBase(std::move(proxies), servicename)
{ }

void
AugmentorInterface::start() {
    MessageLoop::start();
}

void AugmentorInterface::stop() {
    MessageLoop::stop();
}

} // namespace RTBKIT


