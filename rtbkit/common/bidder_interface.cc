/* bidder_interface.cc
   Eric Robert, 2 April 2014
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "jml/db/persistent.h"
#include "rtbkit/common/messages.h"
#include "bidder_interface.h"
#include <dlfcn.h>

using namespace Datacratic;
using namespace RTBKIT;

BidderInterface::BidderInterface(ServiceBase & parent,
                                 std::string const & serviceName) :
    ServiceBase(serviceName, parent),
    router(nullptr),
    bridge(nullptr) {
}

BidderInterface::BidderInterface(std::shared_ptr<ServiceProxies> proxies,
                                 std::string const & serviceName) :
    ServiceBase(serviceName, proxies),
    router(nullptr),
    bridge(nullptr) {
}

void BidderInterface::setInterfaceName(const std::string &name) {
    this->name = name;
}

std::string BidderInterface::interfaceName() const {
    return name;
}

void BidderInterface::init(AgentBridge * value, Router * r) {
    router = r;
    bridge = value;
}

void BidderInterface::start()
{
}

void BidderInterface::shutdown()
{
}

std::shared_ptr<BidderInterface> BidderInterface::create(
        std::string serviceName,
        std::shared_ptr<ServiceProxies> const & proxies,
        Json::Value const & json) {

    auto type = json.get("type", "unknown").asString();

    //auto factory = getFactory(type);
    auto factory = PluginInterface<BidderInterface>::getPlugin(type);
    
    if(serviceName.empty()) {
        serviceName = json.get("serviceName", "bidder").asString();
    }

    return std::shared_ptr<BidderInterface>(factory(serviceName, proxies, json));
}

