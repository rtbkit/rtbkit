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
                                 std::string const & name) :
    ServiceBase(name, parent),
    router(nullptr),
    bridge(nullptr) {
}

BidderInterface::BidderInterface(std::shared_ptr<ServiceProxies> proxies,
                                 std::string const & name) :
    ServiceBase(name, proxies),
    router(nullptr),
    bridge(nullptr) {
}

void BidderInterface::init(AgentBridge * value, Router * r) {
    router = r;
    bridge = value;
}

void BidderInterface::start()
{
}

namespace {
    typedef std::lock_guard<ML::Spinlock> Guard;
    static ML::Spinlock lock;
    static std::unordered_map<std::string, BidderInterface::Factory> factories;
}


BidderInterface::Factory getFactory(std::string const & name) {
    // see if it's already existing
    {
        Guard guard(lock);
        auto i = factories.find(name);
        if (i != factories.end()) return i->second;
    }

    // else, try to load the bidder library
    std::string path = "lib" + name + "_bidder.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        throw ML::Exception("couldn't load bidder library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(lock);
    auto i = factories.find(name);
    if (i != factories.end()) return i->second;

    throw ML::Exception("couldn't find bidder name " + name);
}


void BidderInterface::registerFactory(std::string const & name, Factory callback) {
    Guard guard(lock);
    if (!factories.insert(std::make_pair(name, callback)).second)
        throw ML::Exception("already had a bidder factory registered");
}


std::shared_ptr<BidderInterface> BidderInterface::create(
        std::string name,
        std::shared_ptr<ServiceProxies> const & proxies,
        Json::Value const & json) {
    auto type = json.get("type", "unknown").asString();
    auto factory = getFactory(type);
    if(name.empty()) {
        name = json.get("name", "bidder").asString();
    }

    return std::shared_ptr<BidderInterface>(factory(name, proxies, json));
}

