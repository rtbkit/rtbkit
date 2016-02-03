/* zmq_augmentor_interface.cc
   Mathieu Stefani, 20 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Implementation of the zeromq augmentor interface
*/

#include "zmq_augmentor_interface.h"

using namespace Datacratic;

namespace RTBKIT {

ZmqAugmentorInterface::ZmqAugmentorInterface(
        ServiceBase& parent,
        const std::string& serviceName,
        const Json::Value& config)
    : AugmentorInterface(parent, serviceName)
    , toAugmentors(getZmqContext())
{ }

ZmqAugmentorInterface::ZmqAugmentorInterface(
        std::shared_ptr<ServiceProxies> proxies,
        const std::string& serviceName,
        const Json::Value& config)
    : AugmentorInterface(std::move(proxies), serviceName)
    , toAugmentors(getZmqContext())
{ }

void
ZmqAugmentorInterface::init() {

    registerServiceProvider(serviceName(), { "rtbRouterAugmentation" });

    AugmentorInterface::init();

    toAugmentors.init(getServices()->config, serviceName() + "/augmentors");

    toAugmentors.clientMessageHandler
        = [&] (const std::vector<std::string> & message)
        {
            //cerr << "got augmentor message " << message << endl;
            handleAugmentorMessage(message);
        };

    toAugmentors.bindTcp(getServices()->ports->getRange("augmentors"));

    toAugmentors.onConnection = [=] (const std::string & client)
        {
            cerr << "augmentor " << client << " has connected" << endl;
        };

    toAugmentors.onDisconnection = [=] (const std::string & client)
        {
            cerr << "augmentor " << client << " has disconnected" << endl;
            onDisconnection(client);
        };

    addSource("ZmqAugmentorInterface::toAugmentors", toAugmentors);

}

void
ZmqAugmentorInterface::bindTcp(const PortRange& range) {
    toAugmentors.bindTcp(range);
}

void
ZmqAugmentorInterface::doSendAugmentMessage(
        const std::shared_ptr<AugmentorInstanceInfo>& instance,
        const std::string& name,
        const std::shared_ptr<Auction>& auction,
        const std::set<std::string>& agents,
        Date date)
{
    auto zmqInstance = std::static_pointer_cast<ZmqAugmentorInstanceInfo>(instance);

    std::ostringstream availableAgentsStr;
    ML::DB::Store_Writer writer(availableAgentsStr);
    writer.save(agents);

    toAugmentors.sendMessage(
                zmqInstance->addr,
                "AUGMENT", "1.0", name,
                auction->id.toString(),
                auction->requestStrFormat,
                auction->requestStr,
                availableAgentsStr.str(),
                date);
}

void
ZmqAugmentorInterface::handleAugmentorMessage(const std::vector<std::string>& message)
{
    const std::string & type = message.at(1); 
    if (type == "CONFIG") {
        doConfig(message);
    }
    else if (type == "RESPONSE") {
        doResponse(message);
    }
    else throw ML::Exception("error handling unknown "
                             "augmentor message of type "
                             + type);
}

void
ZmqAugmentorInterface::doConfig(const std::vector<std::string>& message)
{
    ExcCheckGreaterEqual(message.size(), 4, "config message has wrong size");
    ExcCheckLessEqual(message.size(), 5, "config message has wrong size");

    auto addr = message[0];
    auto version = message[2];
    auto name = message[3];

    int maxInFlight = -1;
    if (message.size() >= 5)
        maxInFlight = std::stoi(message[4]);
    if (maxInFlight < 0) maxInFlight = 3000;

    ExcCheckEqual(version, "1.0", "unknown version for config message");
    ExcCheck(!name.empty(), "no augmentor name specified");

    auto info = std::make_shared<ZmqAugmentorInstanceInfo>(addr, maxInFlight);
    onConnection(std::move(name), std::move(info));

}

void
ZmqAugmentorInterface::doResponse(const std::vector<std::string>& message)
{
    //cerr << "doResponse " << message << endl;

    ExcCheckEqual(message.size(), 7, "response message has wrong size");

    const string & version = message[2];
    ExcCheckEqual(version, "1.0", "unknown response version");

    auto addr = message[0];
    Date startTime = Date::parseSecondsSinceEpoch(message[3]);
    Id id(message[4]);
    auto augmentor = message[5];
    auto augmentation = message[6];

    onResponse(AugmentationResponse(std::move(addr), startTime, id, std::move(augmentor), std::move(augmentation)));
}

} // namespace RTBKIT

namespace {

struct AtInit {
    AtInit()
    {
      PluginInterface<AugmentorInterface>::registerPlugin("zmq",
          [](std::string const &serviceName,
             std::shared_ptr<ServiceProxies> const &proxies,
             Json::Value const &json)
          {
              return new ZmqAugmentorInterface(proxies, serviceName, json);
          });
    }
} atInit;

}
