/* augmentor_interface.h
   Mathieu Stefani, 20 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   An abstract application layer between the augmentation loop and the augmentors
*/

#pragma once

#include "soa/service/message_loop.h"
#include "soa/types/date.h"
#include "soa/types/id.h"
#include "soa/service/port_range_service.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/plugin_interface.h"

namespace RTBKIT {

/** Information about a specific augmentor which belongs to an augmentor class.
 */
struct AugmentorInstanceInfo {
    AugmentorInstanceInfo(int maxInFlight = 0) :
        numInFlight(0), maxInFlight(maxInFlight)
    {}

    int numInFlight;
    int maxInFlight;

    /* @Todo @Review 
       I'm not sure if we should keep the notion of "address" here.
       Maybe should we consider that an instance has a "serviceName" instead
    */

    virtual std::string address() const = 0;
};

struct AugmentationResponse {
    AugmentationResponse(
            std::string addr, Datacratic::Date startTime,
            Datacratic::Id auctionId, std::string augmentor, std::string augmentation)
        : addr(std::move(addr))
        , startTime(startTime)
        , auctionId(auctionId)
        , augmentor(std::move(augmentor))
        , augmentation(std::move(augmentation))
    { }

    AugmentationResponse(){}

    std::string addr;
    Datacratic::Date startTime;
    Datacratic::Id auctionId;
    std::string augmentor;
    std::string augmentation;
};

struct AugmentorInterface : public MessageLoop, public ServiceBase {

    typedef std::function<void(std::string&&, std::shared_ptr<AugmentorInstanceInfo>&&)> OnConnection;
    typedef std::function<void(const std::string&)> OnDisconnection;

    typedef std::function<void(AugmentationResponse&&)> OnResponse;

    OnConnection onConnection;
    OnDisconnection onDisconnection;

    OnResponse onResponse;

    AugmentorInterface(
        Datacratic::ServiceBase& parent, const std::string& serviceName);
    AugmentorInterface(
        std::shared_ptr<Datacratic::ServiceProxies> proxies, const std::string& serviceName);


    virtual void init();
    virtual void bindTcp(const Datacratic::PortRange& range) = 0;
    virtual void start();
    virtual void shutdown();

    void sendAugmentMessage(
            const std::shared_ptr<AugmentorInstanceInfo>& instance,
            const std::string& augmentorName,
            const std::shared_ptr<Auction>& auction,
            const std::set<std::string>& agents,
            Datacratic::Date date = Datacratic::Date::now());


    //
    // factory
    //
    static std::shared_ptr<AugmentorInterface>
    create(std::string serviceName,
           std::shared_ptr<ServiceProxies> const & proxies,
           Json::Value const & json);

    typedef std::function<AugmentorInterface *
                (std::string serviceName,
                std::shared_ptr<ServiceProxies> const & proxies,
                Json::Value const & json)> Factory;

    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer
    // so any use of it should be considered deprecated
    static void registerFactory(std::string const & name, Factory factory)
    {
      PluginInterface<AugmentorInterface>::registerPlugin(name, factory);
    }


    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "augmentor";};

protected:

    struct AugmentMessage{
        std::shared_ptr<AugmentorInstanceInfo> instance;
        std::string augmentorName;
        std::shared_ptr<Auction> auction;
        std::set<std::string> agents;
        Datacratic::Date date;
    };

    TypedMessageSink<AugmentMessage> inbox;

    virtual void doSendAugmentMessage(
                const std::shared_ptr<AugmentorInstanceInfo>& instance,
                const std::string& augmentorName,
                const std::shared_ptr<Auction>& auction,
                const std::set<std::string>& agents,
                Datacratic::Date date) = 0;
};


} // namespace RTBKIT
