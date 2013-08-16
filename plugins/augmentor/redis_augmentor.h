/* -*- C++ -*-
 * redis_augmentor.h
 *
 *  Created on: Aug 7, 2013
 *      Author: Jan Sulmont
 */

#ifndef REDIS_AUGMENTOR_H_
#define REDIS_AUGMENTOR_H_

#include <string>
#include "augmentor_base.h"
#include "soa/service/redis.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"

namespace RTBKIT {

/**
 *     Redis Augmentor.
 */
class RedisAugmentor: public RTBKIT::AsyncAugmentor {
public:
    RedisAugmentor(const std::string& augmentorName,
                   const std::string& serviceName,
                   std::shared_ptr<ServiceProxies> proxies,
                   const Redis::Address& redis)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,proxies)
        , agent_config_ (proxies->zmqContext)
        , redis_(std::make_shared<Redis::AsyncConnection>(redis))
    {
    }

    RedisAugmentor(const std::string& augmentorName,
                   const std::string& serviceName,
                   std::shared_ptr<ServiceProxies> proxies,
                   std::shared_ptr<Redis::AsyncConnection> redis)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,proxies)
        , agent_config_ (proxies->zmqContext)
        , redis_(redis)
    {
    }

    RedisAugmentor(const std::string& augmentorName,
                   const std::string& serviceName,
                   ServiceBase& parent,
                   const Redis::Address& redis)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,parent)
        , agent_config_ (parent.getZmqContext())
        , redis_(std::make_shared<Redis::AsyncConnection>(redis))
    {
    }

    RedisAugmentor(const std::string& augmentorName,
                   const std::string& serviceName,
                   ServiceBase& parent,
                   std::shared_ptr<Redis::AsyncConnection> redis)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,parent)
        , agent_config_ (parent.getZmqContext())
        , redis_ (redis)
    {
    }

    void init(int nthreads);
    virtual ~RedisAugmentor() ;
private:
    void onRequest(const AugmentationRequest & request, SendResponseCB sendResponse);
    RTBKIT::AgentConfigurationListener agent_config_;
    std::shared_ptr<Redis::AsyncConnection> redis_ ;
};

} /* namespace RTBKIT */
#endif /* REDIS_AUGMENTOR_H_ */
