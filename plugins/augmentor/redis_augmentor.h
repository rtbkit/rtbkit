/*
 * redis_augmentor.h
 *
 *  Created on: Aug 7, 2013
 *      Author: Jan Sulmont
 */

#ifndef REDIS_AUGMENTOR_H_
#define REDIS_AUGMENTOR_H_

#include <string>
#include "augmentor_base.h"
#include "rtbkit/soa/service/redis.h"
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
                   const std::string& redisUri)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,proxies)
        , agent_config_ (proxies->zmqContext)
        , redis_ (Redis::Address(redisUri))
    {
    }

    RedisAugmentor(const std::string& augmentorName,
                   const std::string& serviceName,
                   ServiceBase& parent,
                   const std::string& redisUri)
        : RTBKIT::AsyncAugmentor(augmentorName,serviceName,parent)
        , agent_config_ (parent.getZmqContext())
        , redis_ (Redis::Address(redisUri))
    {
    }
    void init(int nthreads);
    virtual ~RedisAugmentor() ;
private:
    void onRequest(const AugmentationRequest & request, SendResponseCB sendResponse) override;
    RTBKIT::AgentConfigurationListener agent_config_;
    Redis::AsyncConnection redis_ ;
};

} /* namespace RTBKIT */
#endif /* REDIS_AUGMENTOR_H_ */
