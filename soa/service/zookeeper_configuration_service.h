/** zookeeper_configuration_service.h                              -*- C++ -*-
    Jeremy Barnes, 26 September 2012
    Copyright (c) 2012 Datacratic Inc.  All rights reserved.

    Configuration service built on top of Zookeeper.
*/

#ifndef __service__zookeeper_configuration_service_h__
#define __service__zookeeper_configuration_service_h__


#include "service_base.h"
#include <memory>


namespace Datacratic {

struct ZookeeperConnection;

/*****************************************************************************/
/* ZOOKEEPER CONFIGURATION SERVICE                                           */
/*****************************************************************************/

/** Configuration service built on top of Zookeeper. */

struct ZookeeperConfigurationService
    : public ConfigurationService {

    friend class ServiceDiscoveryScenario;    


    ZookeeperConfigurationService();

    ZookeeperConfigurationService(std::string host,
                                  std::string prefix,
                                  std::string location,
                                  int timeout = 5);
    
    ~ZookeeperConfigurationService();

    void init(std::string host,
              std::string prefix,
              std::string location,
              int timeout = 5);

    virtual Json::Value getJson(const std::string & value,
                                Watch watch = Watch());
    
    virtual void set(const std::string & key,
                     const Json::Value & value);

    virtual std::string setUnique(const std::string & key,
                                  const Json::Value & value);

    virtual std::vector<std::string>
    getChildren(const std::string & key,
                Watch watch = Watch());

    virtual bool forEachEntry(const OnEntry & onEntry,
                              const std::string & startPrefix = "") const;

    /** Recursively remove everything below this path. */
    virtual void removePath(const std::string & path);

private:
    std::unique_ptr<ZookeeperConnection> zoo;
    std::string prefix;
};


} // namespace Datacratic


#endif /* __service__zookeeper_configuration_service_h__ */
