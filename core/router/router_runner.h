/** router_runner.h                                                -*- C++ -*-
    Jeremy Barnes, 17 December 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Program to run the router.
*/

#pragma once


#include <boost/program_options/options_description.hpp>
#include "rtbkit/core/router/router.h"
#include "rtbkit/core/banker/slave_banker.h"

namespace RTBKIT {


/*****************************************************************************/
/* ROUTER RUNNER                                                             */
/*****************************************************************************/

struct RouterRunner {

    RouterRunner();

    std::string zookeeperUri;
    std::string installation;
    std::string nodeName;

    std::vector<std::string> carbonUris;  ///< TODO: zookeeper
    std::vector<std::string> logUris;  ///< TODO: zookeeper

    //std::string routerConfigurationFile;
    std::string exchangeConfigurationFile;
    float lossSeconds;

    void doOptions(int argc, char ** argv,
                   const boost::program_options::options_description & opts
                   = boost::program_options::options_description());

    std::shared_ptr<ServiceProxies> proxies;
    std::shared_ptr<SlaveBanker> banker;
    std::shared_ptr<Router> router;
    Json::Value exchangeConfig;

    void init();

    void start();

    void shutdown();

};

} // namespace RTBKIT

