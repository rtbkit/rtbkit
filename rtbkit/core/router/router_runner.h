/** router_runner.h                                                -*- C++ -*-
    Jeremy Barnes, 17 December 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Program to run the router.
*/

#pragma once


#include <boost/program_options/options_description.hpp>
#include "rtbkit/core/router/router.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/core/banker/local_banker.h"
#include "soa/service/service_utils.h"
#include "soa/service/logs.h"

namespace RTBKIT {


/*****************************************************************************/
/* ROUTER RUNNER                                                             */
/*****************************************************************************/

struct RouterRunner {

    RouterRunner();

    ServiceProxyArguments serviceArgs;
    SlaveBankerArguments bankerArgs;
    std::vector<std::string> logUris;  ///< TODO: zookeeper

    std::string exchangeConfigurationFile;
    std::string bidderConfigurationFile;

    float lossSeconds;
    bool noPostAuctionLoop;
    bool noBidProb;

    bool logAuctions;
    bool logBids;

    float maxBidPrice;
    std::string bankerUri;
    std::string localBankerUri;
    bool localBankerDebug;
    std::string bankerChoice;

    int slowModeTimeout; // Default value =  MonitorClient::DefaultCheckTimeout
    int slowModeTolerance;

    std::string slowModeMoneyLimit;
    bool analyticsOn;
    int analyticsConnections;
    int augmentationWindowms;
    bool dableSlowMode;
    std::string enableJsonFiltersFile;

    void doOptions(int argc, char ** argv,
                   const boost::program_options::options_description & opts
                   = boost::program_options::options_description());

    std::shared_ptr<ServiceProxies> proxies;
    std::shared_ptr<Banker> banker;
    std::shared_ptr<SlaveBanker> slaveBanker;
    std::shared_ptr<LocalBanker> localBanker;
    std::shared_ptr<Router> router;
    Json::Value exchangeConfig;
    Json::Value bidderConfig;
    Json::Value filtersConfig;

    static Logging::Category print;
    static Logging::Category trace;
    static Logging::Category error;

    void init();

    void start();

    void shutdown();

};

} // namespace RTBKIT

