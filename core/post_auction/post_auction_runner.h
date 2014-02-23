/** post_auction_runner.h                                                -*- C++ -*-
    JS Bejeau, 14 February 2014
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Program to run the post_auction_loop.
*/

#pragma once


#include <boost/program_options/options_description.hpp>
#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "soa/service/service_utils.h"

namespace RTBKIT {


/*****************************************************************************/
/* ROUTER RUNNER                                                             */
/*****************************************************************************/

struct PostAuctionRunner {

    PostAuctionRunner();     

    ServiceProxyArguments serviceArgs;

    float auctionTimeout;
    float winTimeout;


    void doOptions(int argc, char ** argv,
                   const boost::program_options::options_description & opts
                   = boost::program_options::options_description());

    std::shared_ptr<ServiceProxies> proxies;
    std::shared_ptr<SlaveBanker> banker;
    std::shared_ptr<PostAuctionLoop> postAuctionLoop;

    void init();

    void start();

    void shutdown();

};

} // namespace RTBKIT

