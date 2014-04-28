/* post_auction_runner.cc
   JS Bejeau , 13 February 2014

   Copyright (c) 2013 Datacratic Inc.  All rights reserved.
*/

#include "post_auction_runner.h"

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "rtbkit/core/banker/slave_banker.h"
#include "soa/service/service_utils.h"

#include "post_auction_service.h"

using namespace std;
using namespace boost::program_options;
using namespace Datacratic;
using namespace RTBKIT;

/************************************************************************/
/* POST AUCTION LOOP RUNNER                                             */
/************************************************************************/
PostAuctionRunner::
PostAuctionRunner() :
    shards(1),
    auctionTimeout(900.0),
    winTimeout(3600.0)
{
}

void 
PostAuctionRunner::
doOptions(int argc, char ** argv,
        const boost::program_options::options_description & opts)
{
    using namespace boost::program_options;

    options_description postAuctionLoop_options("Post Auction Loop options");
    postAuctionLoop_options.add_options()
        ("shards", value<size_t>(&shards),"Number of shards(threads) used for matching.")
        ("win-seconds", value<float>(&winTimeout),"Timeout for storing win auction")
        ("auction-seconds", value<float>(&auctionTimeout),"Timeout to get late win auction");

    options_description all_opt = opts;
    all_opt
        .add(serviceArgs.makeProgramOptions())
        .add(postAuctionLoop_options);

    all_opt.add_options()
        ("help,h","print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
            .options(all_opt)
            .run(),
            vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        exit(1);
    }
}

void
PostAuctionRunner::
init()
{
    auto proxies = serviceArgs.makeServiceProxies();
    auto serviceName = serviceArgs.serviceName("PostAuctionLoop");

    postAuctionLoop = std::make_shared<PostAuctionService>(proxies, serviceName);
    postAuctionLoop->init(shards);

    postAuctionLoop->setWinTimeout(winTimeout);
    postAuctionLoop->setAuctionTimeout(auctionTimeout);


    banker = std::make_shared<SlaveBanker>(proxies->zmqContext,
            proxies->config,
            postAuctionLoop->serviceName() + ".slaveBanker");

    postAuctionLoop->addSource("slave-banker", *banker);
    postAuctionLoop->setBanker(banker);
    postAuctionLoop->bindTcp();

}

void
PostAuctionRunner::
start()
{
    postAuctionLoop->start();
}

void
PostAuctionRunner::
shutdown()
{
    postAuctionLoop->shutdown();
    banker->shutdown();
}



int main(int argc, char ** argv)
{

    PostAuctionRunner runner;

    runner.doOptions(argc, argv);
    runner.init();
    runner.start();


    for (;;) {
        ML::sleep(10.0);
    }

}
