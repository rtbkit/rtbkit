/* post_auction_runner.cc
   Wolfgang Sourdeau, March 2013

   Copyright (c) 2013 Datacratic Inc.  All rights reserved.
*/

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "rtbkit/core/banker/slave_banker.h"
#include "soa/service/service_utils.h"

#include "post_auction_loop.h"

using namespace std;
using namespace boost::program_options;
using namespace Datacratic;
using namespace RTBKIT;


int main(int argc, char ** argv)
{
    ServiceProxyArguments proxyArgs;

    options_description all_opt;
    all_opt.add(proxyArgs.makeProgramOptions());
    all_opt.add_options()
        ("help,h", "print this message");
    
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

    shared_ptr<ServiceProxies> proxies = proxyArgs.makeServiceProxies();

    // First start up the post auction loop
    PostAuctionLoop service(proxies, "postAuction");

    auto banker = make_shared<SlaveBanker>(proxies->zmqContext,
                                           proxies->config,
                                           service.serviceName()
                                           + ".slaveBanker");
    banker->start();

    service.init();
    service.setBanker(banker);
    service.bindTcp();
    service.start();

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10.0);
    }
}
