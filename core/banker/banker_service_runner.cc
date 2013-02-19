/* banker_service_runner.cc                                        -*- C++ -*-
   Jeremy Barnes, 20 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Runner for the banker service.
*/


#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>

#include "rtbkit/core/banker/master_banker.h"
#include "jml/utils/pair_utils.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


int main(int argc, char ** argv)
{
    using namespace boost::program_options;

    options_description configuration_options("Configuration options");

    std::string zookeeperUri;
    std::string installation;
    std::string nodeName;
    std::string redisUri;  ///< TODO: zookeeper

    std::vector<std::string> carbonUris;  ///< TODO: zookeeper
    std::vector<std::string> fixedHttpBindAddresses;

    configuration_options.add_options()
        ("zookeeper-uri,Z", value(&zookeeperUri),
         "URI of zookeeper to use")
        ("installation,I", value(&installation),
         "Name of the installation that is running")
        ("node-name,N", value(&nodeName),
         "Name of the node we're running")
        ("carbon-connection,c", value<vector<string> >(&carbonUris),
         "URI of connection to carbon daemon")
        ("redis-uri,r", value<string>(&redisUri),
         "URI of connection to redis")
        ("fixed-http-bind-address,a", value(&fixedHttpBindAddresses),
         "Fixed address (host:port or *:port) at which we will always listen");

    options_description all_opt;
    all_opt
        .add(configuration_options);
    all_opt.add_options()
        ("help,h", "print this message");
    
    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          //.positional(p)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        exit(1);
    }

    if (installation.empty()) {
        cerr << "'installation' parameter is required" << endl;
        exit(1);
    }

    if (nodeName.empty()) {
        cerr << "'node-name' parameter is required" << endl;
        exit(1);
    }

    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());
    proxies->useZookeeper(zookeeperUri, installation);
    if (!carbonUris.empty())
        proxies->logToCarbon(carbonUris, installation + "." + nodeName);

    MasterBanker banker(proxies, "masterBanker");
    std::shared_ptr<Redis::AsyncConnection> redis;


    if (redisUri != "nopersistence") {
        auto address = Redis::Address(redisUri);
        redis = std::make_shared<Redis::AsyncConnection>(redisUri);
        redis->test();

        banker.init(std::make_shared<RedisBankerPersistence>(redisUri));
    }
    else {
        cerr << "*** WARNING ***" << endl;
        cerr << "BANKER IS RUNNING WITH NO PERSISTENCE" << endl;
        cerr << "IF THIS IS NOT A TESTING ENVIRONEMNT, SPEND WILL BE LOST" << endl;
        banker.init(std::make_shared<NoBankerPersistence>());
    }

    // Bind to any fixed addresses
    for (auto a: fixedHttpBindAddresses)
        banker.bindFixedHttpAddress(a);

    auto addr = banker.bindTcp();
    cerr << "master banker is listening on " << addr.first << ","
         << addr.second << endl;

    banker.start();
    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10);
    }
}
