/* monitor_service_runner.cc
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Monitor service runner
*/

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/make_shared.hpp>

#include <jml/arch/timers.h>

#include "monitor.h"

using namespace boost::program_options;
using namespace std;

using namespace Datacratic;
using namespace RTBKIT;


int main(int argc, char ** argv)
{
    options_description configuration_options("Configuration options");

    string zookeeperUri("localhost:2181");
    string installation;
    string nodeName;

    vector<string> carbonUris;  ///< TODO: zookeeper

    configuration_options.add_options()
        ("zookeeper-uri,Z", value(&zookeeperUri),
         "URI of zookeeper to use")
        ("installation,I", value(&installation),
         "Name of the installation that is running")
        ("node-name,N", value(&nodeName),
         "Name of the node we're running")
        ("carbon-connection,c", value<vector<string> >(&carbonUris),
         "URI of connection to carbon daemon");

    options_description all_opt("Monitor service");
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

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(zookeeperUri, installation);
    if (!carbonUris.empty())
        proxies->logToCarbon(carbonUris, installation + "." + nodeName);

    Monitor monitor(proxies, "monitor");
    monitor.init();
    auto addr = monitor.bindTcp();
    cerr << "monitor is listening on "
         << addr.first << "," << addr.second << endl;

    MonitorProviderProxy proxy(proxies->zmqContext, monitor);
    proxy.init(proxies->config,
               {"router", "postAuction", "masterBanker", "router_logger"});

    monitor.start();
    proxy.start();

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10);
    }
}
