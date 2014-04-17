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

#include "service_base.h"


using namespace std;
using namespace Datacratic;


int main(int argc, char ** argv)
{
    using namespace boost::program_options;

    options_description configuration_options("Configuration options");

    string zookeeperUri("localhost:2181");
    string installation;

    vector<string> carbonUris;  ///< TODO: zookeeper
    vector<string> fixedHttpBindAddresses;

    configuration_options.add_options()
        ("zookeeper-uri,Z", value(&zookeeperUri),
         "URI of zookeeper to use (localhost:2181)")
        ("installation,I", value(&installation),
         "Name of the installation that is running");

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

    shared_ptr<ServiceProxies> proxies(new ServiceProxies());
    proxies->useZookeeper(zookeeperUri, installation);
    proxies->config->dump(cerr);
}
