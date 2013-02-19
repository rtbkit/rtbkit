/* agent_configuration_service_runner.cc
   Jeremy Banres, 18 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Class to run the configuration service.
*/

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>

#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
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

    std::vector<std::string> carbonUris;  ///< TODO: zookeeper

    configuration_options.add_options()
        ("zookeeper-uri,Z", value(&zookeeperUri),
         "URI of zookeeper to use")
        ("installation,I", value(&installation),
         "Name of the installation that is running")
        ("node-name,N", value(&nodeName),
         "Name of the node we're running")
        ("carbon-connection,c", value<vector<string> >(&carbonUris),
         "URI of connection to carbon daemon");

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

    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());
    proxies->useZookeeper(zookeeperUri, installation);
    if (!carbonUris.empty())
        proxies->logToCarbon(carbonUris, installation);

    AgentConfigurationService config(proxies);
    
    config.init();
    config.bindTcp();
    config.start();

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10.0);
    }
}


