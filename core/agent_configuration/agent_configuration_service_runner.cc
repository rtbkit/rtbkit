/* agent_configuration_service_runner.cc
   Jeremy Barnes, 18 December 2012
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
#include "soa/service/service_utils.h"
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

    Datacratic::ServiceProxyArguments args;

    options_description options = args.makeProgramOptions();
    options.add_options() ("help,h", "Print this message");
    
    variables_map vm;
    store(command_line_parser(argc, argv).options(options) .run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << options << endl;
        exit(1);
    }

    auto proxies = args.makeServiceProxies();
    auto serviceName = args.serviceName("agentConfigurationService");

    AgentConfigurationService config(proxies, serviceName);
    config.init();
    config.bindTcp();
    config.start();

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10.0);
    }
}


