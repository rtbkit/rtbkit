/* monitor_service_runner.cc
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Monitor service runner
*/

#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/make_shared.hpp>

#include "jml/arch/timers.h"
#include "soa/service/service_utils.h"
#include "monitor_endpoint.h"

using namespace boost::program_options;
using namespace std;

using namespace Datacratic;
using namespace RTBKIT;


int main(int argc, char ** argv)
{
    Datacratic::ServiceProxyArguments args;

    options_description options = args.makeProgramOptions("Monitor Service");
    options.add_options() ("help,h", "Print this message")
                          ("disabled,d", "Disable the monitor");

    variables_map vm;
    store(command_line_parser(argc, argv) .options(options) .run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << options << endl;
        exit(1);
    }

    auto proxies = args.makeServiceProxies();
    auto serviceName = args.serviceName("monitor");

    MonitorEndpoint monitor(proxies, serviceName);
    monitor.init({
                "rtbRequestRouter",
                "rtbPostAuctionService",
                "rtbBanker",
                "rtbDataLogger",
                "rtbAgentConfiguration"});

    auto addr = monitor.bindTcp();
    cerr << "monitor is listening on "
         << addr.first << "," << addr.second << endl;

    proxies->config->dump(cerr);

    if(vm.count("disabled")) {
        monitor.disabled = true;
        std::cerr << "MONITOR IS DISABLED" << std::endl;
    }

    monitor.startSync();

    return 0;
}
