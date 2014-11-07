
#include <string>
#include "analytics.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace Datacratic;

int main(int argc, char ** argv) {
    
    using namespace boost::program_options;
    
    ServiceProxyArguments serviceArgs;
    
    bool debug = false;
    bool logWins = false;
    bool logUnmatchedWins = false;
    bool logBids = false;

    options_description configuration_options("Configuration options");
    configuration_options.add_options()
        ("log-wins,W", bool_switch(&logWins),
         "Whether or not to log wins.")
        ("log-unmatched-wins,U", bool_switch(&logUnmatchedWins),
         "Whether or not to log unmatched wins.")
        ("log-bids,A", bool_switch(&logBids),
         "Wether or not to log bids, bid logging must also be enabled in the router")
        ("debug", bool_switch(&debug),
         "Debug mode enabled");

    options_description all_opt;
    all_opt
        .add(serviceArgs.makeProgramOptions("General Options"))
        .add(configuration_options);
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

    auto proxies = serviceArgs.makeServiceProxies(CS_INTERNAL);
    auto serviceName = serviceArgs.serviceName("analytics");

    AnalyticsRestEndpoint analytics(proxies, serviceName);
    analytics.init();

    auto addr = analytics.bindTcp();
    cerr << "analytics is listening on " << addr.first << ","
        << addr.second << endl;

    analytics.start();
    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10);
    }

}
