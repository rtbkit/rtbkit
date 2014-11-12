
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
    
    bool enableAllChannels = false;

    ChannelFilter channels(
                { {"AUCTION",       false},
                  {"CLICK",         false},
                  {"CONFIG",        false},
                  {"MARK",          false},
                  {"MATCHEDCLICK",  false},
                  {"MATCHEDLOSS",   false},
                  {"MATCHEDWIN",    false},
                  {"NOBUDGET",      false},
                  {"PAERROR",       false},
                  {"TOOLATE",       false},
                  {"UNMATCHEDWIN",  false},
                  {"UNMATCHEDLOSS", false},
                  {"UNMATCHEDCLICK",false},
                  {"USAGE",         false},
                  {"WIN",           false}
                } );

    options_description configuration_options("Configuration options");
    configuration_options.add_options()
        ("ALL", bool_switch(&enableAllChannels),
         "enable all channels.")
        ("WIN", bool_switch(&channels["WIN"]),
         "log wins channel.")
        ("UNMATCHEDWIN", bool_switch(&channels["UNMATCHEDWIN"]),
         "log unmatched wins.")
        ("UNMATCHEDLOSS", bool_switch(&channels["UNMATCHEDLOSS"]),
         "log unmatched losses.")
        ("UNMATCHEDCLICK", bool_switch(&channels["UNMATCHEDCLICK"]),
         "log unmatched losses.")
        ("CLICK", bool_switch(&channels["CLICK"]),
         "log clicks.")
        ("CONFIG", bool_switch(&channels["CONFIG"]),
         "log config.")
        ("MARK", bool_switch(&channels["MARK"]),
         "log marks.")
        ("MATCHEDCLICK", bool_switch(&channels["MATCHEDCLICK"]),
         "log matched clicks.")
        ("MATCHEDLOSS", bool_switch(&channels["MATCHEDLOSS"]),
         "log matched loss.")
        ("MATCHEDWIN", bool_switch(&channels["MATCHEDWIN"]),
         "log matched win.")
        ("NOBUDGET", bool_switch(&channels["NOBUDGET"]),
         "log no budget.")
        ("PAERROR", bool_switch(&channels["PAERROR"]),
         "log post auction error.")
        ("TOOLATE", bool_switch(&channels["TOOLATE"]),
         "log too late.")
        ("USAGE", bool_switch(&channels["USAGE"]),
         "log usage.")
        ("AUCTIONS", bool_switch(&channels["AUCTIONS"]),
         "log actions.");

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

    analytics.initChannels(channels);

    if (enableAllChannels)
        analytics.enableAllChannels();
    
    auto addr = analytics.bindTcp();
    cerr << "analytics is listening on " << addr.first << ","
        << addr.second << endl;

    analytics.start();
    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10);
    }

}
