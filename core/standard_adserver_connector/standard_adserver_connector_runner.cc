/* standard_adserver_connector_runner.cc
   Wolfgang Sourdeau, May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Program listening for adserver events, relaying them to the post
   auction loop and logging them on disk.
 */


#include <iostream>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "standard_adserver_connector.h"


using namespace std;

using namespace boost::program_options;
using namespace RTBKIT;

int main(int argc, char * argv[])
{
    StandardAdServerArguments config;
    config.winPort = 9001;
    config.eventsPort = 9002;
    config.externalWinPort = 9003;

    options_description all_opt;
    all_opt.add(config.makeProgramOptions());
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

    auto proxies = config.makeServiceProxies();

    StandardAdServerConnector connector(proxies);
    connector.init(config);
    connector.start();

    proxies->config->dump(cerr);

    for (;;) {
        sleep(10);
        connector.recordUptime();
    }
}
