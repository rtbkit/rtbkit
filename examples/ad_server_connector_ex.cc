/** ad_server_connector_ex.cc                                 -*- C++ -*-
    Eric Robert, 03 April 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Example of a simple ad server connector.

*/

#include "mock_ad_server_connector.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace Datacratic;
using namespace ML;

/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char** argv)
{
    using namespace boost::program_options;

    int winPort = 12340;

    ServiceProxyArguments args;
    options_description options = args.makeProgramOptions();
    options_description more("Mock Ad Server Connector");
    more.add_options()
        ("win-port,w", value(&winPort), "listening port for wins");

    options.add(more);
    options.add_options() ("help,h", "Print this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(options).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << options << endl;
        return 1;
    }

    RTBKIT::MockAdServerConnector server(args, "mock-ad-server-connector");
    server.init(winPort);
    server.start();

    while (true) this_thread::sleep_for(chrono::seconds(10));

    // Won't ever reach this point...
    return 0;
}

