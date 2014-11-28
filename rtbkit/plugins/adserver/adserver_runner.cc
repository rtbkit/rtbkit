/* adserver_runner.cc
   Eric Robert, 20 Aug 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Tool to run the ad server
*/

#include "adserver_connector.h"
#include "soa/service/service_utils.h"
#include "jml/utils/file_functions.h"

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

static inline Json::Value loadJsonFromFile(const std::string & filename)
{
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}

/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char** argv)
{
    using namespace boost::program_options;

    std::string configuration = "rtbkit/examples/adserver-config.json";

    ServiceProxyArguments args;
    options_description options = args.makeProgramOptions();
    options_description more("Ad Server");
    more.add_options()
        ("adserver-configuration,f", value(&configuration), "configuration file");

    options.add(more);
    options.add_options() ("help,h", "Print this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(options).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << options << endl;
        return 1;
    }

    auto proxies = args.makeServiceProxies();
    auto serviceName = args.serviceName("");
    auto server = RTBKIT::AdServerConnector::create(serviceName, proxies, loadJsonFromFile(configuration));
    server->start();

    while (true) this_thread::sleep_for(chrono::seconds(10));

    // Won't ever reach this point...
    return 0;
}

