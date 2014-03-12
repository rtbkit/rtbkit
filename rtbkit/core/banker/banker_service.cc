/* banker_service_runner.cc                                        -*- C++ -*-
   Jeremy Barnes, 20 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Runner for the banker service.
*/

#include "master_banker.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include "jml/arch/timers.h"


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


int main(int argc, char ** argv)
{
    using namespace boost::program_options;

    options_description service_options("Service options");

#if 0

    service_options.add_options()
        ("num-threads,t", value<int>(&numThreads),
         "number of threads to start up");

    options_description logging_options("Logging options");

    logging_options.add_options()
        ("log-uri", value<vector<string> >(&logUris),
         "URI to publish logs to")
        ("carbon-connection,c", value<string>(&carbonConnection),
         "URI of connection to carbon daemon");
#endif

    //positional_options_description p;
    //p.add("dataset", -1);

    options_description all_opt;
    all_opt
        .add(service_options);
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
        return 1;
    }

    auto persistence = std::make_shared<RedisBankerPersistence>();
    auto proxies = std::make_shared<ServiceProxies>();

    string serviceName = "masterBanker";

    MasterBanker master(persistence, proxies, serviceName);
    auto addr = master.bindTcp();
    cerr << "master banker is listening on " << addr.first << ","
         << addr.second << endl;

    master.start();

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(10);
    }
}
