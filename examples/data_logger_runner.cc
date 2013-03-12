/* data_logger_runner.cc
   Sunil Rottoo March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

*/

#include "rtbkit/plugins/data_logger/data_logger.h"
#include "soa/logger/file_output.h"
#include "soa/logger/stats_output.h"
#include "soa/logger/multi_output.h"
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;
using namespace boost::program_options;


int main (int argc, char** argv)
{
    string zookeeperURI = "";
    vector<string> subscribeUris ;
    vector<string> serviceClasses = {"adServer", "rtbRequestRouter", "rtbPostAuctionService"};
    string installation="";
    string rotationInterval="1h";
    string logdir="logdir";
    options_description all_opt;

    all_opt.add_options()
          ("zookeeperuri,z", value(&zookeeperURI),
           "Zookeeper URI for the Zookeeper Server where services register")
           ("installation,i", value(&installation),
            "installation name")
           ("logdir,l", value(&logdir),
              "log directory")
           ("directuris,d", value(&subscribeUris),
            "uris to subscriber to directly")
           ("rotation-interval,r", value(&rotationInterval),
             "interval at which we should rotate the logs")
         ("help,h", "print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        return 1;
    }

    string myIdentity
        = installation + "."
        + "data_logger";

    cerr << "Log Directory: "  << logdir << endl;
    cerr << "ZooKeeper URI: " << zookeeperURI << endl;

    DataLogger logger(zookeeperURI, installation);
    // Subscribe to any sockets directly
    for (auto u: subscribeUris) {
        cerr << "subscribing to fixed URI " << u << endl;
        logger.subscribe(u, vector<string>(), myIdentity);
    }
    logger.connectToAllServices(serviceClasses);
    // Setup outputs

    // Log to console
    auto consoleOutput = std::make_shared<ConsoleStatsOutput>();
    logger.addOutput(consoleOutput);

    // File output (appended) for normal logs
    std::shared_ptr<RotatingFileOutput> normalOutput
        (new RotatingFileOutput());
    normalOutput->open(logdir + "/%F/router-%F-%T.log.gz", rotationInterval, "gz");

     logger.addOutput(normalOutput,
                     boost::regex(".*"),
                     boost::regex("AUCTION|BEHAVIOUR|CLICK|DATA|IMPRESSION|INTERACTION|PAERROR|ROUTERERROR|WIN|MATCHEDLOSS"));

    // File output (appended) for router error logs
    std::shared_ptr<RotatingFileOutput> errorOutput
        (new RotatingFileOutput());
    errorOutput->open(logdir + "/%F/errors-%F-%T.log", rotationInterval);
    errorOutput->onFileWrite = [&](const string& channel, size_t bytes) {
    };
    logger.addOutput(errorOutput, boost::regex("ROUTERERROR"), boost::regex());
    logger.addOutput(errorOutput, boost::regex("PAERROR"), boost::regex());

    std::shared_ptr<RotatingFileOutput> writeDelivery
        (new RotatingFileOutput());
    writeDelivery
        ->open(logdir + "/%F/delivery-%F-%T.log.gz", rotationInterval, "gz");
    writeDelivery->onFileWrite = [&](const string& channel, size_t bytes) {
    };
    logger.addOutput(writeDelivery,
                     boost::regex("CLICK|DATA|IMPRESSION|INTERACTION"));

    // Strategy-level data
    auto strategyOutput = std::make_shared<MultiOutput>();

    auto createMatchedWinFile = [&] (const std::string & pattern)
        {
            auto result = std::make_shared<RotatingFileOutput>();
            result->open(pattern, rotationInterval);
            return result;
        };

    strategyOutput->logTo("MATCHEDWIN", logdir + "/%F/$(17)/$(5)/$(0)-%T.log.gz",
                          createMatchedWinFile);
    strategyOutput->logTo("", logdir + "/%F/$(10)/$(11)/$(0)-%T.log.gz",
                          createMatchedWinFile);

    logger.addOutput(strategyOutput, boost::regex("MATCHEDWIN|MATCHEDIMPRESSION|MATCHEDCLICK|MATCHEDVISIT"));

    // Behaviours
    std::shared_ptr<RotatingFileOutput> behaviourOutput
        (new RotatingFileOutput());
    behaviourOutput
        ->open(logdir + "/%F/behaviour-%F-%T.log.gz", rotationInterval, "gz");
    behaviourOutput->onFileWrite = [&](const string& channel, size_t bytes) {
    };
    logger.addOutput(behaviourOutput, boost::regex("BEHAVIOUR"));

    logger.init();
    logger.start();
    while (true) {
         ML::sleep(10.0);
         consoleOutput->dumpStats();
    }
}
