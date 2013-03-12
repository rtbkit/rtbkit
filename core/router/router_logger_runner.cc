/* router_logger.cc
   Rémi Attab and Jeremy Barnes, March 2011
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Launches the router's logger.
*/


#include "soa/service/service_base.h"
#include "soa/service/carbon_connector.h"
#include "soa/service/process_stats.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/logger/file_output.h"
#include "soa/logger/stats_output.h"
#include "soa/logger/multi_output.h"
#include "rtbkit/common/auction.h"
#include "jml/arch/timers.h"
#include "rtbkit/core/monitor/monitor_provider.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/regex.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <thread>
#include <algorithm>

#include "router_logger.h"


using namespace std;
using namespace Datacratic;
using namespace RTBKIT;


bool getArgs (int argc, char** argv);

static struct {
    vector<string> subscribeUris;
    string logDir;
    vector<string> carbonUris;
    string zookeeperUri;
    string installation;
    string nodeName;
    string rotationInterval;
} g_args;


int main (int argc, char** argv)
{
    if (!getArgs(argc, argv)) {
        return -1;
    }

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(g_args.zookeeperUri, g_args.installation);
    if (!g_args.carbonUris.empty())
        proxies->logToCarbon(g_args.carbonUris,
                             g_args.installation + "." + g_args.nodeName);

    proxies->config->dump(cerr);
    string rotationInterval = g_args.rotationInterval;

    cerr << "Log Directory: "  << g_args.logDir << endl;

    RouterLogger logger(proxies);

    string myIdentity
        = g_args.installation + "." 
        + g_args.nodeName + "."
        + "router_logger";
    
    // Subscribe to any sockets directly that include legacy information that
    // should be logged.
    for (auto u: g_args.subscribeUris) {
        cerr << "subscribing to fixed URI " << u << endl;
        logger.subscribe(u, vector<string>(), myIdentity);
    }
    
    // Subscribe to all messages
    logger.connectAllServiceProviders("adServer", "logger");
    logger.connectAllServiceProviders("rtbRequestRouter", "logger");
    logger.connectAllServiceProviders("rtbPostAuctionService", "logger");

    // Setup outputs

    auto consoleOutput = std::make_shared<ConsoleStatsOutput>();
    logger.addOutput(consoleOutput);

    std::shared_ptr<CarbonStatsOutput> carbonOutput
        (new CarbonStatsOutput(proxies->events, "router_logger"));
    logger.addOutput(carbonOutput);

    // File output (appended) for normal logs
    std::shared_ptr<RotatingFileOutput> normalOutput
        (new RotatingFileOutput());
    normalOutput->open(g_args.logDir + "/%F/router-%F-%T.log.gz", rotationInterval, "gz");
    normalOutput->onFileWrite = [&](const string& channel, size_t bytes) {
        carbonOutput->recordBytesWrittenToFile("router", bytes);
    };
    logger.addOutput(normalOutput,
                     boost::regex(".*"),
                     boost::regex("AUCTION|BEHAVIOUR|CLICK|DATA|IMPRESSION|INTERACTION|PAERROR|ROUTERERROR|WIN|MATCHEDLOSS"));

    // File output (appended) for router error logs
    std::shared_ptr<RotatingFileOutput> errorOutput
        (new RotatingFileOutput());
    errorOutput->open(g_args.logDir + "/%F/errors-%F-%T.log", rotationInterval);
    errorOutput->onFileWrite = [&](const string& channel, size_t bytes) {
        carbonOutput->recordBytesWrittenToFile("error", bytes);
    };
    logger.addOutput(errorOutput, boost::regex("ROUTERERROR"), boost::regex());
    logger.addOutput(errorOutput, boost::regex("PAERROR"), boost::regex());

    std::shared_ptr<RotatingFileOutput> writeDelivery
        (new RotatingFileOutput());
    writeDelivery
        ->open(g_args.logDir + "/%F/delivery-%F-%T.log.gz", rotationInterval, "gz");
    writeDelivery->onFileWrite = [&](const string& channel, size_t bytes) {
        carbonOutput->recordBytesWrittenToFile("delivery", bytes);
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

    strategyOutput->logTo("MATCHEDWIN", g_args.logDir + "/%F/$(17)/$(5)/$(0)-%T.log.gz",
                          createMatchedWinFile);
    strategyOutput->logTo("", g_args.logDir + "/%F/$(10)/$(11)/$(0)-%T.log.gz",
                          createMatchedWinFile);

    logger.addOutput(strategyOutput, boost::regex("MATCHEDWIN|MATCHEDIMPRESSION|MATCHEDCLICK|MATCHEDVISIT"));

    // Behaviours
    std::shared_ptr<RotatingFileOutput> behaviourOutput
        (new RotatingFileOutput());
    behaviourOutput
        ->open(g_args.logDir + "/%F/behaviour-%F-%T.log.gz", rotationInterval, "gz");
    behaviourOutput->onFileWrite = [&](const string& channel, size_t bytes) {
        carbonOutput->recordBytesWrittenToFile("behaviour", bytes);
    };
    logger.addOutput(behaviourOutput, boost::regex("BEHAVIOUR"));

    logger.init(proxies->config);
    logger.start();

    // Start periodic stats dump.
    ProcessStats lastStats;
    while (true) {
        ML::sleep(10.0);

        ProcessStats curStats;
        ProcessStats::logToCallback(
                [&](string name, double value) {
                    carbonOutput->recordLevel(name, value); },
                lastStats, curStats, "process");
        lastStats = curStats;

        consoleOutput->dumpStats();
    }
}


bool getArgs (int argc, char** argv)
{
    // Default values.
    g_args.logDir = "router_logger";
    g_args.rotationInterval = "1h";

    using namespace boost::program_options;

    options_description loggerOptions("Logger Options");
    loggerOptions.add_options()
        ("subscribe-uri,s", value<vector<string> >(&g_args.subscribeUris),
                "URI to listen on for events (should be a zmq PUB socket).")
        ("log-dir,d", value<string>(&g_args.logDir),
                "Directory where the folders should be stored.");


    options_description carbonOptions("Carbon Options");
    carbonOptions.add_options()
        ("carbon-connection,c", value<vector<string> >(&g_args.carbonUris),
         "URI of connection to carbon daemon (format: host:port)");

    options_description allOptions;
    allOptions.add_options()
        ("zookeeper-uri,Z", value(&g_args.zookeeperUri),
         "URI of zookeeper to use")
        ("installation,I", value(&g_args.installation),
         "Name of the installation that is running")
        ("node-name,N", value(&g_args.nodeName),
         "Name of the node we're running");
    allOptions.add(loggerOptions).add(carbonOptions);
    allOptions.add_options() ("help,h", "Prints this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(allOptions).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << allOptions << endl;
        return false;
    }

    if (g_args.installation.empty()) {
        cerr << "'installation' parameter is required" << endl;
        return false;
    }

    if (g_args.nodeName.empty()) {
        cerr << "'node-name' parameter is required" << endl;
        return false;
    }

    return true;
}
