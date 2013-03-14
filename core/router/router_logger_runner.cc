/* router_logger.cc
   Rémi Attab and Jeremy Barnes, March 2011
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Launches the router's logger.

   \todo Move to the example folder, annotate and simplify.
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
#include "soa/service/service_utils.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/lexical_cast.hpp>
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


/******************************************************************************/
/* ARGUMENTS                                                                  */
/******************************************************************************/

struct Arguments
{
    ServiceProxyArguments serviceArgs;
    vector<string> subscribeUris;
    string logDir;
    string rotationInterval;
};


/******************************************************************************/
/* LOGGER SETUP                                                               */
/******************************************************************************/

void subscribe(RouterLogger& logger, const Arguments& args)
{
    string myIdentity
        = args.serviceArgs.installation + "."
        + args.serviceArgs.nodeName + "."
        + "router_logger";

    // Subscribe to any sockets directly that include legacy information that
    // should be logged.
    for (auto u: args.subscribeUris) {
        cerr << "subscribing to fixed URI " << u << endl;
        logger.subscribe(u, vector<string>(), myIdentity);
    }

    // Subscribe to all messages
    logger.connectAllServiceProviders("adServer", "logger");
    logger.connectAllServiceProviders("rtbRequestRouter", "logger");
    logger.connectAllServiceProviders("rtbPostAuctionService", "logger");
}


std::shared_ptr<ConsoleStatsOutput>
setupOutputs(
        RouterLogger& logger,
        shared_ptr<ServiceProxies>& proxies,
        const Arguments& args)
{
    auto consoleOutput = make_shared<ConsoleStatsOutput>();
    logger.addOutput(consoleOutput);


    auto carbonOutput =
        make_shared<CarbonStatsOutput>(proxies->events, "router_logger");
    logger.addOutput(carbonOutput);


    // File output (appended) for normal logs
    auto normalOutput = make_shared<RotatingFileOutput>();
    normalOutput->open(
            args.logDir + "/%F/router-%F-%T.log.gz",
            args.rotationInterval,
            "gz");
    normalOutput->onFileWrite = [&](const string& channel, size_t bytes)
        {
            carbonOutput->recordBytesWrittenToFile("router", bytes);
        };
    logger.addOutput(
            normalOutput,
            boost::regex(".*"),
            boost::regex("AUCTION|BEHAVIOUR|CLICK|DATA|IMPRESSION|INTERACTION|PAERROR|ROUTERERROR|WIN|MATCHEDLOSS"));


    // File output (appended) for router error logs
    auto errorOutput = make_shared<RotatingFileOutput>();
    errorOutput->open(
            args.logDir + "/%F/errors-%F-%T.log", args.rotationInterval);
    errorOutput->onFileWrite = [&](const string& channel, size_t bytes)
        {
            carbonOutput->recordBytesWrittenToFile("error", bytes);
        };
    logger.addOutput(errorOutput, boost::regex("ROUTERERROR"), boost::regex());
    logger.addOutput(errorOutput, boost::regex("PAERROR"), boost::regex());


    auto writeDelivery = make_shared<RotatingFileOutput>();
    writeDelivery ->open(
            args.logDir + "/%F/delivery-%F-%T.log.gz",
            args.rotationInterval,
            "gz");
    writeDelivery->onFileWrite = [&](const string& channel, size_t bytes)
        {
            carbonOutput->recordBytesWrittenToFile("delivery", bytes);
        };
    logger.addOutput(
            writeDelivery,
            boost::regex("CLICK|DATA|IMPRESSION|INTERACTION"));


    // Strategy-level data
    auto strategyOutput = make_shared<MultiOutput>();

    auto createMatchedWinFile = [&] (const string & pattern)
        {
            auto result = make_shared<RotatingFileOutput>();
            result->open(pattern, args.rotationInterval);
            return result;
        };

    strategyOutput->logTo(
            "MATCHEDWIN",
            args.logDir + "/%F/$(17)/$(5)/$(0)-%T.log.gz",
            createMatchedWinFile);
    strategyOutput->logTo(
            "",
            args.logDir + "/%F/$(10)/$(11)/$(0)-%T.log.gz",
            createMatchedWinFile);

    logger.addOutput(
            strategyOutput,
            boost::regex("MATCHEDWIN|MATCHEDIMPRESSION|MATCHEDCLICK|MATCHEDVISIT"));


    // Behaviours
    auto behaviourOutput = make_shared<RotatingFileOutput>();
    behaviourOutput ->open(
            args.logDir + "/%F/behaviour-%F-%T.log.gz",
            args.rotationInterval,
            "gz");
    behaviourOutput->onFileWrite = [&](const string& channel, size_t bytes)
        {
            carbonOutput->recordBytesWrittenToFile("behaviour", bytes);
        };
    logger.addOutput(behaviourOutput, boost::regex("BEHAVIOUR"));

    return consoleOutput;
}


void parseArguments(int argc, char** argv, Arguments& args)
{
    // Default values.
    args.logDir = "router_logger";
    args.rotationInterval = "1h";

    using namespace boost::program_options;

    options_description loggerOptions("Logger Options");
    loggerOptions.add_options()
        ("subscribe-uri,s", value<vector<string> >(&args.subscribeUris),
                "URI to listen on for router events.")
        ("log-dir,d", value<string>(&args.logDir),
                "Directory where the folders should be stored.")
        ("rotation-interval,r", value<string>(&args.rotationInterval),
                "Interval between each log rotation.");

    options_description allOptions;
    allOptions.add(loggerOptions).add(args.serviceArgs.makeProgramOptions());
    allOptions.add_options() ("help,h", "Prints this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(allOptions).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << allOptions << endl;
        exit(0);
    }

    cerr << "Log Directory: "  << args.logDir << endl;
}


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main (int argc, char** argv)
{
    Arguments args;
    parseArguments(argc, argv, args);

    auto proxies = args.serviceArgs.makeServiceProxies();
    proxies->config->dump(cerr);

    RouterLogger logger(proxies);
    subscribe(logger, args);
    auto consoleOutput = setupOutputs(logger, proxies, args);
    logger.init(proxies->config);
    logger.start();

    EventRecorder events("", proxies);
    auto recordLevel = [&] (const string& name, double value)
        {
            events.recordLevel(value, name);
        };

    // Start periodic stats dump.
    ProcessStats lastStats;
    while (true) {
        ML::sleep(10.0);

        ProcessStats curStats;
        ProcessStats::logToCallback(recordLevel, lastStats, curStats, "process");
        lastStats = curStats;

        consoleOutput->dumpStats();
    }
}
