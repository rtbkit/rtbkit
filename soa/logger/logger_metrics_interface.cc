/* logger_metrics_interface.cc
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include "soa/jsoncpp/reader.h"
#include "jml/utils/exc_assert.h"
#include "soa/logger/logger_metrics_interface.h"
#include "soa/logger/logger_metrics_void.h"
#include "soa/logger/logger_metrics_term.h"

using namespace std;
using namespace Datacratic;

namespace {

std::mutex m;
std::shared_ptr<ILoggerMetrics> logger;

// getenv that sanely deals with empty strings
string getEnv(const char * variable)
{
    const char * c = getenv(variable);
    return c ? c : "";
}

} // file scope


/****************************************************************************/
/* LOGGER METRICS                                                           */
/****************************************************************************/

const string ILoggerMetrics::METRICS = "metrics";
const string ILoggerMetrics::PROCESS = "process";
const string ILoggerMetrics::META = "meta";

bool ILoggerMetrics::failSafe(true);
string ILoggerMetrics::parentObjectId;

shared_ptr<ILoggerMetrics>
ILoggerMetrics::
setup(const string & configKey, const string & coll, const string & appName)
{
    std::lock_guard<std::mutex> lock(m);

    if (logger) {
        throw ML::Exception("Cannot setup more than once");
    }

    string configFile(getEnv("CONFIG"));

    if (configFile.empty()) {
        cerr << ("Logger Metrics setup: CONFIG is not defined,"
                 " logging disabled\n");
        logger.reset(new LoggerMetricsVoid(coll));
    }
    else if (configKey.empty()) {
        cerr << ("Logger Metrics setup: configKey is empty,"
                 " logging disabled\n");
        logger.reset(new LoggerMetricsVoid(coll));
    }
    else {
        Json::Value config = Json::parseFromFile(getEnv("CONFIG"));
        config = config[configKey];

        setupLogger(config, coll, appName);
    }

    return logger;
}

shared_ptr<ILoggerMetrics>
ILoggerMetrics::
setupFromJson(const Json::Value & config,
              const string & coll, const string & appName)
{
    std::lock_guard<std::mutex> lock(m);

    if (logger) {
        throw ML::Exception("Cannot setup more than once");
    }

    setupLogger(config, coll, appName);

    return logger;
}

void
ILoggerMetrics::
setupLogger(const Json::Value & config,
            const string & coll, const string & appName)
{
    ExcAssert(!config.isNull());

    ILoggerMetrics::parentObjectId = getEnv("METRICS_PARENT_ID");

    if (config["type"].isNull()) {
        throw ML::Exception("Your LoggerMetrics config needs to "
                            "specify a [type] key.");
    }

    string loggerType = config["type"].asString();
    failSafe = config["failSafe"].asBool();
    auto fct = [&] {
        if (loggerType == "term" || loggerType == "terminal") {
            logger.reset(new LoggerMetricsTerm(config, coll, appName));
        }
        else if (loggerType == "void") {
            logger.reset(new LoggerMetricsVoid(config, coll, appName));
        }
        else {
            throw ML::Exception("Unknown logger type [%s]", loggerType.c_str());
        }
    };

    if (failSafe) {
        try {
            fct();
        }
        catch (const exception & exc) {
            cerr << "Logger fail safe caught: " << exc.what() << endl;
            logger = shared_ptr<ILoggerMetrics>(
                new LoggerMetricsTerm(config, coll, appName));
        }
    }
    else {
        fct();
    }

    auto getCmdResult = [&] (const char* cmd) {
        FILE* pipe = popen(cmd, "r");
        if (!pipe) {
            return string("ERROR");
        }
        char buffer[128];
        stringstream result;
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL) {
                result << buffer;
            }
        }
        pclose(pipe);
        string res = result.str();
        return res.substr(0, res.length() - 1);//chop \n
    };

    string now = Date::now().printClassic();
    Json::Value v;
    v["startTime"] = now;
    v["appName"] = appName;
    v["parent_id"] = getEnv("METRICS_PARENT_ID");
    v["user"] = string(getEnv("LOGNAME"));
    char hostname[128];
    int hostnameOk = !gethostname(hostname, 128);
    v["hostname"] = string(hostnameOk ? hostname : "");
    v["workingDirectory"] = string(getEnv("PWD"));
    v["gitBranch"] = getCmdResult("git rev-parse --abbrev-ref HEAD");
    v["gitHash"] = getCmdResult("git rev-parse HEAD");
    // Log environment variable RUNID. Useful to give a name to an
    // experiment.
    v["runid"] = getEnv("RUNID");

    logger->logProcess(v);
    setenv("METRICS_PARENT_ID", logger->getProcessId().c_str(), 1);
}

shared_ptr<ILoggerMetrics>
ILoggerMetrics::
getSingleton()
{
    std::lock_guard<std::mutex> lock(m);

    if (!logger) {
        cerr << ("Calling getSingleton without calling setup first,"
                 " logging implicitly disabled.\n");
        logger.reset(new LoggerMetricsVoid(""));
    }

    return logger;
}

void
ILoggerMetrics::
logMetrics(const Json::Value & json)
{
    vector<string> stack;
    std::function<void(const Json::Value &)> doit;
    doit = [&] (const Json::Value & v) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            string memberName = it.memberName();
            if (v[memberName].isObject()) {
                stack.push_back(memberName);
                doit(v[memberName]);
                stack.pop_back();
            }
            else {
                const Json::Value & current = v[memberName];
                if (!(current.isInt() || current.isUInt() || current.isDouble()
                      || current.isNumeric())) {
                    stringstream key;
                    for (const string & s: stack) {
                        key << s << ".";
                    }
                    key << memberName;
                    string value = current.toString();
                    cerr << value << endl;
                    value = value.substr(1, value.length() - 3);
                    throw ML::Exception("logMetrics only accepts numerical"
                                        " values. Key [%s] has value [%s].",
                                        key.str().c_str(), value.c_str());
                }
            }
        }
    };
    auto fct = [&] () {
        doit(json);
        logInCategory(METRICS, json);
    };
    failSafeHelper(fct);
}

void
ILoggerMetrics::
failSafeHelper(const std::function<void()> & fct)
{
    if (failSafe) {
        try {
            fct();
        }
        catch (const exception & exc) {
            cerr << "Logger fail safe caught: " << exc.what() << endl;
        }
    }
    else {
        fct();
    }
}

void
ILoggerMetrics::
close()
{
    Json::Value v;
    Date endDate = Date::now();
    v["endDate"] = endDate.printClassic();
    v["duration"] = endDate - startDate;
    logInCategory(PROCESS, v);
    logProcess(v);
}
