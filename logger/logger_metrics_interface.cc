/* logger_metrics_interface.cc
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include "soa/jsoncpp/reader.h"
#include "soa/logger/logger_metrics_interface.h"
#include "soa/logger/logger_metrics_mongo.h"
#include "soa/logger/logger_metrics_void.h"
#include "soa/logger/logger_metrics_term.h"

using namespace std;
using namespace Datacratic;

namespace {

std::mutex m;
std::mutex m2;
bool mustSetup = true;

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

string ILoggerMetrics::parentObjectId = "";
shared_ptr<ILoggerMetrics> logger;
const string ILoggerMetrics::METRICS = "metrics";
const string ILoggerMetrics::PROCESS = "process";
const string ILoggerMetrics::META = "meta";
bool ILoggerMetrics::failSafe;

shared_ptr<ILoggerMetrics>
ILoggerMetrics::
setup(const string & configKey, const string & coll, const string & appName)
{
    std::lock_guard<std::mutex> lock(m);
    if (mustSetup) {
        mustSetup = false;
        parentObjectId = getEnv("METRICS_PARENT_ID");
        if (!getenv("CONFIG") || configKey == "") {
            cerr << "Logger Metrics Setup: either CONFIG is not defined "
                "or configKey empty. "
                "Will not log." << endl;
            Json::Value fooConfig;
            logger = shared_ptr<ILoggerMetrics>(
                new LoggerMetricsVoid(fooConfig, coll, appName));
        }
        else {
            Json::Value config = Json::parseFromFile(getEnv("CONFIG"));
            config = config[configKey];
            if (config.isNull()) {
                throw ML::Exception("Your configKey [" + configKey + "] is invalid or your "
                                    "config file is empty");
            }
            if (config["type"].isNull()) {
                throw ML::Exception("Your LoggerMetrics config needs to "
                                    "specify a [type] key.");
            }
            string loggerType = config["type"].asString();
            failSafe = config["failSafe"].asBool();
            auto fct = [&] () {
                if (loggerType == "mongo") {
                    logger = shared_ptr<ILoggerMetrics>(
                        new LoggerMetricsMongo(config, coll, appName));
                }
                else if (loggerType == "term" || loggerType == "terminal") {
                    logger = shared_ptr<ILoggerMetrics>(
                        new LoggerMetricsTerm(config, coll, appName));
                }
                else if (loggerType == "void") {
                    logger = shared_ptr<ILoggerMetrics>(
                        new LoggerMetricsVoid(config, coll, appName));
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
        }
    }
    else {
        throw ML::Exception("Cannot setup more than once");
    }

    auto getCmdResult = [] (const char* cmd) {
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

    return logger;
}

shared_ptr<ILoggerMetrics>
ILoggerMetrics::
getSingleton()
{
    if (mustSetup) {
        std::lock_guard<std::mutex> lock(m2);
        if (mustSetup) {
            cerr << "Calling getSingleton without calling setup first."
                 << "Will return a logger metrics terminal." << endl;
            return setup("", "", "");
        }
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
                    throw ML::Exception("logMetrics only accepts numerical "
                                        "values. Key [%s] has value [%s].",
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
