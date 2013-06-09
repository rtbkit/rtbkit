#include "easy_kvp_logger.h"
#include "soa/types/date.h"

namespace Datacratic{

using namespace std;
using namespace boost::program_options;

EasyKvpLogger::EasyKvpLogger(const boost::property_tree::ptree& pt, 
    const string& coll, const string& envVar, const string& runId,
    const bool& logStartEnd, const strmap& defaults) : coll(coll),
    runId(runId), logStartEnd(logStartEnd), start(Date::now())
{
    setEnvVar(envVar, runId);
    defineLogger(pt, defaults, logStartEnd);
}

EasyKvpLogger::EasyKvpLogger(const boost::property_tree::ptree& pt, 
    const string& coll, const string& envVar, const bool& logStartEnd,
    const strmap& defaults) : coll(coll), logStartEnd(logStartEnd),
    start(Date::now())
{
    setDateAsRunIdIfNotInEnvVar(envVar); 
    defineLogger(pt, defaults, logStartEnd);
}

EasyKvpLogger::EasyKvpLogger(const variables_map& vm, 
    const string& coll, const string& envVar, const bool& logStartEnd,
    const strmap& defaults) : coll(coll), logStartEnd(logStartEnd),
    start(Date::now())
{
    setDateAsRunIdIfNotInEnvVar(envVar);
    defineLogger(vm, defaults, logStartEnd);
}

void EasyKvpLogger::setDateAsRunIdIfNotInEnvVar(const string & envVar)
{    
    char* runIdTmp;
    if((runIdTmp = getenv(envVar.c_str()))){
        runId = string(runIdTmp);
        cout << "EasyKvpLogger reading " << envVar << " as " << runId << endl;
    }else{
        Date d = Date::now();
        runId = d.printClassic();
        setEnvVar(envVar, runId);
    }
}
 

EasyKvpLogger::~EasyKvpLogger(){
    if(logStartEnd){
        Date end = Date::now();
        clog({
            {"key", "end"},
            {"val", end.printClassic()},
            {"duration",
                to_string(end.secondsSinceEpoch() - start.secondsSinceEpoch())}
        });
    }
}

void EasyKvpLogger::logStart(){
    log("start", start.printClassic());
}


void EasyKvpLogger::defineLogger(const boost::property_tree::ptree& pt,
    const strmap& defaults, const bool& logStartEnd)
{
    IKvpLogger::KvpLoggerParams params;
    params.hostAndPort = pt.get<string>("logger.hostAndPort");
    params.db          = pt.get<string>("logger.db");
    params.user        = pt.get<string>("logger.user");
    params.pwd         = pt.get<string>("logger.pwd");
    params.failSafe    = pt.get<bool>("logger.failSafe");
    string type        = pt.get<string>("logger.type");

    logger = std::shared_ptr<IKvpLogger>(
        IKvpLogger::kvpLoggerFactory(type, params));
    setDefaults(defaults);
    if (logStartEnd)
        logStart();
}

void EasyKvpLogger::defineLogger(const variables_map& vm,
    const strmap& defaults, const bool& logStartEnd)
{
    IKvpLogger::KvpLoggerParams params;
    params.hostAndPort  = vm["logger.hostAndPort"].as<string>();
    params.db           = vm["logger.db"].as<string>();
    params.user         = vm["logger.user"].as<string>();
    params.pwd          = vm["logger.pwd"].as<string>();
    string type         = vm["logger.type"].as<string>();
    params.failSafe     = vm["logger.failSafe"].as<bool>();

    logger = std::shared_ptr<IKvpLogger>(
        IKvpLogger::kvpLoggerFactory(type, params));
    setDefaults(defaults);
    if (logStartEnd)
        logStart();
}
    

void EasyKvpLogger
::setEnvVar(const std::string& envVar, const std::string& runId){
    if(getenv(envVar.c_str()) && runId.length() > 0){
        throw ML::Exception(envVar + " is already defined.");
    }
    cout << "EasyKvpLogger defining " << envVar << " as " << runId << endl;
    setenv(envVar.c_str(), runId.c_str(), 0);
}

void EasyKvpLogger::log(strmap& kvpMap){
    addDefaultsToMap(kvpMap);
    logger->log(kvpMap, coll);
}

void EasyKvpLogger::clog(const strmap& kvpMap){
    strmap values = kvpMap; 
    log(values);
}

void EasyKvpLogger::log(const string& key, const string& value){
    strmap kvpMap;
    kvpMap["key"]   = key;
    kvpMap["val"]   = value;
    addDefaultsToMap(kvpMap);
    logger->log(kvpMap, coll);
}

string EasyKvpLogger::getRunId(){
    return runId;
}

void EasyKvpLogger::addDefaultsToMap(strmap& kvpMap){
    kvpMap["runId"] = runId;
    for(strmap_citerator it = defaults.begin(); it != defaults.end(); ++ it){
        kvpMap[it->first] = it->second;
    }
}

void EasyKvpLogger::setDefaults(const strmap& defaults){
    this->defaults = defaults;
}


options_description
EasyKvpLogger::get_options()
{
    options_description options;
    options.add_options()
        ("logger.hostAndPort", value<string>()->required())
        ("logger.db", value<string>()->required())
        ("logger.user", value<string>()->required())
        ("logger.pwd", value<string>()->required())
        ("logger.type", value<string>()->required())
        ("logger.failSafe", value<bool>()->required());
    return options;
}

}//namespace Datacratic
