#include "kvp_logger_interface.h"
#include "kvp_logger_mongodb.h"
#include "kvp_logger_void.h"
#include <boost/property_tree/json_parser.hpp>

namespace Datacratic{

std::shared_ptr<IKvpLogger>
IKvpLogger
::kvpLoggerFactory(const std::string& type, const KvpLoggerParams& params){
    if(type == "mongodb"){
        return std::shared_ptr<IKvpLogger>(new KvpLoggerMongoDb(params));
    }else if(type == "void"){
        return std::shared_ptr<IKvpLogger>(new KvpLoggerVoid());
    }else if(type == "metricsLogger"){
    
    }
    throw ML::Exception("Unknown KvpLogger [" + type + "]");
}

std::shared_ptr<IKvpLogger>
IKvpLogger
::kvpLoggerFactory(const std::string& configKey){
    using namespace boost::property_tree;
    using namespace std;
    ptree pt;
    json_parser::read_json(getenv("CONFIG"), pt);
    pt = pt.get_child(configKey);
    string type = pt.get<string>("type");
    if(type == "mongodb"){
    
    }else if(type == "void"){
    
    }else if(type == "mongodbMetrics"){
    
    }
    throw ML::Exception("Unknown KvpLogger [" + type + "]");
}
}
