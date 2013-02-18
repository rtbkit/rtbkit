#include "kvp_logger_interface.h"
#include "kvp_logger_mongodb.h"
#include "kvp_logger_void.h"
namespace Datacratic{

std::shared_ptr<IKvpLogger>
IKvpLogger
::getKvpLogger(const std::string& type, const KvpLoggerParams& params){
    if(type == "mongodb"){
        return std::shared_ptr<IKvpLogger>(new KvpLoggerMongoDb(params));
    }else if(type == "void"){
        return std::shared_ptr<IKvpLogger>(new KvpLoggerVoid());
    }
    throw ML::Exception("Unknown KvpLogger [" + type + "]");
}

}
