#include "soa/logger/logger_metrics_interface.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_call.h"
#include "soa/js/js_registry.h"
#include "soa/js/js_value.h"
#include <boost/make_shared.hpp>
#include "soa/sigslot/slot.h"
#include "soa/types/js/id_js.h"

using namespace std;
using namespace v8;
using namespace node;

namespace Datacratic {
namespace JS {


extern const char * const ILoggerMetricsModule;
const char * const ILoggerMetricsModule = "iloggermetricscpp";

namespace{

Handle<v8::Value> setup(const Arguments& args)
{
    try{
        if(args.Length() != 3){
            ThrowException(Exception::TypeError(
                String::New("Wrong number of arguments")));
        }
        string configKey = getArg(args, 0, "str");
        string coll      = getArg(args, 1, "str");
        string appName   = getArg(args, 2, "str");
        ILoggerMetrics::setup(configKey, coll, appName);
    }HANDLE_JS_EXCEPTIONS;
    Handle<v8::Value> h;
    return h;
}

Handle<v8::Value>
log(const Arguments& args, const string& method){
    try{
        shared_ptr<ILoggerMetrics> logger = ILoggerMetrics::getSingleton();
        if(args.Length() < 1){
            ThrowException(Exception::TypeError(
                String::New("Wrong number of arguments")));
        }
        if(args.Length() == 1){
            Json::Value v;
            v = getArg(args, 0, "json");
            if(method == "metrics"){
                logger->logMetrics(v);
            }else if(method == "meta"){
                logger->logMeta(v);
            }else if(method == "process"){
                logger->logProcess(v);
            }else{
                string msg = "Unknown log method [" + method + "]";
                ThrowException(Exception::TypeError(
                    String::New(msg.c_str())));
            }
        }else{
            vector<string> v;
            for(int i = 0; i < args.Length() - 1; ++i){
                v.push_back(getArg(args, i, "str"));
            }
            if(method == "metrics"){
                float f = getArg(args, args.Length() - 1, "float");
                logger->logMetrics(v, f);
            }else if(method == "meta"){
                string s = getArg(args, args.Length() - 1, "float");
                logger->logMeta(v, s);
            }else if(method == "process"){
                string s = getArg(args, args.Length() - 1, "float");
                logger->logProcess(v, s);
            }else{
                string msg = "Unknown log method [" + method + "]";
                ThrowException(Exception::TypeError(
                    String::New(msg.c_str())));
            }
        }
    }HANDLE_JS_EXCEPTIONS;
    Handle<v8::Value> h;
    return h;
}

Handle<v8::Value>
logMetrics(const Arguments& args){
    return log(args, "metrics");
}
Handle<v8::Value>
logMeta(const Arguments& args){
    Json::Value v;
    return log(args, "meta");
}
Handle<v8::Value>
logProcess(const Arguments& args){
    return log(args, "process");
}

} //anonymous namespace

/*****************************************************************************/
/* ATTRIBUTOR JS                                                             */
/*****************************************************************************/

const char * ILoggerMetricsName = "ILoggerMetrics";

struct ILoggerMetricsJS
    : public JSWrapped2<ILoggerMetrics, ILoggerMetricsJS, ILoggerMetricsName, ILoggerMetricsModule> {

    static void
    Initialize()
    {
        //Persistent<FunctionTemplate> t = Register(New);
        //registerMemberFn(&ILoggerMetrics::close, "close");
    }

};





/*****************************************************************************/
/* INITIALIZATION */
/*****************************************************************************/

extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, ILoggerMetricsModule);
    target->Set(String::NewSymbol("setup"),
                v8::Persistent<FunctionTemplate>::New
                (v8::FunctionTemplate::New(setup))->GetFunction());
    target->Set(String::NewSymbol("logMetrics"),
                v8::Persistent<FunctionTemplate>::New
                (v8::FunctionTemplate::New(logMetrics))->GetFunction());
    target->Set(String::NewSymbol("logMeta"),
                v8::Persistent<FunctionTemplate>::New
                (v8::FunctionTemplate::New(logMeta))->GetFunction());
    target->Set(String::NewSymbol("logProcess"),
                v8::Persistent<FunctionTemplate>::New
                (v8::FunctionTemplate::New(logProcess))->GetFunction());
}



} // namespace JS
} // namespace Datacratic
