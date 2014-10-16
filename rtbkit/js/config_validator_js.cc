
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"

using namespace std;
using namespace v8;
using namespace node;

namespace Datacratic {
namespace JS {


//set up the module

extern const char * const standaloneModule;
const char * const standaloneModule = "config_validator";

//set up the standalone function
static Handle<v8::Value>
validateConfig(const Arguments & args)
{
    try {
        Json::Value config = getArg(args, 0, "config");
        RTBKIT::AgentConfig::createFromJson(config); //will throw if unhappy
        return JS::toJS(true);
    } HANDLE_JS_EXCEPTIONS;
}


extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, standaloneModule);

    static Persistent<FunctionTemplate> atn
        = v8::Persistent<FunctionTemplate>::New
        (v8::FunctionTemplate::New(validateConfig));

    target->Set(String::NewSymbol("validateConfig"), atn->GetFunction());
}



} // namespace JS
} // namespace Datacratic
