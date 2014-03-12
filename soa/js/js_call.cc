/* js_call.cc
   Jeremy Barnes, 15 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   JS calling and notifier functionality.
*/

#include "js_call.h"
#include <unordered_map>
#include "jml/arch/backtrace.h"
#include "v8.h"
#include "node.h"


using namespace std;

extern "C" {
    // Define as a weak symbol to avoid linker errors when linking without libeio
#   pragma push_macro("eio_custom")
#   ifdef eio_custom
#   undef eio_custom
    __attribute__((__weak__))
    eio_req * eio_custom(void (*)(eio_req*), int, eio_cb, void*, eio_channel*)
    {
        throw ML::Exception("node needs to be linked in to use JS context callbacks");
    }

    __attribute__((__weak__))
    uv_loop_t* uv_default_loop(void) {
        throw ML::Exception("node needs to be linked in to use JS context callbacks");
    }
#   else
    __attribute__((__weak__))
    eio_req * eio_custom(void (*)(eio_req*), int, int (*)(eio_req*), void*)
    {
        throw ML::Exception("node needs to be linked in to use JS context callbacks");
    }
#   endif // eio_custom
#   pragma pop_macro("eio_custom")
};

namespace node {

// Define as a weak symbol to avoid linker errors when linking without node
__attribute__((__weak__))
void FatalException(v8::TryCatch & tc)
{
    throw ML::Exception("node needs to be linked in to use JS context callbacks");
}

} // namespace node


namespace Datacratic {
namespace JS {

/*****************************************************************************/
/* CALL IN JS CONTEXT                                                        */
/*****************************************************************************/

struct CallInJsContextData {
    boost::function<void ()> callback;
    
    ~CallInJsContextData()
    {
    }
};
    
static void doNothing(eio_req * req)
{
    // TODO: don't do this; find how to use libeio properly
}

static int doCallInJs(eio_req * req)
{
    v8::HandleScope scope;

    auto_ptr<CallInJsContextData> data((CallInJsContextData *)req->data);

    v8::TryCatch try_catch;

    try {
        data->callback();
    } catch (const JSPassException & exc) {
        cerr << "got JSPassException" << endl;
        if (!try_catch.HasCaught()) {
            cerr << "handler returned passed exception " << endl;
            ML::backtrace();
            abort();
        }
        // Corner case... but probably shouldn't happen
    } catch (const std::exception & exc) {
        /* v8::Handle<v8::Value> result = */ translateCurrentException();
        // TODO: what do we do with result?
        if (!try_catch.HasCaught()) {
            cerr << "handler returned exception: " << exc.what() << endl;
            ML::backtrace();
            abort();
        }
    } catch (...) {
        /* v8::Handle<v8::Value> result = */ translateCurrentException();
        // TODO: what to do with result?
        if (!try_catch.HasCaught()) {
            cerr << "handler returned exception " << endl;
            ML::backtrace();
            abort();
        }
    }
        
    if (try_catch.HasCaught())
        node::FatalException(try_catch);
    
    return 0;
}

void callInJsThread(const boost::function<void ()> & fn)
{
    // This is called from a thread that is probably not the right thread for
    // JS to be executed in.  Here we arrange for the right thread to be called
    // back by libev once the JS engine is ready to do something.
    
    auto_ptr<CallInJsContextData> data(new CallInJsContextData());
    data->callback = fn;
    eio_custom(doNothing, EIO_PRI_DEFAULT, doCallInJs, data.release());
}

boost::function<void ()>
createCrossThreadCallback(v8::Handle<v8::Function> fn,
                          v8::Handle<v8::Object> This)
{
    calltojsbase args(fn, This);

    auto onceInJsContext = [=] ()
        {
            v8::HandleScope scope;
            v8::Handle<v8::Value> result;
            {
                v8::TryCatch tc;
                result = args.params->fn->Call(args.params->This, 0, 0);
                
                if (result.IsEmpty()) {
                    if(tc.HasCaught()) {
                        tc.ReThrow();
                        throw JSPassException();
                    }
                    throw ML::Exception("didn't return anything");
                }
            }
        };

    auto result = [=] ()
        {
            callInJsContext(onceInJsContext);
        };

    return result;
}


/*****************************************************************************/
/* JS OPERATIONS                                                             */
/*****************************************************************************/

std::unordered_map<const std::type_info *, JSOperations> operations;

void registerJsOps(const std::type_info & type,
                   JSOperations ops)
{
    //cerr << "registered " << ML::demangle(type) << " at " << &type << " with " << &ops << endl;
    //ML::backtrace();
    
    operations[&type] = ops;
}

JSOperations
getOps(const std::type_info & fntype)
{
    auto it = operations.find(&fntype);
    if (it == operations.end())
        throw ML::Exception("no JS operations registered for "
                            + ML::demangle(fntype));
    return it->second;
}

} // namespace JS
} // namespace Datacratic
