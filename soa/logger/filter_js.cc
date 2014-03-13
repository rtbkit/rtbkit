/* filter_js.cc
   Jeremy Barnes, 30 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "filter.h"
#include "json_filter.h"
#include "v8.h"
#include "node.h"
#include "soa/js/js_value.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_wrapped.h"
#include "soa/js/js_call.h"
#include "soa/js/js_registry.h"
#include "jml/arch/timers.h"
#include "jml/utils/guard.h"
#include "soa/sigslot/slot.h"
#include "jml/utils/guard.h"
#include "soa/sigslot/slot_js.h"
#include "ace/Synch.h"
#include "jml/utils/smart_ptr_utils.h"

using namespace std;
using namespace v8;
using namespace node;


namespace Datacratic {
namespace JS {

extern Registry registry;

extern const char * const loggerModule;

Direction from_js(const JSValue & value, Direction * = 0)
{
    string str = cstr(value);
    if (str == "COMPRESS")
        return COMPRESS;
    else if (str == "DECOMPRESS")
        return DECOMPRESS;
    throw ML::Exception("couldn't convert JS value " + str + " to Direction");
}

void to_js(JSValue & value, const Direction & dir)
{
    switch (dir) {
    case COMPRESS:
        value = String::NewSymbol("COMPRESS");  return;
    case DECOMPRESS:
        value = String::NewSymbol("DECOMPRESS");  return;
    default:
        throw ML::Exception("unknown Direction %d", dir);
    }
}

FlushLevel from_js(const JSValue & value, FlushLevel * = 0)
{
    string str = cstr(value);
    if (str == "FLUSH_NONE")
        return FLUSH_NONE;
    else if (str == "FLUSH_SYNC")
        return FLUSH_SYNC;
    else if (str == "FLUSH_FULL")
        return FLUSH_FULL;
    else if (str == "FLUSH_FINISH")
        return FLUSH_FINISH;
    throw ML::Exception("couldn't convert JS value " + str + " to FlushLevel");
}

void to_js(JSValue & value, const FlushLevel & level)
{
    switch (level) {
    case FLUSH_NONE:   value = String::NewSymbol("FLUSH_NONE");    return;
    case FLUSH_SYNC:   value = String::NewSymbol("FLUSH_SYNC");    return;
    case FLUSH_FULL:   value = String::NewSymbol("FLUSH_FULL");    return;
    case FLUSH_FINISH: value = String::NewSymbol("FLUSH_FINISH");  return;
    default:
        throw ML::Exception("Unknown FlushLevel %d", level);
    }
}


/*****************************************************************************/
/* FILTER JS                                                                 */
/*****************************************************************************/

const char * FilterName = "Filter";

struct FilterJS
    : public JSWrapped2<Filter, FilterJS, FilterName,
                        loggerModule, true> {

    FilterJS()
    {
    }

    FilterJS(const v8::Handle<v8::Object> & This,
                  const std::shared_ptr<Filter> & handler
                      = std::shared_ptr<Filter>())
    {
        wrap(This, handler);
    }

    ~FilterJS()
    {
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new FilterJS(args.This());
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "flush", flush);
        NODE_SET_PROTOTYPE_METHOD(t, "flushSync", flushSync);
        NODE_SET_PROTOTYPE_METHOD(t, "process", process);
        NODE_SET_PROTOTYPE_METHOD(t, "processSync", processSync);

        // Members
        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("onOutput"), onOutputGetter,
                          onOutputSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));

        t->InstanceTemplate()
            ->SetAccessor(String::NewSymbol("onError"), onErrorGetter,
                          onErrorSetter, v8::Handle<v8::Value>(), DEFAULT,
                          PropertyAttribute(DontDelete));


        // Class methods
        t->Set(String::NewSymbol("create"),
               FunctionTemplate::New(create));

        // Compress flags
        t->Set(String::NewSymbol("COMPRESS"),
               String::NewSymbol("COMPRESS"));
        t->Set(String::NewSymbol("DECOMPRESS"),
               String::NewSymbol("DECOMPRESS"));

        // Flush flags
        t->Set(String::NewSymbol("FLUSH_NONE"),
               String::NewSymbol("FLUSH_NONE"));
        t->Set(String::NewSymbol("FLUSH_SYNC"),
               String::NewSymbol("FLUSH_SYNC"));
        t->Set(String::NewSymbol("FLUSH_FULL"),
               String::NewSymbol("FLUSH_FULL"));
        t->Set(String::NewSymbol("FLUSH_FINISH"),
               String::NewSymbol("FLUSH_FINISH"));
    }

    static Handle<v8::Value>
    flush(const Arguments & args)
    {
        try {
            if (args[1]->IsUndefined() || args[1]->IsNull())
                getShared(args)->flush(getArg(args, 0, "flushLevel"));
            else {
                // This callback may be called from another thread later on, which
                // will not be in the JS interpreter.
                // As a result, we need to do all of the manipulation of the function
                // now and arrange for it to be called from the right thread.

                auto cb = createCrossThreadCallback(getArg(args, 1, "callback"),
                                                    args.This());

                getShared(args)->flush(getArg(args, 0, "flushLevel"),
                                       cb);
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    flushSync(const Arguments & args)
    {
        try {
            ACE_Semaphore sem(0);
            auto cb = [&] () { sem.release(); };
            
            getShared(args)->flush(getArg(args, 0, "flushLevel"),
                                   cb);

            sem.acquire();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    process(const Arguments & args)
    {
        try {
            if (args[2]->IsUndefined() || args[2]->IsNull())
                getShared(args)->process(getArg<string>(args, 0, "buf"),
                                         getArg(args, 1, FLUSH_NONE, "flushLevel"));
            else {
                // This callback may be called from another thread later on, which
                // will not be in the JS interpreter.
                // As a result, we need to do all of the manipulation of the function
                // now and arrange for it to be called from the right thread.

                auto cb = createCrossThreadCallback(getArg(args, 2, "callback"),
                                                    args.This());
                getShared(args)->process(getArg(args, 0, "buf"),
                                         getArg(args, 1, "flushLevel"),
                                         cb);
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    processSync(const Arguments & args)
    {
        try {
            ACE_Semaphore sem(0);
            auto cb = [&] () { sem.release(); };
            
            getShared(args)->process(getArg(args, 0, "buf"),
                                     getArg(args, 1, FLUSH_NONE, "flushLevel"),
                                     cb);

            sem.acquire();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    create(const Arguments & args)
    {
        try {
            return registry.getWrapper
                (ML::make_std_sp
                 (Filter::create(getArg(args, 0, "extension"),
                                 getArg(args, 1, COMPRESS, "direction"))));
            
        } HANDLE_JS_EXCEPTIONS;
    }

    static v8::Handle<v8::Value>
    onOutputGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            return JS::toJS(Slot(getShared(info.This())->onOutput));
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    onOutputSetter(v8::Local<v8::String> property,
                  v8::Local<v8::Value> value,
                  const v8::AccessorInfo & info)
    {
        try {
            Slot slot = JS::fromJS(value);
            getShared(info.This())->onOutput = slot.as<Filter::OnOutputFn>();
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }

    static v8::Handle<v8::Value>
    onErrorGetter(v8::Local<v8::String> property,
                  const v8::AccessorInfo & info)
    {
        try {
            return JS::toJS(Slot(getShared(info.This())->onError));
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    onErrorSetter(v8::Local<v8::String> property,
                  v8::Local<v8::Value> value,
                  const v8::AccessorInfo & info)
    {
        try {
            Slot slot = JS::fromJS(value);
            getShared(info.This())->onError = slot.as<Filter::OnErrorFn>();
        } HANDLE_JS_EXCEPTIONS_SETTER;
    }
};

std::shared_ptr<Filter>
from_js(const JSValue & value, std::shared_ptr<Filter> *)
{
    return FilterJS::fromJS(value);
}

Filter *
from_js(const JSValue & value, Filter **)
{
    return FilterJS::fromJS(value).get();
}

struct OnOutputJsOps: public JS::JsOpsBase<OnOutputJsOps, Filter::OnOutputFn> {

    static v8::Handle<v8::Value>
    callBoost(const Function & fn,
              const JS::JSArgs & args)
    {
        throw ML::Exception("callBoost onOutput");
#if 0
        Result result
            = JS::callfromjs<Function, Function::arity>::call(fn, args);

        JS::JSValue jsresult;
        JS::to_js(jsresult, result);
        
        return jsresult;
#endif
    }

    static Function
    asBoost(const v8::Handle<v8::Function> & fn,
            const v8::Handle<v8::Object> * This)
    {
        v8::Handle<v8::Object> This2;
        if (!This)
            This2 = v8::Object::New();
        calltojsbase params(fn, This ? *This : This2);

        auto result = [=] (const char * buf, size_t numChars,
                           FlushLevel flush, boost::function<void ()> fn)
            {
                string str(buf, buf + numChars);

                v8::HandleScope scope;
                v8::Handle<v8::Value> result;
                {
                    v8::TryCatch tc;
                    v8::Handle<v8::Value> argv[3];
                    argv[0] = JS::toJS(str);
                    argv[1] = JS::toJS(flush);
                    argv[2] = JS::toJS(Slot(fn));

                    result = params.params->fn->Call(params.params->This, 3, argv);
                    
                    if (result.IsEmpty()) {
                        if(tc.HasCaught()) {
                            tc.ReThrow();
                            throw JSPassException();
                        }
                    }
                }
            };

        return result;
    }
};



RegisterJsOps<Filter::OnOutputFn> reg_onOutputFn(OnOutputJsOps::op);
RegisterJsOps<Filter::OnErrorFn> reg_onErrorFn;
RegisterJsOps<void ()> reg_plainCallback;

#if 0
/*****************************************************************************/
/* JS OUTPUT                                                                 */
/*****************************************************************************/

class JSOutputJS;

/** Class that outputs messages via a javascript function. */

struct JSOutput : public Filter {

    JSOutput()
        : wrapper(0)
    {
    }

    virtual ~JSOutput()
    {
    }

    Slot logMessageSlot;
    JSOutputJS * wrapper;

    struct CallbackData {
        JSOutput * ThisPtr;
        //v8::Persistent<v8::Object> ThisObj;  // make sure we're not GCd
        std::string channel;
        std::string message;

        ~CallbackData()
        {
            //if (!ThisObj.IsEmpty()) {
            //    ThisObj.Dispose();
            //    ThisObj.Clear();
            //}
        }
    };
    
    static int doNothing(eio_req * req)
    {
        // TODO: don't do this; find how to use libeio properly
        return 0;
    }

    static int finishedCallback(eio_req * req)
    {
        HandleScope scope;

        auto_ptr<CallbackData> data((CallbackData *)req->data);

        TryCatch try_catch;

        try {
            data->ThisPtr->
                logMessageSlot.call<void (std::string, std::string)>
                (data->channel, data->message);
        } catch (const JSPassException & exc) {
            cerr << "got JSPassException" << endl;
            if (!try_catch.HasCaught()) {
                cerr << "handler returned passed exception " << endl;
                ML::backtrace();
                abort();
            }
            // Corner case... but probably shouldn't happen
        } catch (const std::exception & exc) {
            v8::Handle<v8::Value> result = translateCurrentException();
            if (!try_catch.HasCaught()) {
                cerr << "handler returned exception: " << exc.what() << endl;
                ML::backtrace();
                abort();
            }
        } catch (...) {
            v8::Handle<v8::Value> result = translateCurrentException();
            if (!try_catch.HasCaught()) {
                cerr << "handler returned exception " << endl;
                ML::backtrace();
                abort();
            }
        }
        
        if (try_catch.HasCaught())
            FatalException(try_catch);

        return 0;
    }

    virtual void logMessage(const std::string & channel,
                            const std::string & message);
};

RegisterJsOps<void (std::string, std::string)> reg_logMessage;

/*****************************************************************************/
/* JS OUTPUT JS                                                              */
/*****************************************************************************/

const char * JSOutputName = "JSOutput";

struct JSOutputJS
    : public JSWrapped3<JSOutput, JSOutputJS, FilterJS, JSOutputName,
                        loggerModule, true> {

    JSOutputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<JSOutput> & output
                          = std::shared_ptr<JSOutput>())
    {
        wrap(This, output);
        getShared(This)->wrapper = this;
    }

    ~JSOutputJS()
    {
        getShared(js_object_)->wrapper = 0;
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new JSOutputJS
                (args.This(),
                 std::shared_ptr<JSOutput>
                 (new JSOutput()));
            
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        RWPropertyHandler<JSOutputJS, JSOutput, Slot, &JSOutput::logMessageSlot>
            handleLogMessage(t, "logMessageFn");
    }
};


void
JSOutput::
logMessage(const std::string & channel,
           const std::string & message)
{
    // This is called from a thread that is probably not the right thread for
    // JS to be executed in.  Here we arrange for the right thread to be called
    // back by libev once the JS engine is ready to do something.
    
    auto_ptr<CallbackData> data(new CallbackData());
    
    data->channel = channel;
    data->message = message;
    data->ThisPtr = this;
    //data->ThisObj = v8::Persistent<v8::Object>::New(wrapper->js_object_);
    
    eio_custom(doNothing, EIO_PRI_DEFAULT, finishedCallback, data.release());
}
#endif


/*****************************************************************************/
/* IDENTITY FILTER JS                                                        */
/*****************************************************************************/

const char * IdentityFilterName = "IdentityFilter";

struct IdentityFilterJS
    : public JSWrapped3<IdentityFilter, IdentityFilterJS, FilterJS,
                        IdentityFilterName, loggerModule, true> {

    IdentityFilterJS()
    {
    }

    IdentityFilterJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<IdentityFilter> & filter
                 = std::shared_ptr<IdentityFilter>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new IdentityFilterJS
                (args.This(),
                 std::shared_ptr<IdentityFilter>(new IdentityFilter()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<IdentityFilter>
from_js(const JSValue & value, std::shared_ptr<IdentityFilter> *)
{
    return IdentityFilterJS::fromJS(value);
}

IdentityFilter *
from_js(const JSValue & value, IdentityFilter **)
{
    return IdentityFilterJS::fromJS(value).get();
}


/*****************************************************************************/
/* ZLIB COMPRESSOR JS                                                        */
/*****************************************************************************/

const char * ZlibCompressorName = "ZlibCompressor";

struct ZlibCompressorJS
    : public JSWrapped3<ZlibCompressor, ZlibCompressorJS, FilterJS,
                        ZlibCompressorName, loggerModule, true> {

    ZlibCompressorJS()
    {
    }

    ZlibCompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<ZlibCompressor> & filter
                 = std::shared_ptr<ZlibCompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new ZlibCompressorJS
                (args.This(),
                 std::shared_ptr<ZlibCompressor>(new ZlibCompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<ZlibCompressor>
from_js(const JSValue & value, std::shared_ptr<ZlibCompressor> *)
{
    return ZlibCompressorJS::fromJS(value);
}

ZlibCompressor *
from_js(const JSValue & value, ZlibCompressor **)
{
    return ZlibCompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* ZLIB DECOMPRESSOR JS                                                      */
/*****************************************************************************/

const char * ZlibDecompressorName = "ZlibDecompressor";

struct ZlibDecompressorJS
    : public JSWrapped3<ZlibDecompressor, ZlibDecompressorJS, FilterJS,
                        ZlibDecompressorName, loggerModule, true> {

    ZlibDecompressorJS()
    {
    }

    ZlibDecompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<ZlibDecompressor> & filter
                 = std::shared_ptr<ZlibDecompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new ZlibDecompressorJS
                (args.This(),
                 std::shared_ptr<ZlibDecompressor>(new ZlibDecompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<ZlibDecompressor>
from_js(const JSValue & value, std::shared_ptr<ZlibDecompressor> *)
{
    return ZlibDecompressorJS::fromJS(value);
}

ZlibDecompressor *
from_js(const JSValue & value, ZlibDecompressor **)
{
    return ZlibDecompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* GZIP COMPRESSOR JS                                                        */
/*****************************************************************************/

const char * GzipCompressorName = "GzipCompressor";

struct GzipCompressorJS
    : public JSWrapped3<GzipCompressorFilter, GzipCompressorJS, FilterJS,
                        GzipCompressorName, loggerModule, true> {

    GzipCompressorJS()
    {
    }

    GzipCompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<GzipCompressorFilter> & filter
                 = std::shared_ptr<GzipCompressorFilter>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new GzipCompressorJS
                (args.This(),
                 std::shared_ptr<GzipCompressorFilter>(new GzipCompressorFilter()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<GzipCompressorFilter>
from_js(const JSValue & value, std::shared_ptr<GzipCompressorFilter> *)
{
    return GzipCompressorJS::fromJS(value);
}

GzipCompressorFilter *
from_js(const JSValue & value, GzipCompressorFilter **)
{
    return GzipCompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* GZIP DECOMPRESSOR JS                                                      */
/*****************************************************************************/

const char * GzipDecompressorName = "GzipDecompressor";

struct GzipDecompressorJS
    : public JSWrapped3<GzipDecompressor, GzipDecompressorJS, FilterJS,
                        GzipDecompressorName, loggerModule, true> {

    GzipDecompressorJS()
    {
    }

    GzipDecompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<GzipDecompressor> & filter
                 = std::shared_ptr<GzipDecompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new GzipDecompressorJS
                (args.This(),
                 std::shared_ptr<GzipDecompressor>(new GzipDecompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<GzipDecompressor>
from_js(const JSValue & value, std::shared_ptr<GzipDecompressor> *)
{
    return GzipDecompressorJS::fromJS(value);
}

GzipDecompressor *
from_js(const JSValue & value, GzipDecompressor **)
{
    return GzipDecompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* LZMA COMPRESSOR JS                                                        */
/*****************************************************************************/

const char * LzmaCompressorName = "LzmaCompressor";

struct LzmaCompressorJS
    : public JSWrapped3<LzmaCompressor, LzmaCompressorJS, FilterJS,
                        LzmaCompressorName, loggerModule, true> {

    LzmaCompressorJS()
    {
    }

    LzmaCompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<LzmaCompressor> & filter
                 = std::shared_ptr<LzmaCompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            int level = getArg(args, 0, 6, "compressionLevel");
            new LzmaCompressorJS
                (args.This(),
                 std::shared_ptr<LzmaCompressor>(new LzmaCompressor(level)));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<LzmaCompressor>
from_js(const JSValue & value, std::shared_ptr<LzmaCompressor> *)
{
    return LzmaCompressorJS::fromJS(value);
}

LzmaCompressor *
from_js(const JSValue & value, LzmaCompressor **)
{
    return LzmaCompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* LZMA DECOMPRESSOR JS                                                      */
/*****************************************************************************/

const char * LzmaDecompressorName = "LzmaDecompressor";

struct LzmaDecompressorJS
    : public JSWrapped3<LzmaDecompressor, LzmaDecompressorJS, FilterJS,
                        LzmaDecompressorName, loggerModule, true> {

    LzmaDecompressorJS()
    {
    }

    LzmaDecompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<LzmaDecompressor> & filter
                 = std::shared_ptr<LzmaDecompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new LzmaDecompressorJS
                (args.This(),
                 std::shared_ptr<LzmaDecompressor>(new LzmaDecompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<LzmaDecompressor>
from_js(const JSValue & value, std::shared_ptr<LzmaDecompressor> *)
{
    return LzmaDecompressorJS::fromJS(value);
}

LzmaDecompressor *
from_js(const JSValue & value, LzmaDecompressor **)
{
    return LzmaDecompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* BZIP2 COMPRESSOR JS                                                       */
/*****************************************************************************/

const char * Bzip2CompressorName = "Bzip2Compressor";

struct Bzip2CompressorJS
    : public JSWrapped3<Bzip2Compressor, Bzip2CompressorJS, FilterJS,
                        Bzip2CompressorName, loggerModule, true> {

    Bzip2CompressorJS()
    {
    }

    Bzip2CompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<Bzip2Compressor> & filter
                 = std::shared_ptr<Bzip2Compressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new Bzip2CompressorJS
                (args.This(),
                 std::shared_ptr<Bzip2Compressor>(new Bzip2Compressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<Bzip2Compressor>
from_js(const JSValue & value, std::shared_ptr<Bzip2Compressor> *)
{
    return Bzip2CompressorJS::fromJS(value);
}

Bzip2Compressor *
from_js(const JSValue & value, Bzip2Compressor **)
{
    return Bzip2CompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* BZIP2 DECOMPRESSOR JS                                                     */
/*****************************************************************************/

const char * Bzip2DecompressorName = "Bzip2Decompressor";

struct Bzip2DecompressorJS
    : public JSWrapped3<Bzip2Decompressor, Bzip2DecompressorJS, FilterJS,
                        Bzip2DecompressorName, loggerModule, true> {

    Bzip2DecompressorJS()
    {
    }

    Bzip2DecompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<Bzip2Decompressor> & filter
                 = std::shared_ptr<Bzip2Decompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new Bzip2DecompressorJS
                (args.This(),
                 std::shared_ptr<Bzip2Decompressor>(new Bzip2Decompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<Bzip2Decompressor>
from_js(const JSValue & value, std::shared_ptr<Bzip2Decompressor> *)
{
    return Bzip2DecompressorJS::fromJS(value);
}

Bzip2Decompressor *
from_js(const JSValue & value, Bzip2Decompressor **)
{
    return Bzip2DecompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* JSON COMPRESSOR JS                                                        */
/*****************************************************************************/

const char * JsonCompressorName = "JsonCompressor";

struct JsonCompressorJS
    : public JSWrapped3<JsonCompressor, JsonCompressorJS, FilterJS,
                        JsonCompressorName, loggerModule, true> {

    JsonCompressorJS()
    {
    }

    JsonCompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<JsonCompressor> & filter
                 = std::shared_ptr<JsonCompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new JsonCompressorJS
                (args.This(),
                 std::shared_ptr<JsonCompressor>(new JsonCompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<JsonCompressor>
from_js(const JSValue & value, std::shared_ptr<JsonCompressor> *)
{
    return JsonCompressorJS::fromJS(value);
}

JsonCompressor *
from_js(const JSValue & value, JsonCompressor **)
{
    return JsonCompressorJS::fromJS(value).get();
}


/*****************************************************************************/
/* JSON DECOMPRESSOR JS                                                      */
/*****************************************************************************/

const char * JsonDecompressorName = "JsonDecompressor";

struct JsonDecompressorJS
    : public JSWrapped3<JsonDecompressor, JsonDecompressorJS, FilterJS,
                        JsonDecompressorName, loggerModule, true> {

    JsonDecompressorJS()
    {
    }

    JsonDecompressorJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<JsonDecompressor> & filter
                 = std::shared_ptr<JsonDecompressor>())
    {
        wrap(This, filter);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new JsonDecompressorJS
                (args.This(),
                 std::shared_ptr<JsonDecompressor>(new JsonDecompressor()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
    }
};

std::shared_ptr<JsonDecompressor>
from_js(const JSValue & value, std::shared_ptr<JsonDecompressor> *)
{
    return JsonDecompressorJS::fromJS(value);
}

JsonDecompressor *
from_js(const JSValue & value, JsonDecompressor **)
{
    return JsonDecompressorJS::fromJS(value).get();
}

} // namespace JS
} // namespace Datacratic
