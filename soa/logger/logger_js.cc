/* logger_logger_js.cc
   Jeremy Barnes, 20 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "logger.h"
#include "soa/types/periodic_utils.h"
#include "remote_output.h"
#include "remote_input.h"
#include "file_output.h"
#include "publish_output.h"
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
#include <boost/static_assert.hpp>
#include "soa/js/js_call.h"

using namespace std;
using namespace v8;
using namespace node;


namespace Datacratic {
namespace JS {

extern Registry registry;

extern const char * const loggerModule;

const char * const loggerModule = "logger";

// Node.js initialization function; called to set up the LOGGER object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, loggerModule);
}


/*****************************************************************************/
/* LOG OUTPUT JS                                                            */
/*****************************************************************************/

const char * LogOutputName = "LogOutput";

struct LogOutputJS
    : public JSWrapped2<LogOutput, LogOutputJS, LogOutputName,
                        loggerModule, true> {

    LogOutputJS()
    {
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            throw ML::Exception("can't construct a LogOutputJS");
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "logMessage", logMessage);
        NODE_SET_PROTOTYPE_METHOD(t, "close", close);
        NODE_SET_PROTOTYPE_METHOD(t, "stats", stats);
        NODE_SET_PROTOTYPE_METHOD(t, "clearStats", clearStats);
    }

    static Handle<v8::Value>
    logMessage(const Arguments & args)
    {
        try {
            if (args[0]->IsArray()) {
                vector<string> values = getArg(args, 0, "values");
                ostringstream ss;
                std::copy(values.begin() + 1, values.end(),
                          std::ostream_iterator<string>(ss, "\t"));
                getShared(args)
                    ->logMessage(values.at(0), ss.str());
            }
            else if (args[1]->IsArray()) {
                string channel = getArg(args, 0, "channel");
                vector<string> values = getArg(args, 1, "message");
                ostringstream ss;
                std::copy(values.begin(), values.end(),
                          std::ostream_iterator<string>(ss, "\t"));
                getShared(args)
                    ->logMessage(channel, ss.str());
            }
            else {
                getShared(args)
                    ->logMessage(getArg(args, 0, "channel"),
                                 getArg(args, 1, "message"));
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    close(const Arguments & args)
    {
        try {
            cerr << "doing output close" << endl;
            getShared(args)->close();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    stats(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)->stats());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    clearStats(const Arguments & args)
    {
        try {
            getShared(args)->clearStats();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<LogOutput>
from_js(const JSValue & value, std::shared_ptr<LogOutput> *)
{
    return LogOutputJS::fromJS(value);
}

LogOutput *
from_js(const JSValue & value, LogOutput **)
{
    return LogOutputJS::fromJS(value).get();
}


/*****************************************************************************/
/* JS OUTPUT                                                                 */
/*****************************************************************************/

class JSOutputJS;

/** Class that outputs messages via a javascript function. */

struct JSOutput : public LogOutput {

    JSOutput()
        : wrapper(0)
    {
    }

    virtual ~JSOutput()
    {
    }

    SlotT<void (std::string, std::string)> logMessageSlot;
    JSOutputJS * wrapper;

    virtual void logMessage(const std::string & channel,
                            const std::string & message)
    {
        //cerr << "logMessage JS " << channel << endl;

        auto cb = [=] () {
            this->logMessageSlot.call(channel, message);
        };
        callInJsContext(cb);
    }

    SlotT<void ()> closeSlot;

    virtual void close()
    {
        if (closeSlot.isEmpty()) return;
        //cerr << "logMessage JS " << channel << endl;

        auto cb = [=] () {
            this->closeSlot.call();
        };
        callInJsContext(cb);
    }
};

RegisterJsOps<void (std::string, std::string)> reg_logMessage;
RegisterJsOps<void (std::string)> reg_fileOpen;
RegisterJsOps<void (std::string, std::size_t)> reg_fileWrite;


/*****************************************************************************/
/* JS OUTPUT JS                                                              */
/*****************************************************************************/

const char * JSOutputName = "JSOutput";

struct JSOutputJS
    : public JSWrapped3<JSOutput, JSOutputJS, LogOutputJS, JSOutputName,
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

        registerRWProperty(&JSOutput::logMessageSlot, "logMessage");
        registerRWProperty(&JSOutput::closeSlot, "close");
    }
};


/*****************************************************************************/
/* FILE OUTPUT JS                                                            */
/*****************************************************************************/

const char * FileOutputName = "FileOutput";

struct FileOutputJS
    : public JSWrapped3<FileOutput, FileOutputJS, LogOutputJS, FileOutputName,
                        loggerModule, true> {

    FileOutputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<FileOutput> & output
                          = std::shared_ptr<FileOutput>())
    {
        wrap(This, output);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            if (!args[0]->IsUndefined()) {
                new FileOutputJS
                    (args.This(),
                     std::shared_ptr<FileOutput>
                     (new FileOutput
                      (getArg<string>(args, 0, "", "filename"))));
            }
            else {
                new FileOutputJS
                    (args.This(),
                     std::shared_ptr<FileOutput>
                     (new FileOutput()));
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "open", open);

        registerAsyncCallback(&FileOutput::onPreFileOpen,
                              "preFileOpen");
        registerAsyncCallback(&FileOutput::onPostFileOpen,
                              "postFileOpen");
        registerAsyncCallback(&FileOutput::onPreFileClose,
                              "preFileClose");
        registerAsyncCallback(&FileOutput::onPostFileClose,
                              "postFileClose");
        registerAsyncCallback(&FileOutput::onFileWrite,
                              "onFileWrite");

    }

    static Handle<v8::Value>
    open(const Arguments & args)
    {
        try {
            getShared(args)
                ->open(getArg(args, 0, "filename"),
                       getArg(args, 1, "", "compression"),
                       getArg(args, 2, -1, "compressionLevel"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};


/*****************************************************************************/
/* ROTATING FILE OUTPUT JS                                                   */
/*****************************************************************************/

const char * RotatingFileOutputName = "RotatingFileOutput";

struct RotatingFileOutputJS
    : public JSWrapped3<RotatingFileOutput, RotatingFileOutputJS,
                        LogOutputJS, RotatingFileOutputName,
                        loggerModule, true> {

    RotatingFileOutputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<RotatingFileOutput> & output
                          = std::shared_ptr<RotatingFileOutput>())
    {
        wrap(This, output);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new RotatingFileOutputJS
                (args.This(),
                 std::shared_ptr<RotatingFileOutput>
                 (new RotatingFileOutput()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
    
    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "open", open);

        NODE_SET_METHOD(t, "findPeriod", findPeriod);
        NODE_SET_METHOD(t, "filenameFor", filenameFor);
        NODE_SET_METHOD(t, "parsePeriod", parsePeriod);

        registerAsyncCallback(&RotatingFileOutput::onBeforeLogRotation,
                              "beforeLogRotation");
        registerAsyncCallback(&RotatingFileOutput::onAfterLogRotation,
                              "afterLogRotation");
        registerAsyncCallback(&RotatingFileOutput::onPreFileOpen,
                              "preFileOpen");
        registerAsyncCallback(&RotatingFileOutput::onPostFileOpen,
                              "postFileOpen");
        registerAsyncCallback(&RotatingFileOutput::onPreFileClose,
                              "preFileClose");
        registerAsyncCallback(&RotatingFileOutput::onPostFileClose,
                              "postFileClose");
        registerAsyncCallback(&RotatingFileOutput::onFileWrite,
                              "onFileWrite");

    }

    static Handle<v8::Value>
    open(const Arguments & args)
    {
        try {
            getShared(args)
                ->open(getArg(args, 0, "filenamePattern"),
                       getArg(args, 1, "period"),
                       getArg(args, 2, "", "compression"),
                       getArg(args, 3, -1, "compressionLevel"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    findPeriod(const Arguments & args)
    {
        try {
            return JS::toJS(Datacratic::findPeriod
                            (getArg(args, 0, "date"),
                             getArg(args, 1, "periodName")));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    filenameFor(const Arguments & args)
    {
        try {
            return JS::toJS(Datacratic::filenameFor
                            (getArg(args, 0, "date"),
                             getArg(args, 1, "filenamePattern")));
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    parsePeriod(const Arguments & args)
    {
        try {
            return JS::toJS(Datacratic::parsePeriod
                            (getArg(args, 0, "period")));
        } HANDLE_JS_EXCEPTIONS;
    }

};


/*****************************************************************************/
/* PUBLISH OUTPUT JS                                                         */
/*****************************************************************************/

const char * PublishOutputName = "PublishOutput";

struct PublishOutputJS
    : public JSWrapped3<PublishOutput, PublishOutputJS, LogOutputJS,
                        PublishOutputName, loggerModule, true> {

    PublishOutputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<PublishOutput> & output
                          = std::shared_ptr<PublishOutput>())
    {
        wrap(This, output);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new PublishOutputJS
                (args.This(),
                 std::shared_ptr<PublishOutput>
                 (new PublishOutput()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "bind", bind);
    }

    static Handle<v8::Value>
    bind(const Arguments & args)
    {
        try {
            getShared(args)
                ->bind(getArg(args, 0, "bindUri"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};


/*****************************************************************************/
/* REMOTE OUTPUT JS                                                          */
/*****************************************************************************/

const char * RemoteOutputName = "RemoteOutput";

struct RemoteOutputJS
    : public JSWrapped3<RemoteOutput, RemoteOutputJS, LogOutputJS,
                        RemoteOutputName, loggerModule, true> {

    RemoteOutputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<RemoteOutput> & output
                          = std::shared_ptr<RemoteOutput>())
    {
        wrap(This, output);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new RemoteOutputJS
                (args.This(),
                 std::shared_ptr<RemoteOutput>
                 (new RemoteOutput()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "connect", connect);
        NODE_SET_PROTOTYPE_METHOD(t, "shutdown", shutdown);
    }

    static Handle<v8::Value>
    connect(const Arguments & args)
    {
        try {
            getShared(args)
                ->connect(getArg(args, 0, "port"),
                          getArg(args, 1, "hostname"),
                          getArg(args, 2, 10.0, "timeout"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    shutdown(const Arguments & args)
    {
        try {
            getShared(args)->shutdown();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};


/*****************************************************************************/
/* REMOTE INPUT JS                                                           */
/*****************************************************************************/

const char * RemoteInputName = "RemoteInput";

struct RemoteInputJS
    : public JSWrapped2<RemoteInput, RemoteInputJS,
                        RemoteInputName, loggerModule, true> {

    RemoteInputJS(const v8::Handle<v8::Object> & This,
                      const std::shared_ptr<RemoteInput> & input
                          = std::shared_ptr<RemoteInput>())
    {
        wrap(This, input);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new RemoteInputJS
                (args.This(),
                 std::shared_ptr<RemoteInput>
                 (new RemoteInput()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "listen", listen);
        NODE_SET_PROTOTYPE_METHOD(t, "shutdown", shutdown);
        NODE_SET_PROTOTYPE_METHOD(t, "port", port);
    }

    static Handle<v8::Value>
    listen(const Arguments & args)
    {
        try {
            ev_ref(ev_default_loop());

            v8::Persistent<v8::Object> phandle
                = v8::Persistent<v8::Object>::New(args.This());

            auto cleanup = [=] ()
                {
                    v8::Persistent<v8::Object> handle = phandle;
                    ev_unref(ev_default_loop());
                    handle.Clear();
                    handle.Dispose();
                };

            ML::Call_Guard doCleanup(cleanup);  // cleanup on exception

            getShared(args)->listen(getArg(args, 0, -1, "port"),
                                    getArg(args, 1, "localhost", "address"),
                                    cleanup);
            
            doCleanup.clear();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    port(const Arguments & args)
    {
        try {
            return JS::toJS(getShared(args)->port());
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    shutdown(const Arguments & args)
    {
        try {
            getShared(args)->shutdown();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};


/*****************************************************************************/
/* LOGGER JS                                                                 */
/*****************************************************************************/

const char * LoggerName = "Logger";

struct LoggerJS
    : public JSWrapped2<Logger, LoggerJS, LoggerName, loggerModule, true> {

    LoggerJS(const v8::Handle<v8::Object> & This,
             const std::shared_ptr<Logger> & logger
                 = std::shared_ptr<Logger>())
    {
        wrap(This, logger);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            new LoggerJS(args.This(),
                         std::shared_ptr<Logger>(new Logger()));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);

        // Instance methods
        NODE_SET_PROTOTYPE_METHOD(t, "logTo", logTo);
        NODE_SET_PROTOTYPE_METHOD(t, "addOutput", addOutput);
        NODE_SET_PROTOTYPE_METHOD(t, "clearOutputs", clearOutputs);
        NODE_SET_PROTOTYPE_METHOD(t, "logMessage", logMessage);
        NODE_SET_PROTOTYPE_METHOD(t, "start", start);
        NODE_SET_PROTOTYPE_METHOD(t, "shutdown", shutdown);
        NODE_SET_PROTOTYPE_METHOD(t, "subscribe", subscribe);
    }

    static Handle<v8::Value>
    start(const Arguments & args)
    {
        try {
            // Make sure Node doesn't exit and we don't get GCd when the
            // event loop is running.

            ev_ref(ev_default_loop());
            v8::Persistent<v8::Object> phandle
                = v8::Persistent<v8::Object>::New(args.This());

            auto cleanup = [=] ()
                {
                    v8::Persistent<v8::Object> handle = phandle;
                    ev_unref(ev_default_loop());
                    handle.Clear();
                    handle.Dispose();
                };

            ML::Call_Guard doCleanup(cleanup);  // cleanup on exception

            getShared(args)->start(cleanup);
            
            doCleanup.clear();

            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static boost::regex getRegex(const Arguments & args,
                                 int argNum,
                                 const std::string & name)
    {
        string s = getArg(args, argNum, "", name);
        if (s == "")
            return boost::regex();
        return boost::regex(s);
    }
                                  

    static Handle<v8::Value>
    addOutput(const Arguments & args)
    {
        try {
            getShared(args)
                ->addOutput(getArg(args, 0, "output"),
                            getRegex(args, 1, "allowChannels"),
                            getRegex(args, 2, "denyChannels"),
                            getArg(args, 3, 1.0, "logProbability"));
            
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    logTo(const Arguments & args)
    {
        try {
            getShared(args)
                ->logTo(getArg(args, 0, "logUri"),
                        getRegex(args, 1, "allowChannels"),
                        getRegex(args, 2, "denyChannels"),
                        getArg(args, 3, 1.0, "logProbability"));
            
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    clearOutputs(const Arguments & args)
    {
        try {
            getShared(args)->clearOutputs();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    logMessage(const Arguments & args)
    {
        try {
            auto getEl = [&] (int el)
                {
                    return getArg<string>(args, el + 1, "logItem");
                };

            getShared(args)->logMessage(getArg(args, 0, "channel"),
                                        args.Length() - 1,
                                        getEl);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    shutdown(const Arguments & args)
    {
        try {
            getShared(args)->shutdown();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    subscribe(const Arguments & args)
    {
        try {
            getShared(args)->subscribe(getArg(args, 0, "subscribeUri"),
                                       getArg(args, 1, vector<string>(),
                                              "channels"),
                                       getArg(args, 2, "", "identity"));
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }
};

std::shared_ptr<Logger>
from_js(const JSValue & value, std::shared_ptr<Logger> *)
{
    return LoggerJS::fromJS(value);
}

Logger *
from_js(const JSValue & value, Logger **)
{
    return LoggerJS::fromJS(value).get();
}


} // namespace JS
} // namespace Datacratic
