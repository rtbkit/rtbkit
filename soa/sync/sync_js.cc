/* sync_js.cc
   Jeremy Barnes, 15 June 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "v8.h"
#include "soa/js/js_registry.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/guard.h"
#include <iostream>
#include <string>
#include <unistd.h>
#include <fcntl.h>

using namespace v8;
using namespace std;
using namespace node;

namespace Datacratic {
namespace JS {

extern const char * const syncModule;
const char * const syncModule = "sync";


/*****************************************************************************/
/* OUTPUT STREAM                                                             */
/*****************************************************************************/

/** Output stream for node that (shock, horror) blocks allowing one to
    concentrate on the function not the structure of code when async isn't
    where it's at.
*/

struct OutputStream {
    OutputStream()
    {
    }

    OutputStream(const std::string & filename)
        : stream(filename), name(filename), fd(-1)
    {
    }

    OutputStream(int fd, const std::string & name)
        : name(name), fd(fd)
    {
    }

    ML::filter_ostream stream;
    std::string name;
    int fd;

    void open(const std::string & filename)
    {
        stream.open(filename);
        name = filename;
        fd = -1;
    }

    void open(int fd, const std::string & name)
    {
        stream.close();
        this->name = name;
        this->fd = fd;
    }

    void close()
    {
        stream.close();
        name = "";
    }

    void write(const std::string & str)
    {
        if (fd == -1) {
            stream << str << flush;
            if (!stream)
                throw ML::Exception("attempting to write to stream %s: %s",
                                    name.c_str(), strerror(errno));
        } else {
            throw ML::Exception("write not done");
        }
    }

    static int makeBlocking(int fd)
    {
        // Look at the blocking flags
        int oldfl = fcntl(fd, F_GETFL);
        if (oldfl == -1)
            throw ML::Exception("couldn't read flags");
        
        // If we're in blocking mode, then unset the flag
        if (oldfl & O_NONBLOCK) {
            int res = fcntl(fd, F_SETFL, oldfl & ~O_NONBLOCK);
            if (res == -1)
                throw ML::Exception("couldn't set up flags again");
        }

        return oldfl;
    }

    static void resetBlocking(int fd, int oldFlags)
    {
        // If we were in blocking mode, then reset the flag
        if (oldFlags & O_NONBLOCK) {
            int res = fcntl(fd, F_SETFL, oldFlags);
            if (res == -1)
                throw ML::Exception("couldn't set up flags again");
        }
    }

    struct MakeBlocking {
        MakeBlocking(int fd)
            : fd(fd), oldFlags(makeBlocking(fd))
        {
        }

        int fd;
        int oldFlags;

        ~MakeBlocking()
        {
            resetBlocking(fd, oldFlags);
        }
    };

    void log(const std::string & str_)
    {
        if (fd == -1) {
            stream << str_ << endl << flush;
            if (!stream)
                throw ML::Exception("attempting to write to stream %s: %s",
                                    name.c_str(), strerror(errno));
        } else {
            string str = str_ + "\n";

            MakeBlocking nbl(fd);
            
            ssize_t written = 0;

            while (written < str.length()) {
                ssize_t res = ::write(fd, str.c_str() + written,
                                      str.length() - written);
                if (res == -1 && errno == EINTR) continue;
                if (res == -1)
                    throw ML::Exception("write to fd %d: %s", fd, strerror(errno));
                written += res;
            }

        }
    }
};


/*****************************************************************************/
/* OUTPUT STREAM JS                                                          */
/*****************************************************************************/

const char * OutputStreamName = "OutputStream";

struct OutputStreamJS
    : public JSWrapped2<OutputStream, OutputStreamJS, OutputStreamName,
                        syncModule> {
    
    OutputStreamJS(v8::Handle<v8::Object> This,
              const std::shared_ptr<OutputStream> & bid
              = std::shared_ptr<OutputStream>())
    {
        HandleScope scope;
        wrap(This, bid);
    }

    static Handle<v8::Value>
    New(const Arguments & args)
    {
        try {
            if (args.Length() == 0)
                new OutputStreamJS(args.This(), ML::make_std_sp(new OutputStream()));
            else if (args[0]->IsNumber()) {
                new OutputStreamJS
                    (args.This(),
                     ML::make_std_sp
                     (new OutputStream(getArg<int>(args, 0, "fd"),
                                       getArg<string>(args, 1, "", "name"))));
            }
            else {
                new OutputStreamJS
                    (args.This(),
                     ML::make_std_sp
                     (new OutputStream(getArg<string>(args, 0, "filename"))));
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static void
    Initialize()
    {
        Persistent<FunctionTemplate> t = Register(New);
        NODE_SET_PROTOTYPE_METHOD(t, "open", open);
        NODE_SET_PROTOTYPE_METHOD(t, "close", close);
        NODE_SET_PROTOTYPE_METHOD(t, "flush", close);
        NODE_SET_PROTOTYPE_METHOD(t, "write", write);
        NODE_SET_PROTOTYPE_METHOD(t, "log", log);
        NODE_SET_PROTOTYPE_METHOD(t, "toString", toString);
        NODE_SET_PROTOTYPE_METHOD(t, "inspect", toString);
    }

    static Handle<v8::Value>
    open(const Arguments & args)
    {
        try {
            if (args[0]->IsNumber()) {
                getShared(args)->open(getArg<int>(args, 0, "fd"),
                                      getArg<string>(args, 1, "", "name"));
            }
            else {
                getShared(args)->open(getArg<string>(args, 0, "filename"));
            }
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    close(const Arguments & args)
    {
        try {
            getShared(args)->close();
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static std::string getMessage(const Arguments & args,
                                  char separator)
    {
        std::string result;
        result.reserve(128);
        for (unsigned i = 0;  i < args.Length();  ++i) {
            if (i != 0 && separator != 0) result += separator;
            result += cstr(args[i]);
        }
        return result;
    }

    static Handle<v8::Value>
    write(const Arguments & args)
    {
        try {
            string message = getMessage(args, 0);
            getShared(args)->write(message);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    log(const Arguments & args)
    {
        try {
            string message = getMessage(args, ' ');
            //cerr << "logging " << message << endl;
            getShared(args)->log(message);
            return args.This();
        } HANDLE_JS_EXCEPTIONS;
    }

    static Handle<v8::Value>
    toString(const Arguments & args)
    {
        try {
            OutputStream & stream = *getShared(args);
            if (stream.fd == -1) {
                if (stream.stream)
                    return JS::toJS(ML::format("[OutputStream for %s]",
                                               stream.name.c_str()));
                else {
                    return JS::toJS(ML::format("[OutputStream for %s failed "
                                               "with status %s]",
                                               stream.name.c_str(),
                                               stream.stream.status().c_str()));
                }
            } else {
                return JS::toJS(ML::format("[OutputStream on fd %d for %s]",
                                           stream.fd,
                                           stream.name.c_str()));
                
            }
        } HANDLE_JS_EXCEPTIONS;
    }
};


static Handle<v8::Value>
makeSynchronous(const Arguments & args)
{
    try {
        // Node, you can pry my synchronous standard streams out of my cold,
        // synchronous hands
        OutputStream::makeBlocking(0);
        OutputStream::makeBlocking(1);
        OutputStream::makeBlocking(2);
        return v8::Undefined();
    } HANDLE_JS_EXCEPTIONS;
}


/*****************************************************************************/
/* INITIALIZATION                                                            */
/*****************************************************************************/

// Node.js initialization function; called to set up the sync object
extern "C" void
init(Handle<v8::Object> target)
{
    std::ios::sync_with_stdio(false);
    
    Datacratic::JS::registry.init(target, syncModule);

    static Persistent<FunctionTemplate> ms
        = v8::Persistent<FunctionTemplate>::New
        (v8::FunctionTemplate::New(makeSynchronous));

    target->Set(String::NewSymbol("makeSynchronous"), ms->GetFunction());

    v8::Handle<v8::Object> COUT
        = OutputStreamJS::toJS(ML::make_std_sp(new OutputStream(1, "STDOUT")));
    v8::Handle<v8::Object> CERR
        = OutputStreamJS::toJS(ML::make_std_sp(new OutputStream(2, "STDERR")));

    target->Set(String::NewSymbol("cout"), COUT);
    target->Set(String::NewSymbol("cerr"), CERR);

    // Let's be nasty and inject some things into the global space
    v8::Local<v8::Object> global
        = v8::Context::GetCurrent()->Global();

    global->Set(String::NewSymbol("cout"), COUT);
    global->Set(String::NewSymbol("cerr"), CERR);
}


} // namespace JS
} // namespace Datacratic
