/* utils_js.cc
   26 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

*/

#include "v8.h"
#include "utils_js.h"
#include "soa/js/js_registry.h"
#include "soa/js/js_wrapped.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/guard.h"
#include <iostream>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

using namespace v8;
using namespace std;
using namespace node;

namespace Datacratic {
namespace JS {

const char * const utilsModule = "sync_utils";


/*****************************************************************************/
/* FOR EACH LINE                                                             */
/*****************************************************************************/

static Handle<v8::Value>
forEachLine(const Arguments & args)
{
    try {
        string filename = getArg(args, 0, "filename");
        v8::Local<v8::Function> fn = getArg(args, 1, "callback");
        //cerr << "fn = " << cstr(fn) << endl;
        
        bool hadError = false;

        ML::filter_istream stream(filename);
        if (stream.fail() || stream.bad())
            throw ML::Exception("couldn't open filename %s: %s",
                                filename.c_str(), strerror(errno));
        
        auto onLine = [&] (const std::string & line, int lineNum)
            {
                v8::HandleScope scope;
                int argc = 2;
                v8::Handle<v8::Value> argv[argc];
                argv[0] = JS::toJS(line);
                argv[1] = JS::toJS(lineNum);
                
                v8::Handle<v8::Value> result
                    = fn->Call(args.This(), argc, argv);
                
                // Exception?
                if (result.IsEmpty()) hadError = true;
            };

        int numLines = 0;

        while (stream && !hadError) {
            string line;
            std::getline(stream, line);

            if (line == "" && stream.gcount() == 0 && stream.eof())
                break;

            if (line == "" && stream.fail())
                throw ML::Exception("stream in failed mode");

            if (line == "" && stream.bad())
                throw ML::Exception("steam is bad");

            onLine(line, numLines);
            ++numLines;
        }

        return JS::toJS(numLines);
    } HANDLE_JS_EXCEPTIONS;
}

static Handle<v8::Value>
runCmd(const Arguments & args)
{
    try {
        string cmdStr = getArg(args, 0, "cmd");
        const char * cmd = cmdStr.c_str();
        FILE* pipe = popen(cmd, "r");
        if (!pipe)
            return JS::toJS("ERROR");
            char buffer[128];
        std::string result = "";
        while(!feof(pipe)) {
            if(fgets(buffer, 128, pipe) != NULL)
                result += buffer;
        }
        pclose(pipe);

        return JS::toJS(result);

/*        string filename = getArg(args, 0, "filename");
        int code = system(filename.c_str());
        return JS::toJS(code);*/
    }  HANDLE_JS_EXCEPTIONS;
}



/*****************************************************************************/
/* INITIALIZATION                                                            */
/*****************************************************************************/

// Node.js initialization function; called to set up the utils object
extern "C" void
init(Handle<v8::Object> target)
{
    Datacratic::JS::registry.init(target, utilsModule);

    static Persistent<FunctionTemplate> fel
        = v8::Persistent<FunctionTemplate>::New
        (v8::FunctionTemplate::New(forEachLine));
    target->Set(String::NewSymbol("forEachLine"), fel->GetFunction());

    static Persistent<FunctionTemplate> rc
        = v8::Persistent<FunctionTemplate>::New
        (v8::FunctionTemplate::New(runCmd));
    target->Set(String::NewSymbol("runCmd"), rc->GetFunction());
}


} // namespace JS
} // namespace utils
