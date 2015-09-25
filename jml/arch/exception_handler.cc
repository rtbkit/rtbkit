/* exception_handler.cc
   Jeremy Barnes, 26 February 2008
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include <cxxabi.h>
#include <cstring>
#include <fstream>

#include "jml/compiler/compiler.h"
#include "jml/utils/environment.h"

#include "backtrace.h"
#include "demangle.h"
#include "exception.h"
#include "exception_hook.h"
#include "format.h"
#include "threads.h"


using namespace std;


namespace ML {

Env_Option<bool> TRACE_EXCEPTIONS("JML_TRACE_EXCEPTIONS", true);

__thread bool trace_exceptions = false;
__thread bool trace_exceptions_initialized = false;

void set_default_trace_exceptions(bool val)
{
    TRACE_EXCEPTIONS.set(val);
}

bool get_default_trace_exceptions()
{
    return TRACE_EXCEPTIONS;
}

void set_trace_exceptions(bool trace)
{
    //cerr << "set_trace_exceptions to " << trace << " at " << &trace_exceptions
    //     << endl;
    trace_exceptions = trace;
    trace_exceptions_initialized = true;
}

bool get_trace_exceptions()
{
    if (!trace_exceptions_initialized) {
        //cerr << "trace_exceptions initialized to = "
        //     << trace_exceptions << " at " << &trace_exceptions << endl;
        set_trace_exceptions(TRACE_EXCEPTIONS);
        trace_exceptions_initialized = true;
    }
    
    //cerr << "get_trace_exceptions returned " << trace_exceptions
    //     << " at " << &trace_exceptions << endl;

    return trace_exceptions;
}


static const std::exception *
to_std_exception(void* object, const std::type_info * tinfo)
{
    /* Check if its a class.  If not, we can't see if it's a std::exception.
       The abi::__class_type_info is the base class of all types of type
       info for types that are classes (of which std::exception is one).
    */
    const abi::__class_type_info * ctinfo
        = dynamic_cast<const abi::__class_type_info *>(tinfo);

    if (!ctinfo) return 0;

    /* The thing thrown was an object.  Now, check if it is derived from
    std::exception. */
    const std::type_info * etinfo = &typeid(std::exception);

    /* See if the exception could catch this.  This is the mechanism
    used internally by the compiler in catch {} blocks to see if
    the exception matches the catch type.

    In the case of success, the object will be adjusted to point to
    the start of the std::exception object.
    */
    void * obj_ptr = object;
    bool can_catch = etinfo->__do_catch(tinfo, &obj_ptr, 0);

    if (!can_catch) return 0;

    /* obj_ptr points to a std::exception; extract it and get the
    exception message.
    */
    return (const std::exception *)obj_ptr;
}

/** We install this handler for when an exception is thrown. */

void default_exception_tracer(void * object, const std::type_info * tinfo)
{
    //cerr << "trace_exception: trace_exceptions = " << get_trace_exceptions()
    //     << " at " << &trace_exceptions << endl;

    if (!get_trace_exceptions()) return;

    const std::exception * exc = to_std_exception(object, tinfo);

    // We don't want these exceptions to be printed out.
    if (dynamic_cast<const ML::SilentException *>(exc)) return;

    /* avoid allocations when std::bad_alloc is thrown */
    bool noAlloc = dynamic_cast<const std::bad_alloc *>(exc);

    size_t bufferSize(1024*1024);
    char buffer[bufferSize];
    char datetime[128];
    size_t totalWritten(0), written, remaining(bufferSize);

    time_t now;
    time(&now);

    struct tm lt_tm;
    strftime(datetime, sizeof(datetime), "%FT%H:%M:%SZ",
             gmtime_r(&now, &lt_tm));

    const char * demangled;
    char * heapDemangled;
    if (noAlloc) {
        heapDemangled = nullptr;
        demangled = "std::bad_alloc";
    }
    else {
        heapDemangled = char_demangle(tinfo->name());
        demangled = heapDemangled;
    }
    auto pid = getpid();
    auto tid = gettid();

    written = ::snprintf(buffer, remaining,
                         "\n"
                         "--------------------------[Exception thrown]"
                         "---------------------------\n"
                         "time:   %s\n"
                         "type:   %s\n"
                         "pid:    %d; tid: %d\n",
                         datetime, demangled, pid, tid);
    if (heapDemangled) {
        free(heapDemangled);
    }
    if (written >= remaining) {
        goto end;
    }
    totalWritten += written;
    remaining -= written;

    if (exc) {
        written = snprintf(buffer + totalWritten, remaining,
                           "what:   %s\n", exc->what());
        if (written >= remaining) {
            goto end;
        }
        totalWritten += written;
        remaining -= written;
    }

    if (noAlloc) {
        goto end;
    }

    written = snprintf(buffer + totalWritten, remaining, "stack:\n");
    if (written >= remaining) {
        goto end;
    }
    totalWritten += written;
    remaining -= written;

    written = backtrace(buffer + totalWritten, remaining, 3);
    if (written >= remaining) {
        goto end;
    }
    totalWritten += written;

    if (totalWritten < bufferSize - 1) {
        strcpy(buffer + totalWritten, "\n");
    }

end:
    cerr << buffer;

    char const * reports = getenv("ENABLE_EXCEPTION_REPORTS");
    if (!noAlloc && reports) {
        std::string path = ML::format("%s/exception-report-%s-%d-%d.log",
                                      reports, datetime, pid, tid);

        std::ofstream file(path, std::ios_base::app);
        if(file) {
            file << getenv("_") << endl;
            backtrace(file, 3);
            file.close();
        }
    }
}

} // namespace ML
