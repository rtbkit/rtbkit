/* exception_handler.cc
   Jeremy Barnes, 26 February 2008
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "exception_hook.h"
#include "demangle.h"
#include <cxxabi.h>
#include "backtrace.h"
#include "jml/compiler/compiler.h"
#include "jml/utils/environment.h"

using namespace std;


namespace ML {

void (*exception_tracer) (void *, const std::type_info *) JML_WEAK_FN = 0;

Env_Option<bool> TRACE_EXCEPTIONS("JML_TRACE_EXCEPTIONS", true);

__thread bool trace_exceptions = false;
__thread bool trace_exceptions_initialized = false;

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


/** We install this handler for when an exception is thrown. */

void trace_exception(void * object, const std::type_info * tinfo)
{
    //cerr << "trace_exception: trace_exceptions = " << get_trace_exceptions()
    //     << " at " << &trace_exceptions << endl;

    if (!get_trace_exceptions()) return;

    cerr << endl;
    cerr << "----------------- Exception thrown ------------------------"
         << endl;
    std::cerr << "type:   " << demangle(tinfo->name()) << endl;

    /* Check if its a class.  If not, we can't see if it's a std::exception.
       The abi::__class_type_info is the base class of all types of type
       info for types that are classes (of which std::exception is one).
    */
    const abi::__class_type_info * ctinfo
        = dynamic_cast<const abi::__class_type_info *>(tinfo);

    if (ctinfo) {
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

        if (can_catch) {
            /* obj_ptr points to a std::exception; extract it and get the
               exception message.
            */
            const std::exception * exc = (const std::exception *)obj_ptr;
            cerr << "what:   " << exc->what() << endl;
        }
    }

    cerr << "stack:" << endl;
    backtrace(cerr, 3);
    cerr << endl;
}

namespace {
struct Install_Handler {
    Install_Handler()
    {
        //cerr << "installing exception tracer" << endl;
        exception_tracer = trace_exception;
    }
    ~Install_Handler()
    {
        if (exception_tracer == trace_exception)
            exception_tracer = 0;
    }
} install_handler;

} // file scope

} // namespace ML
