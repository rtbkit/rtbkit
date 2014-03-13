
#include "exception_hook.h"
#include "demangle.h"
#include <cxxabi.h>
#include "backtrace.h"
#include "jml/compiler/compiler.h"
#include <execinfo.h>
#include <pthread.h>

using namespace std;


namespace ML {

void (*exception_tracer) (void *, const std::type_info *) JML_WEAK_FN = 0;

__thread BacktraceInfo * current_backtrace = 0;

void cleanup_current_backtrace(void * arg)
{
    BacktraceInfo * p = (BacktraceInfo *)arg;
    delete p;
}

void trace_exception_node(void * object, const std::type_info * tinfo)
{
    if (!current_backtrace) {
        current_backtrace = new BacktraceInfo();
        //pthread_cleanup_push(&cleanup_current_backtrace, current_backtrace);
    }

    size_t size = ::backtrace (current_backtrace->frames, 50);
    current_backtrace->size = size;
    current_backtrace->type = tinfo;

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
            current_backtrace->message = exc->what();
        }
    }
}

namespace {
struct Install_Handler {
    Install_Handler()
    {
        exception_tracer = trace_exception_node;
    }
    ~Install_Handler()
    {
        if (exception_tracer == trace_exception_node)
            exception_tracer = 0;
    }
} install_handler;

} // file scope

} // namespace ML
