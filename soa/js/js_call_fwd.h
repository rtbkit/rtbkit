/* js_call_fwd.h                                                   -*- C++ -*-
   Jeremy Barnes, 16 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Forward definitions for the js_call infrastructure.
*/

#ifndef __js__js_call_fwd_h__
#define __js__js_call_fwd_h__

#include <typeinfo>
#include <boost/function.hpp>

namespace v8 {
struct Arguments;
template<typename T> struct Handle;
template<typename T> struct Persistent;
template<typename T> struct Local;
struct Value;
struct Function;
struct Object;

} // namespace v8

namespace Datacratic {
namespace JS {

template<typename Fn, int arity = Fn::arity>
struct callfromjs;

template<typename Fn, int arity = boost::function<Fn>::arity>
struct calltojs;


struct JSArgs;

/** Operations function for Javascript.  Defined in js_call.h.
    
    Operation 0: call boost function
        var1 = pointer to function
        var2 = pointer to JSArgs instance
        var3 = pointer to v8::Handle<v8::Value> for result

    Operation 1: convert to boost::function
        var1 = pointer to v8::Handle<v8::Function> for function
        var2 = pointer to v8::Handle<v8::Object> for This
        var3 = pointer to boost::function for result
*/
typedef void (*JSCallsBoost) (int op,
                              const boost::function_base & fn,
                              const JS::JSArgs & args,
                              v8::Handle<v8::Value> & result);

typedef void (*JSAsBoost) (int op,
                           const v8::Persistent<v8::Function> & fn,
                           const v8::Handle<v8::Object> & This,
                           boost::function_base & result);

// This is compatible with the previous two
typedef void (*JSOperations) (int op,
                             const void * arg1,
                             const void * arg2,
                             void * result);

JSOperations getOps(const std::type_info & fntype);
void registerJsOps(const std::type_info & type,
                   JS::JSOperations ops);
    


} // namespace JS
} // namespace Datacratic

#endif /* __js__js_call_fwd_h__ */

