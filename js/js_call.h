/* js_call.h                                                       -*- C++ -*-
   Jeremy Barnes, 15 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Functions to allow generic calling to/from Javascript.
*/

#ifndef __js__js_call_h__
#define __js__js_call_h__

#include "js_call_fwd.h"
#include "v8.h"
#include "soa/js/js_utils.h"
#include "soa/js/js_value.h"
#include "jml/arch/exception.h"
#include <boost/function.hpp>
#include <boost/bind.hpp>

#ifdef ev_ref
#define ev_default_loop() /* empty parameter */
#endif

namespace Datacratic {
namespace JS {



/*****************************************************************************/
/* CALL FROM JS                                                              */
/*****************************************************************************/


// Given a boost::function type Fn and a TypeList of InPosition values,
// this calls the function with the JS arguments unpacked
template<typename Fn, typename List>
struct CallWithList {
};

// Implementation of that template with the List argument unpacked
template<typename Fn, typename... ArgsWithPosition>
struct CallWithList<Fn, TypeList<ArgsWithPosition...> > {
    static typename Fn::result_type
    call(const Fn & fn, const JS::JSArgs & args)
    {
        return fn(CallWithJsArgs<ArgsWithPosition>::getArgAtPosition(args)...);
    }
};

/** call a boost::function from JS, variardic version with a non-void
    return type.

    There is some funky template magic that has to happen here.  In order
    to know which argument to unpack, we need to turn our list of Args...
    (which have no position encoded) into a list of InPosition<position, arg>
    which encodes the position within the type.
*/
template<typename Return, typename... Args, int arity>
struct callfromjs<boost::function<Return (Args...)>, arity> {
    
    typedef boost::function<Return (Args...)> Fn;

    static Return
    call(const Fn & fn, const JS::JSArgs & args)
    {
        return CallWithList<Fn, typename MakeInPositionList<0, Args...>::List>
            ::call(fn, args);
    }
};

/** Specialization of the previous for the void return case. */
template<typename... Args, int arity>
struct callfromjs<boost::function<void (Args...)>, arity> {
    
    typedef boost::function<void (Args...)> Fn;

    static void
    call(const Fn & fn, const JS::JSArgs & args)
    {
        CallWithList<Fn, typename MakeInPositionList<0, Args...>::List>
            ::call(fn, args);
    }

};


/*****************************************************************************/
/* CALL TO JS                                                                */
/*****************************************************************************/

struct calltojsbase {
    calltojsbase(v8::Handle<v8::Function> fn,
                 v8::Handle<v8::Object> This)
        : params(new Params(fn, This))
    {
    }

    struct Params {
        Params(v8::Handle<v8::Function> fn,
               v8::Handle<v8::Object> This)
            : fn(v8::Persistent<v8::Function>::New(fn)),
              This(v8::Persistent<v8::Object>::New(This))
        {
        }

        ~Params()
        {
            fn.Dispose();
            This.Dispose();
        }
        
        v8::Persistent<v8::Function> fn;
        v8::Persistent<v8::Object> This;
    };

    std::shared_ptr<Params> params;
};

template<typename... Args>
struct ArgUnpacker {
};

// Implementation of that template with the List argument unpacked
template<typename First, typename... Rest>
struct ArgUnpacker<First, Rest...> {
    
    static void
    unpack(v8::Handle<v8::Value> * unpacked, First arg, Rest... rest)
    {
        *unpacked++ = JS::toJS(arg);
        ArgUnpacker<Rest...>::unpack(unpacked, rest...);
    }
};

template<>
struct ArgUnpacker<> {
    static void
    unpack(v8::Handle<v8::Value> * unpacked)
    {
    }
};


template<typename Return, typename... Args, int arity>
struct calltojs<Return (Args...), arity> : public calltojsbase {
    calltojs(v8::Handle<v8::Function> fn,
             v8::Handle<v8::Object> This)
        : calltojsbase(fn, This)
    {
    }
    
    Return operator () (Args... args) const
    {
        //if (!v8::Locker::IsLocked())
        //    throw ML::Exception("callback outside JS context");

        v8::HandleScope scope;
        JSValue result;
        {
            v8::TryCatch tc;
            v8::Handle<v8::Value> argv[arity];
            
            ArgUnpacker<Args...>::unpack(argv, args...);

            result = params->fn->Call(params->This, arity, argv);
            
            if (result.IsEmpty()) {
                if(tc.HasCaught())
                {
                    // Print JS error and stack trace
                    char msg[256];
                    tc.Message()->Get()->WriteAscii(msg, 0, 256);
                    std::cout << msg << std::endl;
                    char st_msg[2500];
                    tc.StackTrace()->ToString()->WriteAscii(st_msg, 0, 2500);
                    std::cout << st_msg << std::endl;

                    tc.ReThrow();
                    throw JSPassException();
                }
                throw ML::Exception("didn't return anything");
            }
        }

        return from_js(result, (Return *)0);
    }
};


/*****************************************************************************/
/* JSOPS                                                                     */
/*****************************************************************************/


template<typename Base, typename Fn>
struct JsOpsBase {
    typedef boost::function<Fn> Function;

    static void op(int op,
                   const void * arg1,
                   const void * arg2,
                   void * result)
    {
        if (op == 0) {
            *(v8::Handle<v8::Value> *)result
                = Base::callBoost(*(const Function *)arg1,
                                  *(const JS::JSArgs *)arg2);
            return;
        }
        else if (op == 1) {
            *(Function *)result
                = Base::asBoost(*(const v8::Handle<v8::Function> *)arg1,
                                (const v8::Handle<v8::Object> *)arg2);
        }
        else throw ML::Exception("unknown op");
    }
};

template<typename Fn,
         typename Result = typename boost::function<Fn>::result_type>
struct DefaultJsOps : public JsOpsBase<DefaultJsOps<Fn, Result>, Fn> {
    typedef typename JsOpsBase<DefaultJsOps<Fn, void>, Fn>::Function Function;

    static v8::Handle<v8::Value>
    callBoost(const Function & fn,
              const JS::JSArgs & args)
    {
        Result result
            = JS::callfromjs<Function, Function::arity>::call(fn, args);

        JS::JSValue jsresult;
        JS::to_js(jsresult, result);
        
        return jsresult;
    }
    
    static Function
    asBoost(const v8::Handle<v8::Function> & fn,
            const v8::Handle<v8::Object> * This)
    {
        v8::Handle<v8::Object> This2;
        if (!This)
            This2 = v8::Object::New();
        return JS::calltojs<Fn, Function::arity>(fn, This ? *This : This2);
    }
};

template<typename Fn>
struct DefaultJsOps<Fn, void> : public JsOpsBase<DefaultJsOps<Fn, void>, Fn> {
    typedef typename JsOpsBase<DefaultJsOps<Fn, void>, Fn>::Function Function;

    static v8::Handle<v8::Value>
    callBoost(const Function & fn,
              const JS::JSArgs & args)
    {
        JS::callfromjs<Function, Function::arity>::call(fn, args);
        return v8::Undefined();
    }
    
    static Function
    asBoost(const v8::Handle<v8::Function> & fn,
            const v8::Handle<v8::Object> * This)
    {
        v8::Handle<v8::Object> This2;
        if (!This)
            This2 = v8::Object::New();
        return JS::calltojs<Fn, Function::arity>(fn, This ? *This : This2);
    }
};

} // namespace JS

template<typename Fn>
struct RegisterJsOps {
    RegisterJsOps(JS::JSOperations ops = JS::DefaultJsOps<Fn>::op)
    {
        JS::registerJsOps(typeid(Fn), ops);
    }
};

} // namespace Datacratic

#endif /* __js__js_call_h__ */
