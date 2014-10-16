/* slot_test.cc
   Jeremy Barnes, 4 November 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Test the slot class.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include "jml/arch/exception_handler.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/string_functions.h"

#include "soa/sigslot/slot.h"
#include "soa/sigslot/slot_impl_js.h"

#include "soa/js/js_call.h"
#include "soa/js/js_utils.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

template<typename T>
void printSize()
{
    cerr << "sizeof(" << ML::type_name<T>() << ") = "
         << sizeof(T) << endl;
}

RegisterJsOps<int (int)> reg4;

BOOST_AUTO_TEST_CASE( test_sizes )
{
    printSize<boost::function<void ()> >();
    printSize<boost::function<int ()> >();
    printSize<boost::function<int (int)> >();
    printSize<boost::function<int (int, int)> >();
    printSize<boost::function<int (int, int, int)> >();
    printSize<Slot>();
}

BOOST_AUTO_TEST_CASE( test_c_calling_c )
{
    typedef int (Fn1) (int);
    typedef int (Fn2) (unsigned);

    auto f1 = [] (int val) -> int { return val; };
    auto f2 = [] (unsigned val) -> int { return 2 * val; };

    Slot n1 = slot<Fn1>(f1);
    BOOST_CHECK_EQUAL(n1.call<Fn1>(1), 1);

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(n1.call<Fn2>(1), ML::Exception);
    }

    n1 = slot<Fn1>(f2);
    BOOST_CHECK_EQUAL(n1.call<Fn1>(1), 2);
}

BOOST_AUTO_TEST_CASE( test_js_calling_c )
{
    typedef int (Fn1) (int);
    typedef int (Fn2) (unsigned);

    auto f1 = [] (int val) -> int { return val; };
    auto f2 = [] (unsigned val) -> int { return 2 * val; };

    Slot n1 = slot<Fn1>(f1);
    BOOST_CHECK_EQUAL(n1.call<Fn1>(1), 1);

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(n1.call<Fn2>(1), ML::Exception);
    }

    n1 = slot<Fn1>(f2);
    BOOST_CHECK_EQUAL(n1.call<Fn1>(1), 2);

    using namespace v8;

    HandleScope handle_scope;
    Persistent<Context> context = Context::New();
    Context::Scope context_scope(context);
    
    // Goal: call the slot from Javascript
    int argc = 1;
    Handle<Value> argv[argc];
    argv[0] = JS::toJS(1);

    BOOST_CHECK_EQUAL(JS::cstr(n1.call(context->Global(), argc, argv)), "2");
    n1 = slot<Fn1>(f1);
    BOOST_CHECK_EQUAL(JS::cstr(n1.call(context->Global(), argc, argv)), "1");
  
    context.Dispose();
}

BOOST_AUTO_TEST_CASE( test_c_calling_js )
{
    using namespace v8;

    HandleScope handle_scope;
    Persistent<Context> context = Context::New();
    Context::Scope context_scope(context);
    
    v8::Handle<v8::Function> fn1
        = JS::getFunction("function f1(val) { return val; };  f1;");
    v8::Handle<v8::Function> fn2
        = JS::getFunction("function f2(val) { return val * 2; };  f2;");

    typedef int (Fn) (int);

    Slot n1(fn1);

    BOOST_CHECK_EQUAL(n1.call<Fn>(1), 1);

    n1 = Slot(fn2);
    
    BOOST_CHECK_EQUAL(n1.call<Fn>(1), 2);

    context.Dispose();
}

BOOST_AUTO_TEST_CASE( test_js_calling_js )
{
    using namespace v8;

    HandleScope handle_scope;
    Persistent<Context> context = Context::New();
    Context::Scope context_scope(context);

    v8::Handle<v8::Function> fn1
        = JS::getFunction("function f1(val) { return val; };  f1;");
    v8::Handle<v8::Function> fn2
        = JS::getFunction("function f2(val) { return val * 2; };  f2;");

    Slot n1(fn1);
    
    // Goal: call the slot from Javascript
    int argc = 1;
    Handle<Value> argv[argc];
    argv[0] = JS::toJS(1);

    BOOST_CHECK_EQUAL(JS::cstr(n1.call(context->Global(), argc, argv)), "1");
    n1 = Slot(fn2);
    BOOST_CHECK_EQUAL(JS::cstr(n1.call(context->Global(), argc, argv)), "2");
  
    context.Dispose();
}

RegisterJsOps<void (void)> reg5;

BOOST_AUTO_TEST_CASE( test_conversion_to_function )
{
    int i = 0;

    boost::function<void ()> fn;

    SlotT<void ()> slot([&] () { i += 1; });

    fn = slot;

    fn();

    BOOST_CHECK_EQUAL(i, 1);

    slot();

    BOOST_CHECK_EQUAL(i, 2);

    Slot slot2 = slot;

    slot2.call<void ()>();

    //fn = slot2;

    BOOST_CHECK_EQUAL(i, 3);
}

BOOST_AUTO_TEST_CASE( dispose )
{
    v8::V8::Dispose();
}

