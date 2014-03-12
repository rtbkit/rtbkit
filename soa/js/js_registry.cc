/* js_registry.cc
   Jeremy Barnes, 27 July 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Registry for Javascript templates.
*/

#include "js_registry.h"
#include "jml/arch/exception.h"
#include <iostream>
#include "js_utils.h"
#include "jml/arch/backtrace.h"

using namespace std;
using namespace v8;


namespace Datacratic {
namespace JS {


/*****************************************************************************/
/* REGISTRY                                                                  */
/*****************************************************************************/

Registry::
Registry()
    : num_uninitialized(0)
{
}

Registry::
~Registry()
{
    // Close everything so that we can garbage collect them
    templates.clear();
}

const v8::Persistent<v8::FunctionTemplate> &
Registry::
operator [] (const std::string & name) const
{
    auto it = templates.find(name);
    if (it == templates.end())
        throw ML::Exception("didn't find required template "
                            + name);

    return it->second.tmpl;
}

void
Registry::
import(const Registry & other, const std::string & name)
{
    if (templates.count(name))
        throw ML::Exception("attempt to import JS template "
                            + name + " a second time");
    
    auto it = other.templates.find(name);
    if (it == other.templates.end())
        throw ML::Exception("attempt to import JS template "
                            + name + " that doesn't exist");
        
    Entry entry = it->second;
    entry.imported = true;

    templates[name] = entry;
}

void
Registry::
introduce(const std::string & name,
          const std::string & module,
          const InitFunction init,
          const std::string & base)
{
    //cerr << "introduce " << name << " in module " << module
    //     << " with base " << base << endl;

    if (templates.count(name))
        throw ML::Exception("attempt to introduce JS template "
                            + name + " in module " + module
                            + " a second time");

    Entry entry;
    entry.init = init;
    entry.base = base;
    entry.module = module;

    templates[name] = entry;

    ++num_uninitialized;

    //cerr << "introduced " << name;
    //if (base != "") cerr << " based on " << base;
    //cerr << endl;
}

void
Registry::
get_to_know(const std::string & name,
            const std::string & module,
            const v8::Persistent<v8::FunctionTemplate> & tmpl,
            SetupFunction setup)
{
    if (!templates.count(name))
        throw ML::Exception("attempt to get_to_know function template "
                            + name + " without introducing first");

    Entry & entry = templates[name];

    if (entry.module != module)
        throw ML::Exception("wrong module");

    entry.tmpl = tmpl;
    entry.setup = setup;
    entry.name = name;

    templates[name] = entry;
}

void
Registry::
init(v8::Handle<v8::Object> target, const std::string & module)
{
    if (num_uninitialized != 0)
        initialize();

    for (auto it = templates.begin(), end = templates.end();  it != end;  ++it)
        if (!it->second.imported && it->second.module == module)
            it->second.setup(it->second.tmpl, it->second.name, target); 
}


v8::Local<v8::Function>
Registry::
getConstructor(const std::string & cls) const
{
    const v8::Persistent<v8::FunctionTemplate> & tmpl = (*this)[cls];
    return tmpl->GetFunction();
}

v8::Local<v8::Object>
Registry::
constructInstance(const std::string & cls, OnConstructorError e) const
{
    HandleScope scope;
    v8::Local<v8::Object> result = getConstructor(cls)->NewInstance();
    if (result.IsEmpty() && e == THROW_ON_ERROR)
        throw JSPassException();
    return scope.Close(result);
}

v8::Local<v8::Object>
Registry::
constructInstance(const std::string & cls,
                  const v8::Arguments & args,
                  OnConstructorError e) const
{
    HandleScope scope;
    int argc = args.Length();

    vector<Handle<Value> > argv(argc);
    for (unsigned i = 0;  i < argc;  ++i)
        argv[i] = args[i];

    Local<Function> constructor = getConstructor(cls);

    v8::Local<v8::Object> result
        = constructor->NewInstance(argc, &argv[0]);

    if (result.IsEmpty() && e == THROW_ON_ERROR)
        throw JSPassException();
    return scope.Close(result);
}

v8::Local<v8::Object>
Registry::
constructInstance(const std::string & cls,
                  int argc, v8::Handle<v8::Value> * argv,
                  OnConstructorError e) const
{
    v8::Local<v8::Object> result = getConstructor(cls)->NewInstance(argc, argv);
    if (result.IsEmpty() && e == THROW_ON_ERROR)
        throw JSPassException();
    return result;
}

void
Registry::
initialize()
{
    for (auto it = templates.begin(), end = templates.end();
         it != end;  ++it)
        do_initialize(it->first);

    if (num_uninitialized != 0)
        throw ML::Exception("initialize(): num_uninitialized was wrong");
}

void
Registry::
do_initialize(const std::string & name)
{
    auto it = templates.find(name);
    if (it == templates.end())
        throw ML::Exception("didn't find required template "
                            + name);

    Entry & entry = it->second;

    // Already done?  Nothing else to do
    if (!entry.tmpl.IsEmpty()) return;

    // Otherwise, first initialize its base
    if (entry.base != "")
        do_initialize(entry.base);

    // And now initialize it
    entry.init();

    if (entry.tmpl.IsEmpty())
        throw ML::Exception("wasn't initialized");

    --num_uninitialized;
}

Registry registry;


} // namespace JS
} // namespace Datacratic
