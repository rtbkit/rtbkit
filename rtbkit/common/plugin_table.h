/* plugin_table.h
   Flavio Moreira, 15 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#pragma once

#include <string>
#include <unordered_map>
#include <boost/any.hpp>
#include <typeinfo>
#include <functional>
#include <mutex>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#include "jml/arch/exception.h"
#include "jml/utils/string_functions.h" // ML::split()

namespace RTBKIT {

template<typename T>
struct PluginTable
{
public:

    static PluginTable& instance();

    // allow plugins to register themselves from their Init/AtInit
    // all legacy plugins use this.
    //template <typename T>
    void registerPlugin(const std::string& name, T& functor);

    // get a plugin factory this is a generic version, but requires that
    // the library suffix that contains this plugin to be provided
    //template <typename T>
    T& getPlugin(const std::string& name, const std::string& libSufix);

    // returns a list of plugin names
    std::vector<std::string> getNames() const;

    // delete copy constructors
    PluginTable(PluginTable&) = delete;

    // delete assignement operator
    PluginTable& operator=(const PluginTable&) = delete;

private:
 
    // data
    // -----
    std::unordered_map<std::string, T> table;

    // lock
    mutable std::mutex mu;

    // default constructor can only be accessed by the class itself
    // used by the static method instance
    PluginTable(){};

    // load library by calling dlopen
    void loadLib(const std::string& path);

};


// inject a new "factory" (functor) - called from the plugin dll
template <typename T>
void
PluginTable<T>::registerPlugin(const std::string& name, T& functor)
{
    // some safeguards...
    if (name.empty()) {
        throw ML::Exception("'name' parameter cannot be empty");
    }

    // assemble the element
    auto element = std::pair<std::string, T>(name, functor);

    // lock and write
    std::lock_guard<std::mutex> guard(mu);
    table.insert(element);
}

 
// get the functor from the name
template <typename T>
T&
PluginTable<T>::getPlugin(const std::string& name, const std::string& libSuffix)
{
    using std::vector;
    using std::string;

    // some safeguards...
    if (name.empty()) {
        throw ML::Exception("'name' parameter cannot be empty");
    }

    string libPath, plugName;
    vector<string> libAndPlug = ML::split(name,'.');

    if (libAndPlug.size() > 2)
        throw ML::Exception("Error in %s: Library and plugin name can only "
           "contain two dot separated parameters and no more",name.c_str());

    if (libAndPlug.size() == 2) {
        plugName = libAndPlug[1];
        libPath = "lib" + libAndPlug[0] + ".so";
    } else {
        plugName = name;
        libPath = "lib" + name + "_" + libSuffix + ".so";
    }

    // check if it already exists
    {
        std::lock_guard<std::mutex> guard(mu);
        auto iter = table.find(plugName);
        if(iter != table.end())
            return iter->second;
    }

    // since it was not found we have to try to load the library
    // loadlib calls opendl function which in its turn automatically
    // instantiates all global variables inside .so libraries.
    // Hence, in each plugin library, there should be  a global
    // structure whose default constructor call registerPlugin function
    loadLib(libPath);

    // check if it is created
    {
        std::lock_guard<std::mutex> guard(mu);
        auto iter = table.find(plugName);
        if(iter != table.end())
            return iter->second;
    }

    // else: getting the functor fails
    throw ML::Exception("couldn't get requested plugin");
}

// returns a vector representing all objects in the plugin
template <typename T>
std::vector<std::string> PluginTable<T>::getNames() const
{
    std::lock_guard<std::mutex> guard(mu);

    std::vector<std::string> list;
    list.reserve(table.size());
    for(const auto& ele: table){
        list.push_back(ele.first);
    }

    return list;
}

// get singleton instance
template<typename T>
PluginTable<T>&
PluginTable<T>::instance()
{
    static PluginTable<T> singleton;
    return singleton;
}

// loads a dll
template<typename T>
void
PluginTable<T>::loadLib(const std::string& path)
{
    // some safeguards...
    if (path.empty()) {
        throw ML::Exception("'path' parameter cannot be empty");
    }

    // load lib
    void * handle = dlopen(path.c_str(), RTLD_NOW);

    if (!handle) {
    std::cerr << dlerror() << std::endl;
        throw ML::Exception("couldn't load library from %s", path.c_str());
    }
}

}; // namespace RTBKIT
