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
#include <exception>
#include <mutex>
#include <string>
#include <dlfcn.h>
#include <iostream>

#include "jml/arch/spinlock.h"
#include "jml/arch/exception.h"

namespace RTBKIT {

template<class T>
struct PluginTable
{
public:
  static PluginTable& instance();

  // allow plugins to register themselves from their Init/AtInit
  // all legacy plugins use this.
  //template <class T>
  void registerPlugin(const std::string& name, T& functor);

  // get a plugin factory this is a generic version, but requires that
  // the library suffix that contains this plugin to be provided
  //template <class T>
  T& getPlugin(const std::string& name, const std::string& libSufix);

  // destructor
  ~PluginTable(){};
  // delete copy constructors
  PluginTable(PluginTable&) = delete;
  // delete assignement operator
  PluginTable& operator=(const PluginTable&) = delete;
  

private:
 
  // data
  // -----
  std::unordered_map<std::string, T> table;

  // lock
  ML::Spinlock lock;

  // default constructor can only be accessed by the class itself
  // used by the statc method instance
  PluginTable(){};  
  
  // load library
  void loadLib(const std::string& path);

};


// inject a new "factory" (functor) - called from the plugin dll
template <class T>
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
  std::lock_guard<ML::Spinlock> guard(lock);
  table.insert(element);
}

 
// get the functor from the name
template <class T>
T&
PluginTable<T>::getPlugin(const std::string& name, const std::string& libSufix)
{
  // some safeguards...
  if (name.empty()) {
    throw ML::Exception("'name' parameter cannot be empty");
  }

  // get the plugin or/and load the lib
  for (int i=0; i<2; i++)
  {
    // check if it already exists
    {
      std::lock_guard<ML::Spinlock> guard(lock);
      auto iter = table.find(name);
      if(iter != table.end())
      {
	return iter->second;
      }
    }
    
    if (i == 0) // try to load it
    {
      // since it was not found we have to try to load the library
      std::string path = "lib" + name + "_" + libSufix + ".so";
      loadLib(path);
    } // we can add alternative forms of plugin load here

    ///////////
    // now hopefully the plugin is loaded
    // and we can load it in the next loop
  }  

  // else: getting the functor fails
  throw ML::Exception("couldn't get requested plugin");
}


// get singleton instance
template<class T>
PluginTable<T>&
PluginTable<T>::instance()
{
  static PluginTable<T> singleton;
  return singleton;
}

// loads a dll
template<class T>
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

