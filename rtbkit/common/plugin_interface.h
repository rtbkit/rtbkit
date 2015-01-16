/* plugin_interface.h
   Flavio Moreira, 15 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#pragma once

#import "rtbkit/common/plugin_table.h"
#import <string>

namespace RTBKIT {

template<class T>
struct PluginInterface
{
  static void registerPlugin(const std::string& name,
			     typename T::Factory functor);
  static typename T::Factory& getPlugin(const std::string& name);
};

template<class T>
void PluginInterface<T>::registerPlugin(const std::string& name,
					       typename T::Factory functor)
{
  PluginTable<typename T::Factory>::instance().registerPlugin(name, functor);
}

template<class T>
typename T::Factory& PluginInterface<T>::getPlugin(const std::string& name)
{
  return PluginTable<typename T::Factory>::instance().getPlugin(name, T::libNameSufix());
}


};

