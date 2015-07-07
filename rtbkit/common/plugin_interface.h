/* plugin_interface.h
   Flavio Moreira, 15 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#pragma once

#include "rtbkit/common/plugin_table.h"
#include <string>

namespace RTBKIT {

namespace details {
    template<typename Plugin>
    struct has_factory {
        template<typename T>
        static std::true_type test(typename T::Factory* = 0);

        template<typename T>
        static std::false_type test(...);

        static constexpr bool value
            = std::is_same<decltype(test<Plugin>(nullptr)), std::true_type>::value;
    };
}

template<class T>
struct PluginInterface
{
  static_assert(details::has_factory<T>::value, "The plugin must provide a Factory type");

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

