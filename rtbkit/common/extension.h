/* extension.h
   Mathieu Stefani, 14 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   A template system to extend some core components of RTBKIT
*/

#pragma once

#include "jml/arch/exception.h"
#include "rtbkit/common/plugin_interface.h"
#include "soa/jsoncpp/value.h"
#ifndef NO_EXTENSION_SAFE_CAST
#include "soa/utils/fnv_hash.h"
#include <typeinfo>
#endif
#include <string>
#include <unordered_map>
#include <type_traits>
#include <memory>

namespace RTBKIT {

struct Extension {
    typedef std::function<std::unique_ptr<Extension>()> Factory;

    static std::string libNameSufix() { return "extension"; }

    virtual const char* name() const = 0;
    virtual void parse(const Json::Value& value) = 0;
    virtual Json::Value toJson() const = 0;
#ifndef NO_EXTENSION_SAFE_CAST
    virtual uint64_t hash() const = 0;
#endif

};

#ifndef NO_EXTENSION_SAFE_CAST
  #define NAME(extName) \
      static constexpr const char* Name = extName;                   \
      static constexpr uint64_t Hash = Datacratic::fnv_hash64(Name); \
      const char* name() const { return Name; }                      \
      uint64_t hash() const { return Hash; }
#else
  #define NAME(extName)                            \
      static constexpr const char* Name = extName; \
      const char* name() const { return Name; }
#endif

template<typename Ext>
struct IsExtension {
    template<typename E>
    static auto test(E *) -> decltype(E::Name, std::true_type());

    template<typename E>
    static auto test(...) -> std::false_type;

    static constexpr bool value =
        std::is_base_of<Extension, Ext>::value &&
        std::is_same<decltype(test<Ext>(nullptr)), std::true_type>::value;
};

#ifndef NO_EXTENSION_SAFE_CAST
template<typename To>
typename std::enable_if<
           IsExtension<To>::value, std::shared_ptr<To>
         >::type
extension_cast(const std::shared_ptr<Extension>& from) {
    return static_cast<To *>(0)->Hash == from->hash() ?
        std::static_pointer_cast<To>(from) : nullptr;
}

template<typename To>
typename std::enable_if<
           IsExtension<To>::value, std::shared_ptr<const To>
         >::type
extension_cast(const std::shared_ptr<const Extension>& from) {
    return static_cast<To *>(0)->Hash == from->hash() ?
        std::static_pointer_cast<const To>(from) : nullptr;
}
#endif

class ExtensionPool {
public:
    template<typename Ext>
    typename std::enable_if<
               IsExtension<Ext>::value, std::shared_ptr<Ext>
             >::type
    get() {
#ifndef NO_EXTENSION_SAFE_CAST
        auto obj = extension_cast<Ext>(get(Ext::Name));
        if (!obj)
            throw std::bad_cast();

        return obj;
#else
        return std::static_pointer_cast<Ext>(get(Ext::Name));
#endif
    }

    template<typename Ext>
    typename std::enable_if<
               IsExtension<Ext>::value, std::shared_ptr<const Ext>
             >::type
    get() const {
#ifndef NO_EXTENSION_SAFE_CAST
        auto obj = extension_cast<Ext>(get(Ext::Name));
        if (!obj)
            throw std::bad_cast();

        return obj;
#else
        return std::static_pointer_cast<Ext>(get(Ext::Name));
#endif
    }

    template<typename Ext>
    typename std::enable_if<
               IsExtension<Ext>::value, std::shared_ptr<Ext>
             >::type
    tryGet() {
#ifndef NO_EXTENSION_SAFE_CAST
        return extension_cast<Ext>(tryGet(Ext::Name));
#else
        return std::static_pointer_cast<Ext>(get(Ext::Name));
#endif
    }

    template<typename Ext>
    typename std::enable_if<
               IsExtension<Ext>::value, std::shared_ptr<const Ext>
             >::type
    tryGet() const {
#ifndef NO_EXTENSION_SAFE_CAST
        return extension_cast<Ext>(tryGet(Ext::Name));
#else
        return std::static_pointer_cast<Ext>(get(Ext::Name));
#endif
    }

    bool has(const std::string& name) const;

    void add(const std::shared_ptr<Extension>& ext);

    std::shared_ptr<Extension> get(const std::string& name);
    std::shared_ptr<const Extension> get(const std::string& name) const;

    std::shared_ptr<Extension> tryGet(const std::string& name);
    std::shared_ptr<const Extension> tryGet(const std::string& name) const;

    std::vector<std::shared_ptr<Extension>> list() const;

private:
    std::pair<bool, std::shared_ptr<Extension>>
    getImpl(const std::string& name) const;

    std::unordered_map<std::string, std::shared_ptr<Extension>> data;
};

struct ExtensionRegistry {
    template<typename Ext>
    static typename std::enable_if<
                        IsExtension<Ext>::value, void
                    >::type
    registerFactory() {
        PluginInterface<Extension>::registerPlugin(Ext::Name,
            []() {
                return std::unique_ptr<Extension>(new Ext);
        });
    }

    static std::unique_ptr<Extension>
    create(const std::string& name, const Json::Value& json) {
        auto factory = PluginInterface<Extension>::getPlugin(name);
        auto ext = factory();
        ext->parse(json);
        return ext;
    }

};


} // namespace RTBKIT
