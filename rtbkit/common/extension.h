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

    virtual const char* extensionName() const = 0;
    virtual void parse(const Json::Value& value) = 0;
    virtual Json::Value toJson() const = 0;
};

#define NAME(extName)                            \
  static constexpr const char* Name = extName; \
  const char* extensionName() const { return Name; }

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

#define STATIC_ASSERT_EXTENSION(Type) \
    static_assert(IsExtension<Type>::value, "The type must be an extension and provide a Name (you should use the NAME macro)"); \
    (void) 0

class ExtensionPool {
public:
    template<typename Ext>
    std::shared_ptr<Ext>
    get() {
        STATIC_ASSERT_EXTENSION(Ext);
        return std::static_pointer_cast<Ext>(get(Ext::Name));
    }

    template<typename Ext>
    std::shared_ptr<const Ext>
    get() const {
        STATIC_ASSERT_EXTENSION(Ext);
        return std::static_pointer_cast<const Ext>(get(Ext::Name));
    }

    template<typename Ext>
    std::shared_ptr<Ext>
    tryGet() {
        STATIC_ASSERT_EXTENSION(Ext);
        return std::static_pointer_cast<Ext>(tryGet(Ext::Name));
    }

    template<typename Ext>
    std::shared_ptr<const Ext>
    tryGet() const {
        STATIC_ASSERT_EXTENSION(Ext);
        return std::static_pointer_cast<const Ext>(tryGet(Ext::Name));
    }

    bool has(const std::string& name) const;
    void add(const std::shared_ptr<Extension>& ext);

    std::vector<std::shared_ptr<Extension>> list() const;

private:
    std::shared_ptr<Extension> get(const std::string& name);
    std::shared_ptr<const Extension> get(const std::string& name) const;

    std::shared_ptr<Extension> tryGet(const std::string& name);
    std::shared_ptr<const Extension> tryGet(const std::string& name) const;

    std::pair<bool, std::shared_ptr<Extension>>
    getImpl(const std::string& name) const;

    std::unordered_map<std::string, std::shared_ptr<Extension>> data;
};

struct ExtensionRegistry {
    template<typename Ext>
    static void
    registerFactory() {
        STATIC_ASSERT_EXTENSION(Ext);
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
