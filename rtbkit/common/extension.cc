/* extension.cc
   Mathieu Stefani, 14 janvier 2016
   Copyright (c) 2015 Datacratic.  All rights reserved.
 
*/

#include "extension.h"
#include <iostream>

namespace RTBKIT {

bool
ExtensionPool::has(const std::string& name) const {
    return getImpl(name).first;
}

void
ExtensionPool::add(const std::shared_ptr<Extension>& ext){
    data.insert(std::make_pair(ext->extensionName(), ext));
}

std::shared_ptr<Extension>
ExtensionPool::get(const std::string& name) {
    auto ext = getImpl(name);
    if (!ext.first) {
        throw ML::Exception("Unknown extension '%s'", name.c_str());
    }

    return ext.second;
}

std::shared_ptr<const Extension>
ExtensionPool::get(const std::string& name) const {
    auto ext = getImpl(name);
    if (!ext.first) {
        throw ML::Exception("Unknown extension '%s'", name.c_str());
    }

    return ext.second;
}

std::shared_ptr<Extension>
ExtensionPool::tryGet(const std::string& name) {
    auto ext = getImpl(name);
    if (!ext.first) {
        return nullptr;
    }

    return ext.second;
}

std::shared_ptr<const Extension>
ExtensionPool::tryGet(const std::string& name) const {
    auto ext = getImpl(name);
    if (!ext.first) {
        return nullptr;
    }

    return ext.second;
}

std::vector<std::shared_ptr<Extension>>
ExtensionPool::list() const {
    std::vector<std::shared_ptr<Extension>> result;
    result.reserve(data.size());
    for (const auto& ext: data) {
        result.push_back(ext.second);
    }

    return result;
}

std::pair<bool, std::shared_ptr<Extension>>
ExtensionPool::getImpl(const std::string& name) const {
    auto it = data.find(name);
    if (it == std::end(data)) {
        return std::make_pair(false, nullptr);
    }

    return std::make_pair(true, it->second);
}

} // namespace RTBKIT
