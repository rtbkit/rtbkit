/* static_configuration.cc
   Mathieu Stefani, 16 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Implementation of the static discovery
*/

#include "static_configuration.h"
#include "jml/arch/exception.h"
#include "jml/utils/file_functions.h"

namespace RTBKIT {
namespace Discovery {

namespace {
    Json::Value loadJsonFromFile(const std::string & filename)
    {
        ML::File_Read_Buffer buf(filename);
        return Json::parse(std::string(buf.start(), buf.end()));
    }

    Json::Value jsonMember(const Json::Value& value, const char* fieldName, const char* object) {
        if (!value.isMember(fieldName))
            throw ML::Exception("Expected field '%s' in '%s'", fieldName, object);

        return value[fieldName];
    }

    template<typename T>
    struct TypedJson;

#define TYPED_JSON_INTEGER(Integer) \
    template<> \
    struct TypedJson<Integer> { \
        static Integer extract(const Json::Value& val, const char* name) { \
            auto v = val.asInt(); \
            const auto max = std::numeric_limits<Integer>::max(); \
            if (v > max) \
                throw ML::Exception("json int value '%s' overflows type '%s' (%llu > %llu)", \
                     name, #Integer, static_cast<unsigned long long>(v), static_cast<unsigned long long>(max)); \
            return v; \
        } \
    };

#define TYPED_JSON_UNSIGNED_INTEGER(Integer) \
    template<> \
    struct TypedJson<Integer> { \
        static Integer extract(const Json::Value& val, const char *name) { \
            auto v = val.asUInt(); \
            const auto max = std::numeric_limits<Integer>::max(); \
            if (v > max) \
                throw ML::Exception("json int value '%s' overflows type '%s' (%llu > %llu)", \
                     name, #Integer, static_cast<unsigned long long>(v), static_cast<unsigned long long>(max)); \
            return v; \
        } \
    };

    TYPED_JSON_INTEGER(int8_t)
    TYPED_JSON_INTEGER(int16_t)
    TYPED_JSON_INTEGER(int32_t)
    TYPED_JSON_INTEGER(int64_t)

    TYPED_JSON_UNSIGNED_INTEGER(uint8_t)
    TYPED_JSON_UNSIGNED_INTEGER(uint16_t)
    TYPED_JSON_UNSIGNED_INTEGER(uint32_t)
    TYPED_JSON_UNSIGNED_INTEGER(uint64_t)

#undef TYPED_JSON_INTEGER
#undef TYPED_JSON_UNSIGNED_INTEGER

    template<>
    struct TypedJson<float> {
        static double extract(const Json::Value& val, const char*) {
            return val.asDouble();
        }
    };

    template<>
    struct TypedJson<double> {
        static double extract(const Json::Value& val, const char*) {
            return val.asDouble();
        }
    };

    template<>
    struct TypedJson<std::string> {
        static std::string extract(const Json::Value& val, const char*) {
            return val.asString();
        }
    };

    template<>
    struct TypedJson<Protocol> {
        static Protocol extract(const Json::Value& val, const char*) {
            auto str = val.asString();

            if (str == "zmq")
                return Protocol::Zmq;
            else if (str == "http")
                return Protocol::Http;
            else if (str == "rest")
                return Protocol::Rest;
            else
                throw ML::Exception("Unknown protocol '%s'", str.c_str());
        }
    };

    template<>
    struct TypedJson<ZmqEndpointType> {
        static ZmqEndpointType extract(const Json::Value& val, const char*) {
            auto str = val.asString();
            if (str == "bus")
                return ZmqEndpointType::Bus;
            else if (str == "publisher")
                return ZmqEndpointType::Publisher;

            throw ML::Exception("Unknown endopint type '%s'", str.c_str());
        }
    };

    template<typename T>
    T typedJsonMember(const Json::Value& value, const char* fieldName, const char* object) {
        auto val = jsonMember(value, fieldName, object);

        return TypedJson<T>::extract(val, fieldName);
    }

    template<typename T>
    T typedJsonValue(const Json::Value& value, const char* name) {
        return TypedJson<T>::extract(value, name);
    }

    void
    jsonForeach(const Json::Value& val, std::function<void (std::string, const Json::Value&)> func) {
        for (auto it = val.begin(), end = val.end(); it != end; ++it) {
            func(it.memberName(), *it);
        }
    }

    Endpoint parseEndpoint(const Json::Value& value, std::string name) {
        const char *n = name.c_str();

        std::string serviceName;
        if (value.isMember("serviceName"))
            serviceName = typedJsonMember<std::string>(value, "serviceName", n);
        else
            serviceName = name;

        auto protocol    = typedJsonMember<Protocol>(value, "protocol", n);
        Port port;
        if (protocol == Protocol::Rest) {
            port = Port::parseMulti(jsonMember(value, "ports", n), n);
        } else {
            port = Port::parseSingle(jsonMember(value, "port", n), n);
        }

        Endpoint endpoint(
                std::move(name), std::move(serviceName),
                protocol, std::move(port));

        if (protocol == Protocol::Zmq) {
            endpoint.setData(std::make_shared<ZmqData>(
                    typedJsonMember<ZmqEndpointType>(value, "type", n)));
        }

        return endpoint;

    }
}

Port::operator uint16_t() const {
    if (type != Type::Single)
        throw ML::Exception("Can not retrieve a single value of a multiple ports");

    return value;
}

Port
Port::single(uint16_t value) {
    return Port(value);
}

Port
Port::multiple(const std::map<std::string, uint16_t>& values) {
    return Port(values);
}

Port
Port::parseSingle(const Json::Value& value, const char* name) {
    Port port;
    port.value = typedJsonValue<uint16_t>(value, name);
    port.type = Type::Single;

    return port;
}

Port
Port::parseMulti(const Json::Value& value, const char* name) {
   Port port;

   for (auto it = value.begin(), end = value.end(); it != end; ++it) {
       auto val = typedJsonValue<uint16_t>(*it, name);
       port.values.insert(std::make_pair(it.memberName(), val));
   }
   port.type = Type::Multi;

   return port; 
}

Port::iterator
Port::begin() {
    assertMulti();
    return values.begin();
}

Port::iterator
Port::end() {
    assertMulti();
    return values.end();
}

Port::const_iterator
Port::begin() const {
    assertMulti();
    return values.begin();
}

Port::const_iterator
Port::end() const {
    assertMulti();
    return values.end();
}

Port::const_iterator
Port::find(const std::string& name) const {
    assertMulti();
    return values.find(name);
}

void
Port::assertMulti() const {
    if (type != Type::Multi)
        throw ML::Exception("Invalid operation on single port");
}

Port
operator+(Port lhs, uint16_t value)
{
    if (lhs.isSingle()) {
        lhs.value += value;
    } else {
        for (auto& val: lhs.values) {
            val.second += value;
        }
    }

    return lhs;
}

Binding::Context
Binding::context(const Endpoints& endpoints, std::string name) {
    return Binding::Context { endpoints, std::move(name) };
}

Binding
Binding::fromExpression(const Json::Value& value, const Context& context) {
    return Binding::fromExpression(value.asString(), context);
}

Binding
Binding::fromExpression(const std::string& value, const Context& context) {

    auto findEndpoint = [&](const std::string& name) {
        auto it = context.endpoints.find(name);
        if (it == std::end(context.endpoints))
            throw ML::Exception("Could not find endpoint '%s' for binding expression '%s'", name.c_str(), context.name.c_str());

        return it->second;
    };

    auto pos = value.find(':');
    if (pos == std::string::npos) {
        auto ep = findEndpoint(value);
        return Binding(ep, ep.port());
    } else {
        std::string e = value.substr(0, pos);
        std::string p = value.substr(pos + 1);

        auto ep = findEndpoint(e);

        const char *raw = p.c_str();
        if (*raw == '$') {
            if (*++raw != '+')
                throw ML::Exception("Binding expression for '%s': expected '+' got '%c' (%d)",
                    context.name.c_str(), *raw, static_cast<int>(*raw));

            uint16_t incr = std::strtol(++raw, nullptr, 10);
            if (incr == 0)
                throw ML::Exception("Invalid operation for binding expression '%s': '%s'", context.name.c_str(), p.c_str());

            Port port = ep.port() + incr;
            return Binding(ep, ep.port() + incr);
        }

        return Binding(ep, ep.port());
    }
}

Binding
Service::Node::binding(const std::string& name) const {
    auto it = std::find_if(std::begin(bindings), std::end(bindings), [&](const Binding& binding) {
        return binding.endpoint().name() == name;
    });

    // ExcAssert(it != std::end(bindings))
    return *it;
}

void
Service::addNode(const Service::Node& node) {
    nodes.insert(std::make_pair(node.serviceName, node));
}

bool
Service::hasNode(const std::string& name) const {
    return nodes.find(name) != std::end(nodes);
}

Service::Node
Service::node(const std::string& name) const {
    if (!hasNode(name))
        throw ML::Exception("Node '%s' for service '%s' does not exist",
                name.c_str(), className.c_str());

    return nodes.find(name)->second;
}

std::vector<Service::Node>
Service::allNodes() const {
    std::vector<Service::Node> res;
    res.reserve(nodes.size());
    for (const auto& node: nodes) {
        res.push_back(node.second);
    }

    return res;
}

StaticDiscovery
StaticDiscovery::fromFile(const std::string& fileName)
{
    return StaticDiscovery::fromJson(loadJsonFromFile(fileName));
}

StaticDiscovery
StaticDiscovery::fromJson(const Json::Value& value) {
    StaticDiscovery res;

    if (!value.isObject())
        throw ML::Exception("root: expected a json object");

    auto &epts = value["endpoints"];
    if (!epts.isObject())
        throw ML::Exception("endpoints: expected a json object");

    std::map<std::string, Endpoint> endpoints;

    jsonForeach(epts, [&](std::string name, const Json::Value& value) {
        auto endpoint = parseEndpoint(value, name.c_str());
        endpoints.insert(std::make_pair(std::move(name), std::move(endpoint)));
    });

    auto &srvs = value["services"];
    if (!srvs.isObject())
        throw ML::Exception("services: expected a json object");

    static constexpr const char* KnownServices[] = {
        "agentConfiguration",
        "router",
        "augmentation",
        "postAuction",
        "banker",
        "adserver",
        "monitor"
    };

    std::map<std::string, Service> services;
    jsonForeach(srvs, [&](std::string srvClass, const Json::Value& service) {
        auto srvIt = std::find(std::begin(KnownServices), std::end(KnownServices), srvClass);
        if (srvIt == std::end(KnownServices))
            throw ML::Exception("Unknown service class '%s'", srvClass.c_str());

        jsonForeach(service, [&](std::string serviceName, const Json::Value& value) {
            auto bindArr = jsonMember(value, "bind", serviceName.c_str());
            if (!bindArr.isArray())
                throw ML::Exception("bind for '%s': expected array", serviceName.c_str());

            Service service(srvClass);
            std::vector<Binding> bindings;
            for (const auto& bind: bindArr) {
                auto binding = Binding::fromExpression(bind, Binding::context(endpoints, serviceName));
                bindings.push_back(std::move(binding));
            }

            service.addNode(Service::Node(std::move(serviceName), std::move(bindings)));

            services.insert(std::make_pair(std::move(srvClass), std::move(service)));

        });

    });

    res.endpoints = std::move(endpoints);
    res.services = std::move(services);

    return res;
}

} // namespace Discovery

} // namespace RTBKIT
