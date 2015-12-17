/* static_configuration.h
   Mathieu Stefani, 16 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Static configuration system for RTBKit
*/

#pragma once

#include <map>
#include <string>
#include "soa/jsoncpp/json.h"

namespace RTBKIT {
namespace Discovery {

    enum class Protocol { Zmq, Http, Rest };

    struct Port {
        typedef std::map<std::string, uint16_t> Values;
        typedef Values::iterator iterator;
        typedef Values::const_iterator const_iterator;

        Port()
            : type(Null)
        { }

        friend Port operator+(Port lhs, uint16_t value);

        operator uint16_t() const;

        static Port single(uint16_t value);
        static Port multiple(const Values& values);

        static Port parseSingle(const Json::Value& value, const char* name);
        static Port parseMulti(const Json::Value& value, const char* name);

        iterator begin();
        iterator end();

        const_iterator begin() const;
        const_iterator end() const;

        const_iterator find(const std::string& name) const;

        bool isNull() const { return type == Type::Null; }
        bool isSingle() const { return type == Type::Single; }
        bool isMulti() const { return type == Type::Multi; }

    private:
        Port(uint16_t value)
            : value(value)
            , type(Type::Single)
        { }
        Port(const Values& values)
            : values(values)
            , type(Type::Multi)
        { }

        void assertMulti() const;

        enum Type { Null, Single, Multi };
        uint16_t value;
        Values values;

        Type type;
    };

    Port operator+(Port lhs, uint16_t value);

    struct Endpoint {
        Endpoint(std::string name, std::string serviceName, Protocol protocol, Port port)
            : name_(std::move(name))
            , serviceName_(std::move(serviceName))
            , protocol_(protocol)
            , port_(std::move(port))
        { }

        std::string name() const { return name_; }
        std::string serviceName() const { return serviceName_; }
        Protocol protocol() const { return protocol_; }
        Port port() const { return port_; }

    private:
        std::string name_;
        std::string serviceName_;
        Protocol protocol_;
        Port port_;
    };

    typedef std::map<std::string, Endpoint> Endpoints;

    struct Binding {
        struct Context {
            Endpoints endpoints;
        };

        Binding(Endpoint endpoint, Port port)
            : endpoint_(std::move(endpoint))
            , port_(std::move(port))
        { }

        static Context context(const Endpoints& endpoint);

        static Binding fromExpression(const Json::Value& value, const Context& context);
        static Binding fromExpression(const std::string& value, const Context& context);

        Endpoint endpoint() const { return endpoint_; }
        Port port() const { return port_; }

    private:
        Endpoint endpoint_;
        Port port_;
    };

class StaticDiscovery {
public:
     static StaticDiscovery fromFile(const std::string& fileName);
     static StaticDiscovery fromJson(const Json::Value& value);     

     Endpoint namedEndpoint(const std::string& name) const;

private:
     Endpoints endpoints;
};

} // namespace Discovery

} // namespace RTBKIT

namespace std {
}
