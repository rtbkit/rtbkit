/* static_configuration.h
   Mathieu Stefani, 16 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Static configuration system for RTBKit
*/

#pragma once

#include <map>
#include <string>
#include <memory>
#include <iostream>
#include "soa/jsoncpp/json.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/service_base.h"
#include "soa/service/port_range_service.h"
#include "jml/arch/exception.h"

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
            , extraData_(nullptr)
        { }

        std::string name() const { return name_; }
        std::string serviceName() const { return serviceName_; }
        Protocol protocol() const { return protocol_; }
        Port port() const { return port_; }

        template<typename Ptr>
        void setData(Ptr&& data) {
            extraData_ = std::forward<Ptr>(data);
        }

        template<typename T>
        T *data() const {
            return std::static_pointer_cast<T>(extraData_).get();
        }

    private:
        std::string name_;
        std::string serviceName_;
        Protocol protocol_;
        Port port_;
        std::shared_ptr<void> extraData_;
    };

    enum class ZmqEndpointType { Bus, Publisher };

    struct ZmqData {
        ZmqData(ZmqEndpointType type)
            : type(type)
        { }

        ZmqEndpointType type;
    };

    typedef std::map<std::string, Endpoint> Endpoints;

    struct Binding {
        struct Context {
            Endpoints endpoints;
            std::string name;
        };

        Binding(Endpoint endpoint, Port port)
            : endpoint_(std::move(endpoint))
            , port_(std::move(port))
        { }

        static Context context(const Endpoints& endpoint, std::string name);

        static Binding fromExpression(const Json::Value& value, const Context& context);
        static Binding fromExpression(const std::string& value, const Context& context);

        Endpoint endpoint() const { return endpoint_; }
        Port port() const { return port_; }

    private:
        Endpoint endpoint_;
        Port port_;
    };

    struct Service {
        struct Node {
            Node(std::string serviceName, std::string hostName, const std::vector<Binding>& bindings)
                : serviceName(std::move(serviceName))
                , hostName(std::move(hostName))
                , bindings(bindings)
            { }

            Binding binding(const std::string& name) const;
            std::vector<Binding> protocolBindings(Protocol protocol) const;
            std::string fullServiceName(const std::string& endpointName) const {
                return serviceName + "/" + endpointName;
            }

            std::string serviceName;
            std::string hostName;
            std::vector<Binding> bindings;
        };

        Service(std::string className)
            : className(std::move(className))
        { }

        void addNode(const Node& node);
        bool hasNode(const std::string& name) const;
        Node node(const std::string& name) const;
        std::vector<Node> allNodes() const;

    private:
        std::map<std::string, Node> nodes;
        std::string className;
    };

    class StaticDiscovery {
    public:
         static StaticDiscovery fromFile(const std::string& fileName);
         static StaticDiscovery fromJson(const Json::Value& value);

         void parseFromFile(const std::string& fileName);
         void parseFromJson(const Json::Value& value);

         Service::Node node(const std::string& serviceName) const {
             for (const auto& service: services) {
                 if (service.second.hasNode(serviceName))
                     return service.second.node(serviceName);
             }

             throw ML::Exception("Unknown node '%s'", serviceName.c_str());
         }

         Service service(const std::string& serviceClass) const {
             auto it = services.find(serviceClass);
             if (it == std::end(services))
                 throw ML::Exception("Unknown service '%s'", serviceClass.c_str());

             return it->second;
         }

    private:
         Endpoints endpoints;
         std::map<std::string, Service> services;
    };

    struct StaticConfigurationService : public Datacratic::ConfigurationService {

        void init(const std::shared_ptr<StaticDiscovery>& discovery);

        Json::Value getJson(
                const std::string& value, Watch watch = Watch());

        void set(const std::string& key,
                const Json::Value& value);

        std::string setUnique(const std::string& key, const Json::Value& value);

        std::vector<std::string>
        getChildren(const std::string& key,
                    Watch watch = Watch());

        bool forEachEntry(const OnEntry& onEntry,
                          const std::string& startPrefix = "") const;

        void removePath(const std::string& path);

    private:
        std::vector<std::string> splitKey(const std::string& key) const;
        std::shared_ptr<StaticDiscovery> discovery;

    };

    struct StaticPortRangeService : public Datacratic::PortRangeService {

        StaticPortRangeService(
                const std::shared_ptr<StaticDiscovery>& discovery,
                const std::string& nodeName);

        Datacratic::PortRange getRange(const std::string& name);
        void dump(std::ostream& stream = std::cerr) const;

    private:
        std::shared_ptr<StaticDiscovery> discovery;
        std::string nodeName;

        std::vector<std::string> splitPort(const std::string& name) const;
    };

} // namespace Discovery

} // namespace RTBKIT

namespace std {
}
