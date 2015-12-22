/* args.h
   Mathieu Stefani, 21 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Command line arguments for RTBKit, based on ServiceProxyArguments
*/

#pragma once
#include "rtbkit/common/static_configuration.h"
#include "soa/service/service_utils.h"
#include <boost/program_options/options_description.hpp>

namespace RTBKIT {

class ProxyArguments : public Datacratic::ServiceProxyArguments {
public:

    boost::program_options::options_description
    makeProgramOptions(const std::string& title = "General Options")
    {
        using namespace boost::program_options;

        auto opts = Datacratic::ServiceProxyArguments::makeProgramOptions(title);
        opts.add_options()
            ("static-discovery", value(&staticDiscovery),
             "path to the static discovery file that will be used instead of "
             "ZooKeeper");

        return opts;
    }

    std::shared_ptr<Datacratic::ServiceProxies>
    makeServiceProxies(const std::string& serviceName = "")
    {
        auto services = Datacratic::ServiceProxyArguments::makeServiceProxies();

        if (!staticDiscovery.empty()) {
            ExcCheck(!serviceName.empty(), "Must provide a serviceName when using static discovery");

            auto discovery = std::make_shared<Discovery::StaticDiscovery>();
            discovery->parseFromFile(staticDiscovery);

            auto config = std::make_shared<Discovery::StaticConfigurationService>();
            config->init(discovery);

            services->config = config;
            services->ports.reset(new Discovery::StaticPortRangeService(discovery, serviceName));
        }

        return services;
    }

private:
    std::string staticDiscovery;

};

} // namespace RTBKIT
