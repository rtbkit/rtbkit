/** service_utils.h                                 -*- C++ -*-
    Rémi Attab, 13 Mar 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Service utilities

*/

#pragma once

#include "service_base.h"

#include <boost/program_options/options_description.hpp>
#include <vector>
#include <string>
#include <sys/utsname.h>


namespace Datacratic {

/******************************************************************************/
/* SERVICE PROXIES ARGUMENTS                                                  */
/******************************************************************************/

enum ConfigurationServiceType {
    CS_NULL, CS_INTERNAL, CS_ZOOKEEPER
};

enum ProgramOptions {
    WITH_ZOOKEEPER, NO_ZOOKEEPER
};

/** Turns command line arguments into a ServiceProxy object */
struct ServiceProxyArguments
{
    boost::program_options::options_description
    makeProgramOptions(const std::string& title = "General Options",
                       ProgramOptions opt = WITH_ZOOKEEPER)
    {
        using namespace boost::program_options;

        options_description options(title);
        options.add_options()
            ("service-name,N", value(&serviceName_),
             "unique name for the service")
            ("bootstrap,B", value(&bootstrap),
             "path to bootstrap.json file")
            ("carbon-connection,c", value(&carbonUri),
             "URI for connecting to carbon daemon")
            ("installation,I", value(&installation),
             "name of the current installation")
            ("location,L", value(&location),
             "Name of the current location");

        if (opt == WITH_ZOOKEEPER) {
            options.add_options()
                ("zookeeper-uri,Z", value(&zookeeperUri),
                 "URI for connecting to zookeeper server");
        }

        return options;
    }

    std::string serviceName(const std::string& defaultValue) const
    {
        return serviceName_.empty() ? defaultValue : serviceName_;
    }

    std::shared_ptr<ServiceProxies>
    makeServiceProxies(ConfigurationServiceType configurationType = CS_ZOOKEEPER)
    {
        auto services = std::make_shared<ServiceProxies>();

        if (!bootstrap.empty())
            services->bootstrap(bootstrap);

        if (configurationType == CS_ZOOKEEPER) {
            if (!zookeeperUri.empty()) {
                ExcCheck(!installation.empty(), "installation is required");
                ExcCheck(!location.empty(), "location is required");
                services->useZookeeper(zookeeperUri, installation, location);
            }
        }
        else if (configurationType == CS_INTERNAL) {
            services->config.reset(new InternalConfigurationService);
        }
        else if (configurationType == CS_NULL) {
            services->config.reset(new NullConfigurationService);
        }

        if (!carbonUri.empty()) {
            ExcCheck(!installation.empty(), "installation is required");
            services->logToCarbon(carbonUri, installation);
        }

        return services;
    }

    std::string bootstrap;
    std::string zookeeperUri;
    std::string carbonUri;
    std::string installation;
    std::string location;

private:

    std::string serviceName_;

};

} // namespace Datacratic
