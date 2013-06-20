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

/** Turns command line arguments into a ServiceProxy object */
struct ServiceProxyArguments
{
    boost::program_options::options_description
    makeProgramOptions(const std::string& title = "General Options")
    {
        using namespace boost::program_options;

        options_description options(title);
        options.add_options()
            ("bootstrap,B", value(&bootstrap),
             "path to bootstrap.json file")
            ("zookeeper-uri,Z", value(&zookeeperUri),
             "URI for connecting to zookeeper server")
            ("carbon-connection,c", value(&carbonUri),
             "URI for connecting to carbon daemon")
            ("installation,I", value(&installation),
             "name of the current installation")
            ("location,L", value(&location),
             "Name of the current location");

        return options;
    }

    std::shared_ptr<ServiceProxies> makeServiceProxies()
    {
        auto services = std::make_shared<ServiceProxies>();

        if (!bootstrap.empty())
            services->bootstrap(bootstrap);

        if (!zookeeperUri.empty()) {
            ExcCheck(!installation.empty(), "installation is required");
            ExcCheck(!location.empty(), "location is required");
            services->useZookeeper(zookeeperUri, installation, location);
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
};

} // namespace Datacratic
