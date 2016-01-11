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
#include <stdlib.h>                         // ::getenv()
#include <sys/utsname.h>
#include <dlfcn.h>                          // dlopen()
#include "jml/utils/string_functions.h"     // ML::split()
#include "jml/utils/file_functions.h"       // ML::fileExists()
#include "jml/utils/json_parsing.h"
#include "jml/arch/exception.h"             // ML::Exception
#include "jml/utils/environment.h"          // ML::Environment

namespace {

/******************************************************************************/
/* HELPER FUNCTIONS FOR PRELOADING LIBS DYNAMICALLY                           */
/******************************************************************************/

void loadLib(const std::string & file)
{
    void * handle = dlopen(file.c_str(), RTLD_NOW);
    if (!handle) {
        std::cerr << dlerror() << std::endl;
        throw ML::Exception("couldn't load library from %s", file.c_str());
    }
}

void loadJsonLib(const std::string & jsonFile)
{
    using std::string;
    using std::vector;

    ML::File_Read_Buffer buf(jsonFile);
    Json::Value jsonListLibs = Json::parse(std::string(buf.start(), buf.end()));

    if (jsonListLibs == Json::Value::null) return;

    if (!jsonListLibs.isArray())
        throw ML::Exception("Library list must be an array");

    string file;
    for (const auto & fileAsJson : jsonListLibs) {

        file = fileAsJson.asString();

        if (file.empty()) throw ML::Exception("File name cannot be empty");

        if (ML::endsWith(file,".so")) {
            if (!ML::fileExists(file))
                throw ML::Exception("File %s in %s does not exist", file.c_str(), jsonFile.c_str());
            loadLib(file);
            continue;
        }

        file = file + ".so";
        if (!ML::fileExists(file))
            throw ML::Exception("File %s in %s does not exist", file.c_str(), jsonFile.c_str());
        loadLib(file);
    }
}

void loadLibList(const std::string & fileList)
{
    using std::string;
    using std::vector;

    vector<string> listToPreload=ML::split(fileList,',');
    string fileSo;
    string fileJson;

    for (const string & file : listToPreload) {

        if (file.empty()) throw ML::Exception("File name cannot be empty");

        if (ML::endsWith(file,".so")) {
            if (!ML::fileExists(file))
                throw ML::Exception("File %s does not exist", file.c_str());
            loadLib(file);
            continue;
        }

        if (ML::endsWith(file,".json")) {
            if (!ML::fileExists(file))
                throw ML::Exception("File %s does not exist", file.c_str());
            loadJsonLib(file);
            continue;
        }

        fileSo = file + ".so";
        if (ML::fileExists(fileSo)) {
            loadLib(fileSo);
            continue;
        }

        fileJson = file + ".json";
        if (ML::fileExists(fileJson)) {
            loadJsonLib(fileJson);
            continue;
        }

        throw ML::Exception("File %s does not exist in neither .so nor .json format", file.c_str());
    }

}

} // namespace anonymuous

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
             "Name of the current location")
            ("preload,P", value(&preload),
             "Comma separated list of libraries to preload and/or json files");

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
        preloadDynamicLibs();

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
    std::string preload;

private:

    std::string serviceName_;

    void preloadDynamicLibs() const
    { 
        char * envPreloadC = ::getenv("RTBKIT_PRELOAD");
        std::string envPreload = envPreloadC ? envPreloadC : "";

        if(!envPreload.empty())
            loadLibList(envPreload);

        if (!preload.empty())
            loadLibList(preload);
    }

};

} // namespace Datacratic
