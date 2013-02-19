/* data_logger.cc
   Sunil Rottoo, 13 February 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Simple Class for logging
*/

#include "data_logger.h"
#include "soa/service/service_base.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

DataLogger::DataLogger(std::string zookeeperURI, string installation):
                zookeeperURI_(zookeeperURI), installation_(installation)
{
   createProxies(zookeeperURI_, installation_);
   multipleSubscriber_ = make_shared<ZmqNamedMultipleSubscriber>(proxies_->zmqContext);
}

DataLogger::~DataLogger()
{
    shutdown();
}

void
DataLogger::start()
{
    Logger::start();
    multipleSubscriber_->start();
}

void
DataLogger::createProxies(std::string zookeeperURI, std::string installation)
{
    proxies_ = std::make_shared<ServiceProxies>();
    proxies_->useZookeeper(zookeeperURI, installation);
}

void
DataLogger::init()
{
    Logger::init();
    multipleSubscriber_->init(proxies_->config);
    multipleSubscriber_->messageHandler
            = [&] (vector<zmq::message_t> && msg) {
            // forward to logger class
            vector<string> s;
            s.reserve(msg.size());
            for (auto & m: msg)
                s.push_back(m.toString());
            this->logMessageNoTimestamp(s);
        };

}
void
DataLogger::connectToAllServices(const std::vector<std::string> &services)
{
    for( auto service: services)
        multipleSubscriber_->connectAllServiceProviders(service, "logger");
}

void DataLogger::shutdown()
{
    Logger::shutdown();
    multipleSubscriber_->shutdown();
}

} // namespace Datacratic
