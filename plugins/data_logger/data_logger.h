/* data_logger.h                                                        -*- C++ -*-
   Sunil Rottoo
   Copyright (c) 2013 Datacratic.  All rights reserved.

*/
#pragma once

#include "soa/logger/logger.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_named_pub_sub.h"
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace RTBKIT {
/******************************************************************************/
/*  Data Logger                                                               */
/******************************************************************************/

 /**
  * This is class that can be used to connect to well-known service classes
  * that are registered to ZooKeeper. Typically all service providers will
  * register a "logger" endpoint in Zookeeper to which this logger will connect.
  */
class DataLogger : public Datacratic::Logger {

public:
     /*
      * Create a logger:
      * zookeeperURI: the URI of the ZooKeeper service we are using
      * installation: The prefix being used for this installation
      */
     DataLogger(std::string zookeeperURI, std::string installation);
    ~DataLogger();
    /*
     * Initialize the logger *
     */
    void init();
    /*
     * Start the logger
     */
    void start() ;
    /*
     * Shutdown the logger
     */
    void shutdown() ;
    /*
     * Connect to all the services specified
     */
    void connectToAllServices(const std::vector<std::string> &services);

protected:
    void createProxies(std::string zookeeperURI, std::string installation);
    std::string zookeeperURI_;
    std::string installation_;
    std::shared_ptr<Datacratic::ServiceProxies> proxies_;
    std::shared_ptr<Datacratic::ZmqNamedMultipleSubscriber> multipleSubscriber_;
} ;

}// namespace RTBKIT
