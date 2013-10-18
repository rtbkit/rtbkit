/**
 *  router_multiplexer.h
 *
 *  Created on: Oct 11, 2013
 *      Author: jan
 */

#ifndef ROUTER_MULTIPLEXER_H_
#define ROUTER_MULTIPLEXER_H_

#include <mutex>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <functional>
#include <unordered_map>
#include <initializer_list>


#include "soa/service/message_loop.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"

namespace RTBKIT
{

/** Format of a message to a router. */
//  TODO share with bidding_agent.h
struct RouterMessage
{
    RouterMessage(const std::string & toRouter = "",
                  const std::string & type = "",
                  const std::vector<std::string> & payload = std::vector<std::string>())
        : dest_(toRouter),
          type_(type),
          payload_(payload)
    {
    }
    std::string dest_, type_;
    std::vector<std::string> payload_;
};


class RouterMultiplexer: public Datacratic::MessageLoop,
    public Datacratic::ServiceBase
{
public:
    RouterMultiplexer(std::shared_ptr<Datacratic::ServiceProxies> proxies,
                      const std::string &                         router_mx_name = "router_mx",
                      const std::string &                         service_class  = "rtbRequestRouter",
                      const std::string &                         endpoint_name  = "agents");
    RouterMultiplexer(Datacratic::ServiceBase &                   parent,
                      const std::string &                         router_mx_name = "router_mx",
                      const std::string &                         service_class  = "rtbRequestRouter",
                      const std::string &                         endpoint_name  = "agents");

    void init();

    void shutdown();

    virtual ~RouterMultiplexer() ;


    //
    RouterMultiplexer (const RouterMultiplexer&)             = delete;
    RouterMultiplexer (const RouterMultiplexer&&)            = delete;
    RouterMultiplexer& operator= (const RouterMultiplexer&)  = delete;
    RouterMultiplexer& operator= (const RouterMultiplexer&&) = delete;



    typedef std::function<void (std::string)> ConnectionHandler;
    typedef std::function<void (std::string, std::vector<std::string>)> MessageHandler;

    /**
     *  client enters the multiplex.
     */
    std::pair<Datacratic::ZmqMultipleNamedClientBusProxy*,
        Datacratic::TypedMessageSink<RouterMessage>*>
        hook (const std::string&   client_name);


private:
    struct Entry;
    std::unordered_map<std::string, std::shared_ptr<Entry>>   clients_ ;
    std::string                                               service_class_;
    std::string                                               endpoint_name_;
    std::mutex                                                mtx_; // serialize calls to join()

};

} /* namespace RTBKIT */
#endif /* ROUTER_MULTIPLEXER_H_ */
