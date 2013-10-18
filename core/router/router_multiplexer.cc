/**
 *  router_multiplexer.cc
 *
 *  Created on: Oct 11, 2013
 *      Author: jan
 */

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <initializer_list>

#include "core/router/router_multiplexer.h"
#include "soa/service/typed_message_channel.h"


using namespace std;
using namespace Datacratic;

namespace RTBKIT
{


RouterMultiplexer::
RouterMultiplexer(shared_ptr<Datacratic::ServiceProxies> proxies,
                  const string &                         rmux_name,
                  const string &                         service_class,
                  const string &                         endpoint_name)
    : Datacratic::ServiceBase(rmux_name, proxies)
    , service_class_ (service_class)
    , endpoint_name_ (endpoint_name)
{

}

RouterMultiplexer::
RouterMultiplexer(Datacratic::ServiceBase &              parent,
                  const string &                         rmux_mame,
                  const string &                         service_class,
                  const string &                         endpoint_name)
    : Datacratic::ServiceBase(rmux_mame, parent)
    , service_class_ (service_class)
    , endpoint_name_ (endpoint_name)
{

}

void RouterMultiplexer::init()
{
}

void RouterMultiplexer::shutdown()
{
    Datacratic::MessageLoop::shutdown();
}

RouterMultiplexer::~RouterMultiplexer()
{
    /* TODO */
}

struct RouterMultiplexer::Entry
{
    ZmqMultipleNamedClientBusProxy  routers_;
    TypedMessageSink<RouterMessage> sink_;
    std::string                                 name_;
    Entry        (ServiceBase& service, const std::string& name)
        : routers_ (service.getZmqContext())
        , sink_    (65536)
        , name_    (name)
    {
    }
};

/**
 *  client enters the multiplex.
 *
 */
pair<ZmqMultipleNamedClientBusProxy*,TypedMessageSink<RouterMessage>*>
RouterMultiplexer::hook (const std::string&   client_name)
{
    std::lock_guard<std::mutex> L(this->mtx_);
    auto it = clients_.find (client_name);
    if (it != clients_.end())
        return make_pair(&(*it).second->routers_, &(*it).second->sink_);
    auto entry = shared_ptr<Entry>(new Entry(*this,client_name));
    clients_[client_name] = entry ;
    return make_pair(&entry->routers_,&entry->sink_);
}

} // namespace RTBKIT

