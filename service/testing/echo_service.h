#ifndef ECHO_SERVICE_H
#define ECHO_SERVICE_H

#include <memory>
#include <string>
#include <vector>
#include "soa/service/zmq_endpoint.h"
#include "soa/service/service_base.h"
#include "jml/utils/exc_assert.h"

namespace Datacratic {

/*****************************************************************************/
/* ECHO SERVICE                                                              */
/*****************************************************************************/

/** Simple test service that listens on zeromq and simply echos everything
    that it gets back.
*/

struct EchoService : public ServiceBase {

    EchoService(std::shared_ptr<ServiceProxies> proxies,
                const std::string & name)
        : ServiceBase(name, proxies),
          toClients(getZmqContext())
    {
        proxies->config->removePath(serviceName());
        registerServiceProvider(serviceName(), { "echo" });

        auto handler = [=] (std::vector<std::string> message)
            {
                //cerr << "got message " << message << endl;
                ExcAssertEqual(message.size(), 3);
                ExcAssertEqual(message[1], "ECHO");
                message[1] = "REPLY";
                return message;
            };

        toClients.clientMessageHandler = handler;
    }

    ~EchoService()
    {
        shutdown();
    }

    void init()
    {
        toClients.init(getServices()->config, serviceName() + "/echo");
    }

    void start()
    {
        toClients.start();
    }

    void shutdown()
    {
        toClients.shutdown();
    }

    std::string bindTcp()
    {
        return toClients.bindTcp();
    }

    ZmqNamedClientBus toClients;
};

} // namespace Datacratic

#endif
