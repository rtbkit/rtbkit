/* augmentor_base.cc

   Jeremy Barnes, 4 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Object that handles doing augmented bid requests.
*/

#include "augmentor_base.h"
#include "soa/service/zmq_utils.h"
#include "jml/arch/timers.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/futex.h"
#include <memory>


using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* AUGMENTOR                                                                 */
/*****************************************************************************/

// Determined via a very scientific method: 2^16 should be enough... right?
// \todo Need to make this queue way smaller so that we don't build up a backlog
// of requests.
enum { QueueSize = 65536 };

Augmentor::
Augmentor(const std::string & augmentorName,
          const std::string & serviceName,
          std::shared_ptr<ServiceProxies> proxies)
    : ServiceBase(serviceName, proxies),
      augmentorName(augmentorName),
      toRouters(getZmqContext()),
      responseQueue(QueueSize),
      requestQueue(QueueSize),
      loopMonitor(*this),
      loadStabilizer(loopMonitor)
{
}

Augmentor::
Augmentor(const std::string & augmentorName,
          const std::string & serviceName,
          ServiceBase& parent)
    : ServiceBase(serviceName, parent),
      augmentorName(augmentorName),
      toRouters(getZmqContext()),
      responseQueue(QueueSize),
      requestQueue(QueueSize),
      loopMonitor(*this),
      loadStabilizer(loopMonitor)
{
}

Augmentor::
~Augmentor()
{
    shutdown();
}

void
Augmentor::
init(int numThreads)
{
    responseQueue.onEvent = [=] (const Response& resp)
        {
            const AugmentationRequest& request = resp.first;
            const AugmentationList& response = resp.second;

            toRouters.sendMessage(
                    request.router,
                    "RESPONSE",
                    "1.0",
                    request.startTime,
                    request.id.toString(),
                    request.augmentor,
                    chomp(response.toJson().toString()));

            recordHit("messages.RESPONSE");
        };

    addSource("Augmentor::responseQueue", responseQueue);

    toRouters.init(getServices()->config, serviceName());

    toRouters.connectHandler = [=] (const std::string & newRouter)
        {
            toRouters.sendMessage(newRouter, "CONFIG", "1.0", augmentorName);
            recordHit("messages.CONFIG");
        };

    toRouters.disconnectHandler = [=] (const std::string & oldRouter)
        {
            cerr << "disconnected from router " << oldRouter << endl;
        };

    toRouters.messageHandler = [=] (const std::string & router,
                                    std::vector<std::string> message)
        {
            handleRouterMessage(router, message);
        };


    toRouters.connectAllServiceProviders("rtbRouterAugmentation", "augmentors");

    addSource("Augmentor::toRouters", toRouters);


    stopWorkers = false;
    for (size_t i = 0; i < numThreads; ++i)
        workers.create_thread([=] { this->runWorker(); });

    loopMonitor.init();
    loopMonitor.addMessageLoop("augmentor", this);
    loopMonitor.onLoadChange = [=] (double) {
        recordLevel(this->loadStabilizer.shedProbability(), "shedProbability");
    };
    addSource("Augmentor::loopMonitor", loopMonitor);
}

void
Augmentor::
start()
{
    MessageLoop::start();
}

void
Augmentor::
shutdown()
{
    stopWorkers = true;
    workers.join_all();
    MessageLoop::shutdown();
    toRouters.shutdown();
}

void
Augmentor::
respond(const AugmentationRequest & request, const AugmentationList & response)
{
    if (responseQueue.tryPush(make_pair(request, response)))
        return;

    cerr << "Dropping augmentation response: response queue is full" << endl;
}

void
Augmentor::
parseMessage(AugmentationRequest& request, Message& message)
{
    const string & version = message.second.at(1);
    ExcCheckEqual(version, "1.0", "unexpected version in augment");

    request.router = message.first;
    request.timeAvailableMs = 0.05;
    request.augmentor = std::move(message.second.at(2));
    request.id = Id(std::move(message.second.at(3)));

    const string & brSource = std::move(message.second.at(4));
    const string & brStr = std::move(message.second.at(5));
    request.bidRequest.reset(BidRequest::parse(brSource, brStr));

    istringstream agentsStr(message.second.at(6));
    ML::DB::Store_Reader reader(agentsStr);
    reader.load(request.agents);

    const string & startTimeStr = message.second.at(7);
    request.startTime = Date::fromSecondsSinceEpoch(strtod(startTimeStr.c_str(), 0));
}

void
Augmentor::
handleRouterMessage(const std::string & router, std::vector<std::string> & message)
{
    ExcCheck(handleRequest, "No request callback set");

    const std::string & type = message.at(0);
    recordHit("messages." + type);

    if (type == "CONFIGOK") {}

    else if (type == "AUGMENT") {

        bool shedMessage = loadStabilizer.shedMessage();

        if (!shedMessage) {
            Message value = make_pair(router, std::move(message));
            shedMessage = !requestQueue.tryPush(std::move(value));
        }

        if (shedMessage) {
            toRouters.sendMessage(
                    router,
                    "RESPONSE",
                    message.at(1), // version
                    message.at(7), // startTime
                    message.at(3), // auctionId
                    message.at(2), // augmentor
                    "null");       // response
            recordHit("shedMessages");
        }
    }

    else cerr << "unknown router message type: " << type << endl;
}

void
Augmentor::
runWorker()
{
    AugmentationRequest request;
    Message message;

    while(!stopWorkers) {
        if (!requestQueue.tryPop(message, 1.0)) continue;

        try { parseMessage(request, message); }
        catch (const std::exception& ex) {
            cerr << "error while parsing message: "
                << message << " -> " << ex.what()
                << endl;
            continue;
        }

        handleRequest(request);
    }
}

} // namespace RTBKIT

