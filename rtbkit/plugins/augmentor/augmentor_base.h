/* augmentor_base.h                                                  -*- C++ -*-
   Jeremy Barnes, 3 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for bid request augmentors.
*/

#ifndef __rtb__augmentor_base_h__
#define __rtb__augmentor_base_h__

#include <atomic>

#include "soa/service/service_base.h"
#include "soa/service/zmq.hpp"
#include "soa/types/id.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/augmentation.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_utils.h"
#include "soa/service/socket_per_thread.h"
#include "soa/service/typed_message_channel.h"
#include "soa/service/loop_monitor.h"
#include "jml/arch/futex.h"
#include "jml/utils/ring_buffer.h"
#include "soa/service/zmq_endpoint.h"

#include <boost/function.hpp>
#include <boost/thread.hpp>


namespace RTBKIT {

/******************************************************************************/
/* AUGMENTATION REQUEST                                                       */
/******************************************************************************/

/** Regroups the various parameters for the augmentation request.

    Note that this object must be relatively copy friendly because it must be
    transfered to worker threads in the MultiThreadedAugmentor and the
    AsyncAugmentor.
 */
struct AugmentationRequest
{
    std::string augmentor;                    // Name of the augmentor
    std::string router;                       // Router to respond to
    Id id;                                    // Auction id
    std::shared_ptr<BidRequest> bidRequest;   // Bid request to augment
    std::vector<std::string> agents;          // Agents availble to bid
    double timeAvailableMs;                   // Time to respond
    Date startTime;                           // Start of the latency timer
};


/*****************************************************************************/
/* AUGMENTOR BASE                                                            */
/*****************************************************************************/

/** Class that implements a bid request augmentor.  Real augmentors should
    build on top of this class.
*/

struct Augmentor : public ServiceBase, public MessageLoop {

    Augmentor(const std::string & augmentorName,
                  const std::string & serviceName,
                  std::shared_ptr<ServiceProxies> proxies);

    Augmentor(const std::string & augmentorName,
                  const std::string & serviceName,
                  ServiceBase & parent);

    ~Augmentor();

    void init(int numThreads = 1);
    void start();
    void shutdown();

    /** Function to be called to respond to an augmentation request. */
    void respond(const AugmentationRequest & request,
                 const AugmentationList & response);

    double sampleLoad() { return loopMonitor.sampleLoad().load; }
    double shedProbability() { return loadStabilizer.shedProbability(); }


protected:

    typedef boost::function<void (const AugmentationRequest &)> RequestHandle;
    RequestHandle handleRequest;

private:
    std::string augmentorName; // This can differ from the servicenName!

    ZmqMultipleNamedClientBusProxy toRouters;

    typedef std::pair<AugmentationRequest, AugmentationList> Response;
    TypedMessageSink<Response> responseQueue;

    typedef std::pair<std::string, std::vector<std::string> > Message;
    ML::RingBufferSWMR<Message> requestQueue;

    boost::thread_group workers;
    std::atomic<bool> stopWorkers;

    LoopMonitor loopMonitor;
    LoadStabilizer loadStabilizer;

    void runWorker();
    void handleRouterMessage(const std::string & router,
                             std::vector<std::string> & message);

    void parseMessage(AugmentationRequest& req, Message& msg);
};


/******************************************************************************/
/* SYNC AUGMENTOR BASE                                                        */
/******************************************************************************/

/** Multi-threaded augmentor designed for synchronous augmentations.

    To return an augmentation to the router, simply return it from the doRequest
    function.

 */
struct SyncAugmentor : public Augmentor
{


    SyncAugmentor(
            const std::string & augmentorName,
            const std::string& serviceName,
            std::shared_ptr<ServiceProxies> proxies) :
        Augmentor(augmentorName, serviceName, proxies)
    {
        setup();
    }

    SyncAugmentor(
            const std::string & augmentorName,
            const std::string & serviceName,
            ServiceBase& parent)
        : Augmentor(augmentorName, serviceName, parent)
    {
        setup();
    }

    boost::function<AugmentationList(const AugmentationRequest &) > doRequest;

    virtual AugmentationList
    onRequest(const AugmentationRequest &)
    {
        throw ML::Exception("onRequest or doRequest must be overridden");
    }

private:

    void setup()
    {
        doRequest = boost::bind(&SyncAugmentor::onRequest, this, _1);

        handleRequest = [=] (const AugmentationRequest & request) {
            AugmentationList response = doRequest(request);
            respond(request, response);
        };
    }

};


/******************************************************************************/
/* ASYNC AUGMENTOR BASE                                                       */
/******************************************************************************/

/** Multi-threaded augmentor designed for asynchronous augmentations.

    To return the augmentation to the router, pass the augmentation to the
    sendResponse callback provided as a parameter to the doRequest function.

 */
struct AsyncAugmentor : public Augmentor
{

    AsyncAugmentor(
            const std::string & augmentorName,
            const std::string & serviceName,
            std::shared_ptr<ServiceProxies> proxies) :
        Augmentor(augmentorName, serviceName, proxies)
    {
        setup();
    }

    AsyncAugmentor(
            const std::string & augmentorName,
            const std::string & serviceName,
            ServiceBase& parent) :
        Augmentor(augmentorName, serviceName, parent)
    {
        setup();
    }

    typedef std::function<void (const AugmentationList &)> SendResponseCB;

    boost::function<void (const AugmentationRequest &, SendResponseCB) >
        doRequest;

    virtual void
    onRequest(const AugmentationRequest & request, SendResponseCB sendResponse)
    {
        throw ML::Exception("onRequest or doRequest must be overridden");
    };

private:

    void setup()
    {
        doRequest = boost::bind(&AsyncAugmentor::onRequest, this, _1, _2);

        handleRequest = [=] (const AugmentationRequest & request) {
            auto sendResponse = [=](const AugmentationList & response) {
                respond(request, response);
            };
            doRequest(request, sendResponse);
        };
    }

};

} // namespace RTBKIT


#endif /* __rtb__augmentor_base_h__ */


