/* augmentor_base.h                                                  -*- C++ -*-
   Jeremy Barnes, 3 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for bid request augmentors.
*/

#ifndef __rtb__augmentor_base_h__
#define __rtb__augmentor_base_h__

#include "soa/service/zmq.hpp"
#include "soa/types/id.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/augmentation.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_utils.h"
#include "soa/service/socket_per_thread.h"
#include "jml/arch/futex.h"
#include "jml/utils/ring_buffer.h"
#include "soa/service/zmq_endpoint.h"

#include <boost/make_shared.hpp>
#include <boost/function.hpp>
#include <boost/thread.hpp>


namespace RTBKIT {

/******************************************************************************/
/* AUGMENTATION REQUEST                                                       */
/******************************************************************************/

/** Regroups the various parameters for the augmentation request.

    Note that this object must be relatively copy friendly because it must be
    transfered to worker threads in the MultiThreadedAugmentorBase and the
    AsyncAugmentorBase.
 */
struct AugmentationRequest
{
    std::string augmentor;                    // Name of the augmentor
    std::string router;                       // Router to respond to.
    Id id;                                    // Id of... something...
    std::shared_ptr<BidRequest> bidRequest; // Bid request to augment
    std::vector<std::string> agents;          // Agents availble to bid.
    double timeAvailableMs;                   // Time to respond.
    Date startTime;                           // Start of the latency timer.
};


/*****************************************************************************/
/* AUGMENTOR BASE                                                            */
/*****************************************************************************/

/** Class that implements a bid request augmentor.  Real augmentors should
    build on top of this class.
*/

struct AugmentorBase : public ServiceBase, public MessageLoop {

    AugmentorBase(const std::string & augmentorName,
                  const std::string & serviceName,
                  std::shared_ptr<ServiceProxies> proxies);

    AugmentorBase(const std::string & augmentorName,
                  const std::string & serviceName,
                  ServiceBase & parent);

    ~AugmentorBase();

    void init();
    void start();
    void shutdown();

    /** Send configuration information and await confirmation that it has been
        received.
    */
    void configureAndWait();

    /** Type of the callback for an augmentation request. */
    typedef boost::function<void (const AugmentationRequest &)> OnRequest;

    /** Function to be called on an augmentation request. */
    OnRequest onRequest;

    /** Function to be called to respond to an augmentation request. */
    void respond(const AugmentationRequest & request,
                 const AugmentationList & response);

private:
    std::string augmentorName; // This can differ from the servicenName!

    ZmqMultipleNamedClientBusProxy toRouters;

    void handleRouterMessage(const std::string & router,
                             const std::vector<std::string> & message);
};


/*****************************************************************************/
/* MULTI THREADED AUGMENTOR                                                   */
/*****************************************************************************/

/** Multi-threaded augmentor base class. */

struct MultiThreadedAugmentorBase : public AugmentorBase {

    MultiThreadedAugmentorBase(const std::string & augmentorName,
                               const std::string & serviceName,
                               std::shared_ptr<ServiceProxies> proxies);

    MultiThreadedAugmentorBase(const std::string & augmentorName,
                               const std::string & serviceName,
                               ServiceBase & parent);

    ~MultiThreadedAugmentorBase();

    void init(int numThreads);
    void shutdown();

protected:

    virtual void doRequestImpl(const AugmentationRequest &) = 0;

private:
    uint64_t numWithInfo;

    ML::RingBufferSWMR<AugmentationRequest> ringBuffer;

    boost::thread_group workers;

    void runWorker();

    boost::thread_group workerThreads;

    int numThreadsCreated;

    volatile bool shutdown_;

    void pushRequest(const AugmentationRequest &);
};


/******************************************************************************/
/* SYNC AUGMENTOR BASE                                                        */
/******************************************************************************/

/** Multi-threaded augmentor designed for synchronous augmentations.

    To return an augmentation to the router, simply return it from the doRequest
    function.

 */
struct SyncAugmentorBase : public MultiThreadedAugmentorBase
{

    SyncAugmentorBase(const std::string & augmentorName,
                      const std::string& serviceName,
                      std::shared_ptr<ServiceProxies> proxies)
            : MultiThreadedAugmentorBase(augmentorName, serviceName, proxies)
    {
        doRequest = boost::bind(&SyncAugmentorBase::onRequest, this, _1);
    }

    SyncAugmentorBase(const std::string & augmentorName,
                      const std::string & serviceName,
                      ServiceBase& parent)
        : MultiThreadedAugmentorBase(augmentorName, serviceName, parent)
    {
        doRequest = boost::bind(&SyncAugmentorBase::onRequest, this, _1);
    }

    boost::function<AugmentationList(const AugmentationRequest &) > doRequest;

    virtual AugmentationList
    onRequest(const AugmentationRequest &)
    {
        throw ML::Exception("onRequest or doRequest must be overridden");
    }

protected:

    virtual void
    doRequestImpl(const AugmentationRequest & request)
    {
        AugmentationList response = doRequest(request);
        respond(request, response);
    }
};


/******************************************************************************/
/* ASYNC AUGMENTOR BASE                                                       */
/******************************************************************************/

/** Multi-threaded augmentor designed for asynchronous augmentations.

    To return the augmentation to the router, pass the augmentation to the
    sendResponse callback provided as a parameter to the doRequest function.

 */
struct AsyncAugmentorBase : public MultiThreadedAugmentorBase
{

    AsyncAugmentorBase(const std::string & augmentorName,
                       const std::string & serviceName,
                       std::shared_ptr<ServiceProxies> proxies) :
        MultiThreadedAugmentorBase(augmentorName, serviceName, proxies)
    {
        doRequest = boost::bind(&AsyncAugmentorBase::onRequest, this, _1, _2);
    }

    AsyncAugmentorBase(const std::string & augmentorName,
                       const std::string & serviceName,
                       ServiceBase& parent) :
        MultiThreadedAugmentorBase(augmentorName, serviceName, parent)
    {
        doRequest = boost::bind(&AsyncAugmentorBase::onRequest, this, _1, _2);
    }

    typedef std::function<void (const AugmentationList &)> SendResponseCB;

    boost::function<void (const AugmentationRequest &, SendResponseCB) >
        doRequest;

    virtual void
    onRequest(const AugmentationRequest & request, SendResponseCB sendResponse)
    {
        throw ML::Exception("onRequest or doRequest must be overridden");
    };

protected:

    virtual void
    doRequestImpl(const AugmentationRequest & request)
    {
        auto sendResponse = [=](const AugmentationList & response) {
            respond(request, response);
        };
        doRequest(request, sendResponse);
    }
};

} // namespace RTBKIT


#endif /* __rtb__augmentor_base_h__ */


