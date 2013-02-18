/** flume_endpoint.cc
    Jeremy Barnes, 15 November 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "flume_endpoint.h"
#include "jml/utils/exc_assert.h"

#include "ThriftFlumeEventServer.h"
#include <protocol/TBinaryProtocol.h>
#include <server/TSimpleServer.h>
#include <server/TThreadedServer.h>
#include <server/TNonblockingServer.h>
#include <server/TThreadPoolServer.h>
#include <transport/TServerSocket.h>
#include <transport/TBufferTransports.h>
//#include <transport/TFramedTransports.h>
#include <iostream>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;
using namespace ::apache::thrift::concurrency;


using namespace std;



namespace Datacratic {


/*****************************************************************************/
/* FLUME RPC ENDPOINT                                                        */
/*****************************************************************************/

struct FlumeRpcEndpoint::Handler
    : virtual public ThriftFlumeEventServerIf,
      public boost::enable_shared_from_this<Handler> {
    
    Handler(FlumeRpcEndpoint * ep)
        : ep(ep)
    {
    }

    void append(const ThriftFlumeEvent& evt)
    {
        if (ep->onFlumeMessage)
            ep->onFlumeMessage(evt.timestamp, evt.priority, evt.body,
                               evt.nanos, evt.host, evt.fields);
    }

    void close()
    {
        if (ep->onClose) ep->onClose();
    }

    FlumeRpcEndpoint * ep;
    boost::shared_ptr<TServer> server;

    boost::shared_ptr<boost::thread> thread;

    void start(int port)
    {

        boost::shared_ptr<TProcessor> processor(new ThriftFlumeEventServerProcessor(shared_from_this()));
        boost::shared_ptr<TServerTransport> serverTransport(new TServerSocket(port));
        boost::shared_ptr<TTransportFactory> transportFactory(new TBufferedTransportFactory());
        boost::shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());
	boost::shared_ptr<ThreadManager> threadManager = ThreadManager::newSimpleThreadManager(15);

        server.reset(new TThreadedServer(processor, serverTransport, transportFactory, protocolFactory));

        //server.reset(new TThreadPoolServer(processor, serverTransport, transportFactory, protocolFactory, threadManager));

        //server.reset(new TNonblockingServer(processor, transportFactory, transportFactory, protocolFactory, protocolFactory, port));

        auto threadMain = [=] ()
            {
                try {
                    this->server->serve();
                } catch (const TTransportException & exc) {
                    ep->recordEvent("transportException");
                    cerr << "got Flume transport exception: " << exc.what()
                    << endl;
                }
            };

        thread.reset(new boost::thread(threadMain));
    }
};

FlumeRpcEndpoint::
FlumeRpcEndpoint()
    : handler(new Handler(this))
{
}

FlumeRpcEndpoint::
FlumeRpcEndpoint(int port)
    : handler(new Handler(this))
{
    init(port);
}

FlumeRpcEndpoint::
~FlumeRpcEndpoint()
{
    shutdown();
}

void
FlumeRpcEndpoint::
init(int port)
{
    handler->start(port);
}
    
void
FlumeRpcEndpoint::
shutdown()
{
    ExcAssert(handler);
    ExcAssert(handler->server);

    handler->server->stop();

    ExcAssert(handler->thread);
    
    handler->thread->join();
}

} // namespace Datacratic
