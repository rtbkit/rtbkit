/* active_endpoint.cc
   Jeremy Barnes, 29 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Active endpoint class.
*/

#include "soa/service//active_endpoint.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace ML;

namespace Datacratic {


/*****************************************************************************/
/* ACTIVE ENDPOINT                                                           */
/*****************************************************************************/

void
ActiveEndpoint::
throwExceptionOnConnectionError(const std::string & error)
{
    throw Exception("Connection error: " + error);
}

void
ActiveEndpoint::
doNothingOnConnectionError(const std::string & error)
{
}

int
ActiveEndpoint::
init(int port, const std::string & hostname,
     int connections, int threads, bool synchronous,
     bool throwIfAnyConnectionError,
     OnConnectionError onEachConnectionError,
     double timeout)
{
    //cerr << "connecting on " << hostname << ":" << port << endl;

    this->port_ = port;
    this->hostname_ = hostname;

    if (hostname == "localhost" || hostname == "")
        addr.set(port);
    else addr.set(port, hostname.c_str());

    //cerr << "connections " << connections << " threads " << threads << endl;

    spinup(threads, synchronous);

    return createConnections(connections, synchronous,
                             throwIfAnyConnectionError,
                             onEachConnectionError,
                             timeout);
}

int
ActiveEndpoint::
createConnections(int nconnections, bool synchronous,
                  bool throwIfAnyConnectionError,
                  OnConnectionError onEachConnectionError,
                  double timeout)
{
    // Pre-create connections so that we don't have to wait for them
    //cerr << "pre-creating " << nconnections << " connections" << endl;

    int finished = 0;
    int errors = 0;
    string first_error;

    //set<int> inProgress;

    auto onConnection
        = [&] (const std::shared_ptr<TransportBase> & transport, int n)
        {
            Guard guard(this->lock);
            //cerr << "transport " << transport << " connected: finished "
            //     << finished << " errors " << errors << endl;
            if (!inactive.count(transport))
                throw Exception("new connection not known in inactive");
            ML::atomic_add(finished, 1);
            //inProgress.erase(n);
        };

    auto onConnectionError2 = [&] (string error, int n)
        {
            cerr << "error creating connection " << n << ": " << error << endl;
            Guard guard(this->lock);
            ML::atomic_add(finished, 1);
            ML::atomic_add(errors, 1);
            if (onEachConnectionError)
                onEachConnectionError(error);
            //inProgress.erase(n);

            if (first_error == "")
                first_error = error;
        };
    
    for (unsigned i = 0;  i < nconnections;  ++i) {
        //{
        //    Guard guard(this->lock);
        //    inProgress.insert(i);
        //}
        
        newConnection(boost::bind<void>(onConnection, _1, i),
                      boost::bind<void>(onConnectionError2, _1, i),
                      timeout);
    }

    while (finished < nconnections) {
        ML::sleep(0.1);
        //int nevents = handleEvents(0.1);
        cerr << "finished " << finished << " of "
             << nconnections << " errors " << errors
             << " connections " << inactive.size()
             << endl;
    }
    
    cerr << inactive.size() << " connections created with "
         << errors << " errors" << endl;

    if ((errors != 0 || inactive.size() != nconnections)
        && throwIfAnyConnectionError)
        throw Exception("error creating connections: " + first_error);
    
    return inactive.size();
}

void
ActiveEndpoint::
getConnection(OnNewConnection onNewConnection,
              OnConnectionError onConnectionError,
              double timeout)
{
    Guard guard(lock);

    if (!inactive.empty()) {
        /* If there's a spare connection then use it */
        std::shared_ptr<TransportBase> result = *inactive.begin();
        
        if (active.count(result))
            throw Exception("doubling up on IDs");
        
        inactive.erase(inactive.begin());
        
        if (active.empty()) idle.acquire();  // no longer idle
        active.insert(result);
        guard.release();

        auto finish = [=] ()
            {
                onNewConnection(result);
            };

        result->doAsync(finish, "getConnection");

        return;
    }

    guard.release();

    auto newOnConnectionAvailable
        = [=] (const std::shared_ptr<TransportBase> & transport)
        {
            Guard guard(lock);

            if (active.count(transport))
                throw Exception("doubling up on IDs 2");
            if (!inactive.count(transport))
                throw Exception("inactive doesn't include the connection");

            inactive.erase(transport);
            if (active.empty()) idle.acquire();
            
            active.insert(transport);

            guard.release();
            onNewConnection(transport);
            return;
        };
    
    newConnection(newOnConnectionAvailable, onConnectionError, timeout);
}

void
ActiveEndpoint::
shutdown()
{
    active.clear();
    inactive.clear();
    EndpointBase::shutdown();
}

void
ActiveEndpoint::
notifyNewTransport(const std::shared_ptr<TransportBase> & transport)
{
    EndpointBase::notifyNewTransport(transport);
}

void
ActiveEndpoint::
notifyTransportOpen(const std::shared_ptr<TransportBase> & transport)
{
#if 0
    cerr << "notifyTransportOpen " << transport << " "
         << transport->status() << endl;

    backtrace();
#endif

    Guard guard(lock);

    if (active.count(transport))
        throw ML::Exception("attempt to add new transport twice");
    if (inactive.count(transport))
        throw ML::Exception("attempt to add new transport %p (%d,%s) twice 2",
                            transport.get(),
                            transport->getHandle(),
                            transport->status().c_str());

    inactive.insert(transport);
}

void
ActiveEndpoint::
notifyCloseTransport(const std::shared_ptr<TransportBase> & transport)
{
    Guard guard(lock);
    if (inactive.count(transport))
        inactive.erase(transport);
    if (active.count(transport))
        active.erase(transport);

    EndpointBase::notifyCloseTransport(transport);

    if (active.empty())
        idle.release();
}

void
ActiveEndpoint::
notifyRecycleTransport(const std::shared_ptr<TransportBase> & transport)
{
    Guard guard(lock);

    if (!active.count(transport))
        throw ML::Exception("recycled transport was not active");
    if (inactive.count(transport))
        throw ML::Exception("recycled transport already inactive");
    if (transport->hasSlave())
        throw ML::Exception("recycled transport has a slave");

    active.erase(transport);
    inactive.insert(transport);

    if (active.empty())
        idle.release();
}

void
ActiveEndpoint::
dumpState() const
{
    Guard guard(lock);
    cerr << endl << endl;
    cerr << "----------------------------------------------" << endl;
    cerr << "Active Endpoint of type " << type_name(*this)
         << " with " << active.size() << " active connections and "
         << inactive.size() << " inactive connections" << endl;

    int i = 0;
    for (auto it = active.begin(), end = active.end();  it != end && i < 10;  ++it,++i) {
        cerr << "  active " << i << (*it)->status() << endl;
        (*it)->activities.dump();
        cerr << endl;
    }
}

} // namespace Datacratic
