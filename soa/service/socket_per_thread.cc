/* socket_per_thread.cc
   Jeremy Barnes, 5 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   One socket per thread.
*/

#include "soa/service/socket_per_thread.h"
#include "jml/arch/format.h"
#include "ace/OS_NS_Thread.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/spinlock.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/utils/exc_assert.h"
#include "soa/service/zmq_utils.h"


using namespace std;
using namespace ML;


namespace Datacratic {

/*****************************************************************************/
/* SOCKET PER THREAD                                                         */
/*****************************************************************************/

SocketPerThread::
SocketPerThread()
    : context(0), type(0), numOpen(0), state(NOTINITIALIZED),
      entries(onFreeEntry)
{
}

SocketPerThread::
SocketPerThread(zmq::context_t & context,
                int type,
                const std::string & uri,
                bool allowForceClose)
    : context(&context), type(type), uri(uri),
      allowForceClose(allowForceClose), numOpen(0),
      state(READY), entries(onFreeEntry)
{
    //std::cerr << "finished init of socket " << uri << std::endl;
}

SocketPerThread::
~SocketPerThread()
{
    shutdown();
}

void
SocketPerThread::
init(zmq::context_t & context,
     int type,
     const std::string & uri,
     bool allowForceClose)
{
    if (this->context)
        throw ML::Exception("attempt to double initialize a SocketPerThread");
    this->context = &context;
    this->type = type;
    this->uri = uri;
    this->allowForceClose = allowForceClose;

    state = READY;
    //using namespace std;
    //cerr << "initializing SocketPerThread with uri " << uri << endl;
}

void
SocketPerThread::
shutdown()
{
    state = FINISHED;

    entries.reset();
    context = 0;
    //using namespace std;
    //cerr << "destroying SocketPerThread with uri " << uri << " and "
    //     << numOpen << " open entries" << endl;

    if (numOpen > 0) {
        if (!allowForceClose) {
            throw ML::Exception("attempt to destroy SocketPerThread with %d open entries",
                                numOpen);
        }

        while (!allThreads.empty()) {
            auto it = allThreads.begin();
            onFreeEntry(*it);
        }

        ExcAssertEqual(numOpen, 0);
    }
}

void
SocketPerThread::
initForThisThread() const
{
    if (entries.get())
        return;

    //cerr << "initializing zeromq socket for this thread to connect to "
    //     << uri << endl;

    if (!context)
        throw ML::Exception("attempt to use a SocketPerThread "
                            "without initializing");

    auto mThis = const_cast<SocketPerThread *>(this);

    std::auto_ptr<Entry> newEntry(new Entry());
    newEntry->owner = mThis;
            
    std::auto_ptr<zmq::socket_t> newPtr
        (new zmq::socket_t(*context, type));
    setIdentity(*newPtr, ML::format("thr%lld",
                                    (long long)ACE_OS::thr_self()));
    newPtr->connect(uri.c_str());

    newEntry->sock = newPtr.release();
    //if (!allForThread.get())
    //    allForThread.reset(new std::set<SocketPerThread *>());
    //allForThread->insert(mThis);
    entries.reset(newEntry.release());

    mThis->addThreadEntry(entries.get());
    ML::atomic_inc(numOpen);

    // wait for ZMQ when connecting...
    ML::sleep(0.1);
}
    
void
SocketPerThread::
onFreeEntry(Entry * entry)
{
    using namespace std;
    //cerr << "onFreeEntry " << entry << endl;
    //cerr << "closing zmq socket" << entry->sock << " with owner "
    //     << entry->owner << endl;
    delete entry->sock;
    //cerr << "erasing" << endl;
    //allForThread->erase(entry->owner);
    //cerr << "unowning" << endl;
    ML::atomic_dec(entry->owner->numOpen);

    entry->owner->removeThreadEntry(entry);

    delete entry;

    //Datacratic::close(*sock);
    //ML::backtrace();
    //cerr << endl << endl;
}

void
SocketPerThread::
cleanupThisThread()
{
    //cerr << "cleaning up socket " << entries->sock << endl;
    entries.reset();
}

void
SocketPerThread::
cleanupAllForThread()
{
    if (!allForThread.get()) return;
    //cerr << "cleaning up " << allForThread->size() << " sockets" << endl;

    for (auto it = allForThread->begin();  it != allForThread->end();
         it = allForThread->begin())
        (*it)->cleanupThisThread();

    //cerr << "we now have " << allForThread->size() << " sockets left" << endl;
}

void
SocketPerThread::
removeThreadEntry(Entry * entry)
{
    Guard guard(allThreadsLock);
    allThreads.erase(entry);
}

void
SocketPerThread::
addThreadEntry(Entry * entry)
{
    Guard guard(allThreadsLock);
    allThreads.insert(entry);
}

boost::thread_specific_ptr<std::set<SocketPerThread *> >
SocketPerThread::allForThread;

} // namespace Datacratic
