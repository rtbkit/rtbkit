/* singleton_loop.cc
   Wolfgang Sourdeau, December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/


#include "async_writer_source.h"
#include "singleton_loop.h"


using namespace std;
using namespace Datacratic;


/****************************************************************************/
/* SINGLETON LOOP ADAPTOR                                                   */
/****************************************************************************/

SingletonLoopAdaptor::
SingletonLoopAdaptor()
    : AsyncWriterSource(nullptr, nullptr, nullptr, 0, 0)
{
}

SingletonLoopAdaptor::
~SingletonLoopAdaptor()
{
}

void
SingletonLoopAdaptor::
addSource(AsyncEventSource & newSource)
{
    int fd = newSource.selectFd();
    auto callback = [&] (const ::epoll_event &) {
        newSource.processOne();
    };
    registerFdCallback(fd, callback);
    addFd(fd, true, false);
}

void
SingletonLoopAdaptor::
removeSource(const AsyncEventSource & source)
{
    int done(false);
    auto onDone = [&] () {
        done = true;
        ML::futex_wake(done);
    };
    int fd = source.selectFd();
    unregisterFdCallback(fd, true, onDone);
    while (!done) {
        ML::futex_wait(done, false);
    }
    removeFd(fd);
}


/****************************************************************************/
/* SINGLETON LOOP                                                           */
/****************************************************************************/

SingletonLoop::
SingletonLoop()
    : started_(false),
      adaptor_(new SingletonLoopAdaptor())
{
}

SingletonLoop::
~SingletonLoop()
{
    shutdown();
}

void
SingletonLoop::
start()
{
    if (!started_) {
        loop_.start();
        loop_.addSource("adaptor", adaptor_);
        started_ = true;
    }
}

void
SingletonLoop::
shutdown()
{
    if (started_) {
        loop_.removeSourceSync(adaptor_.get());
        loop_.shutdown();
        started_ = false;
    }
}

void
SingletonLoop::
addSource(AsyncEventSource & newSource)
{
    adaptor_->addSource(newSource);
}

void
SingletonLoop::
removeSource(const AsyncEventSource & source)
{
    adaptor_->removeSource(source);
}


