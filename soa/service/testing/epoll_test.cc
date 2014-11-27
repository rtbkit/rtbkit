/* epoll_test.cc
   Wolfgang Sourdeau, 17 November 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   Assumption tests for epoll
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <fcntl.h>
#include <unistd.h>
#include <sys/epoll.h>

#include <atomic>
#include <iostream>
#include <thread>

#include <boost/test/unit_test.hpp>

#include <jml/arch/exception.h>
#include <jml/arch/futex.h>
#include <jml/arch/timers.h>
#include <jml/utils/exc_assert.h>

using namespace std;

namespace {

#if 1
/* Assumption test - this disproves the race condition initially suspected
 * with EPOLLONESHOT in the following scenario:

 - armed fd
 - (data received)
 - epoll_wait returns due to armed fd and data received
 - read fd
 - (data received)
 - rearmed fd
 - epoll_wait waits indefinitely due to data emitted before fd rearmed fd
*/

void thread1Fn(atomic<int> & stage, int epollFd, int pipeFds[2])
{
    uint32_t readData;

    struct epoll_event armEvent, event;
    armEvent.events = EPOLLIN | EPOLLONESHOT;

    ::fprintf(stderr, "thread 1: arming fd\n");
    int rc = ::epoll_ctl(epollFd, EPOLL_CTL_ADD, pipeFds[0], &armEvent);
    if (rc == -1) {
        throw ML::Exception(errno, "epoll_ctl");
    }

    stage = 1; ML::futex_wake(stage);

    ::fprintf(stderr, "thread 1: waiting 1\n");
    rc = ::epoll_wait(epollFd, &event, 1, -1);
    if (rc == -1) {
        throw ML::Exception(errno, "epoll_wait");
    }

    ::fprintf(stderr, "thread 1: reading 1\n");
    rc = ::read(pipeFds[0], &readData, sizeof(readData));
    if (rc == -1) {
        throw ML::Exception(errno, "read");
    }
    ExcAssertEqual(rc, sizeof(readData));
    ExcAssertEqual(readData, 1);

    ML::sleep(1.0);

    ::fprintf(stderr, "thread 1: reading 2\n");
    rc = ::read(pipeFds[0], &readData, sizeof(readData));
    if (rc == -1) {
        throw ML::Exception(errno, "read");
    }
    ExcAssertEqual(rc, sizeof(readData));
    ExcAssertEqual(readData, 0x1fffffff);

    ML::sleep(1.0);

    rc = ::read(pipeFds[0], &readData, sizeof(readData));
    ExcAssert(rc == -1);
    ExcAssert(errno == EWOULDBLOCK);

    stage = 2; ML::futex_wake(stage);

    ::fprintf(stderr,
              "thread 1: data read, awaiting final notification from thread"
              " 2\n");
    while (stage.load() != 3) {
        ML::futex_wait(stage, 2);
    }
    ::fprintf(stderr,
              "thread 1: notified of final payload from thread 2\n");
    ML::sleep(1.0);

    ::fprintf(stderr, "thread 1: rearming\n");
    rc = ::epoll_ctl(epollFd, EPOLL_CTL_MOD, pipeFds[0], &armEvent);
    if (rc == -1) {
        throw ML::Exception(errno, "epoll_ctl");
    }

    ::fprintf(stderr, "thread 1: epoll_wait for final payload\n");
    rc = ::epoll_wait(epollFd, &event, 1, 2000);
    if (rc == -1) {
        throw ML::Exception(errno, "epoll_wait");
    }
    else if (rc == 0)
        ::fprintf(stderr, "thread 1: second epoll wait has no event\n");
    else if (rc > 0)
        ::fprintf(stderr, "thread 1: second epoll wait has %d events\n", rc);

    /* This proves that the data written in thread 2 was properly received
       despite being sent before the rearming of our end of the pipe. */
    BOOST_CHECK_EQUAL(rc, 1);
}

void thread2Fn(atomic<int> & stage, int epollFd, int pipeFds[2])
{
    ::fprintf(stderr, "thread 2: initial wait for thread1\n");
    while (stage.load() != 1) {
        ML::futex_wait(stage, 0);
    }

    uint32_t writeData(1);
    int rc = ::write(pipeFds[1], &writeData, sizeof(writeData));
    if (rc == -1) {
        throw ML::Exception(errno, "write");
    }
    ExcAssert(rc == sizeof(writeData));
    ::fprintf(stderr, "thread 2: payload 1 written\n");

    writeData = 0x1fffffff;
    rc = ::write(pipeFds[1], &writeData, sizeof(writeData));
    if (rc == -1) {
        throw ML::Exception(errno, "write");
    }
    ExcAssert(rc == sizeof(writeData));
    ::fprintf(stderr, "thread 2: payload 2 written\n");

    ::fprintf(stderr, "thread 2: waiting thread 1\n");
    while (stage.load() != 2) {
        ML::futex_wait(stage, 1);
    }
    ::fprintf(stderr, "thread 2: thread1 done reading, writing again\n");

    writeData = 0x12345678;
    rc = ::write(pipeFds[1], &writeData, sizeof(writeData));
    if (rc == -1) {
        throw ML::Exception(errno, "write");
    }
    ExcAssert(rc == sizeof(writeData));
    ::fprintf(stderr, "thread 2: payload 3 written\n");

    ::fprintf(stderr, "thread 2: writing complete, notifying thread 1\n");
    stage++; ML::futex_wake(stage);
}

}

BOOST_AUTO_TEST_CASE( test_epolloneshot )
{
    int epollFd = ::epoll_create(666);
    if (epollFd == -1) {
        throw ML::Exception(errno, "epoll_create");
    }

    int pipeFds[2];
    if (::pipe2(pipeFds, O_NONBLOCK) == -1) {
        throw ML::Exception(errno, "pipe2");
    }

    atomic<int> stage(0);

    /* receiving thread */
    auto thread1Lda = [&] () {
        thread1Fn(stage, epollFd, pipeFds);
        ::fprintf(stderr, "thread1 done\n");
    };
    thread thread1(thread1Lda);

    /* sending thread */
    auto thread2Lda = [&] () {
        thread2Fn(stage, epollFd, pipeFds);
        ::fprintf(stderr, "thread 2 done\n");
    };
    thread thread2(thread2Lda);

    thread2.join();
    thread1.join();

    ::close(pipeFds[0]);
    ::close(pipeFds[1]);
    ::close(epollFd);
}
#endif
