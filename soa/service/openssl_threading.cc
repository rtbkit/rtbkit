/* openssl_threading.cc
   Wolfgang Sourdeau, 15 July 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

   This module registers OpenSSL-specific callbacks required to ensure proper
   functioning of the library when used in multi-threaded programs. See
   manpage "CRYPTO_set_dynlock_create_callback" for further details.
*/

#include <pthread.h>
#include <openssl/crypto.h>

#include <mutex>
#include <vector>

#include "jml/arch/threads.h"
#include "openssl_threading.h"


using namespace std;


extern "C" {

struct CRYPTO_dynlock_value {
    std::mutex lock;
};

}


namespace {

std::mutex initLock;
bool threadingInit(false);


/* basic lock callbacks */

pthread_mutex_t * cryptoLocks; /* vector<std::mutex> is not available */

void
lockingFunc(int mode, int n, const char *, int line)
{
    pthread_mutex_t * lock = cryptoLocks + n;

    if (mode & CRYPTO_LOCK) {
        ::pthread_mutex_lock(lock);
    }
    else {
        ::pthread_mutex_unlock(lock);
    }
}

void
threadIdFunc(CRYPTO_THREADID * threadIdPtr)
{
    pid_t tid = getpid();
    CRYPTO_THREADID_set_numeric(threadIdPtr, tid);
}


/* dyn lock callbacks */

struct CRYPTO_dynlock_value *
dynLockCreateFunc(const char * file, int line)
{
    return new CRYPTO_dynlock_value();
}

void
dynLockLockFunc(int mode, CRYPTO_dynlock_value * l, const char *, int)
{
    auto & lock = l->lock;
    if (mode & CRYPTO_LOCK) {
        lock.lock();
    }
    else {
        lock.unlock();
    }
}

void
dynLockDestroyFunc(CRYPTO_dynlock_value * l, const char *, int)
{
    delete l;
}


void
setupCallbacks()
{
    CRYPTO_set_locking_callback(lockingFunc);
    CRYPTO_THREADID_set_callback(threadIdFunc);

    CRYPTO_set_dynlock_create_callback(dynLockCreateFunc);
    CRYPTO_set_dynlock_lock_callback(dynLockLockFunc);
    CRYPTO_set_dynlock_destroy_callback(dynLockDestroyFunc);
}

}

void
Datacratic::
initOpenSSLThreading()
{
    std::unique_lock<std::mutex> guard(initLock);

    if (threadingInit) {
        return;
    }

    size_t maxLocks = CRYPTO_num_locks();
    cryptoLocks = new pthread_mutex_t[maxLocks];
    for (size_t i = 0; i < CRYPTO_num_locks(); i++) {
        ::pthread_mutex_init(&cryptoLocks[i], NULL);
    }

    setupCallbacks();
    threadingInit = true;
}
