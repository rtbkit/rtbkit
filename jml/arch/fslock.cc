/* fs_lock.cc
   Wolfgang Sourdeau, December 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "threads.h"
#include "exception.h"
#include "fslock.h"

using namespace std;
using namespace ML;


namespace {

struct Call_Guard {
    
    typedef std::function<void ()> Fn;

    Call_Guard(const Fn & fn)
        : fn_(fn)
    {
    }

    ~Call_Guard()
    {
        if (fn_) fn_();
    }

private:
    Fn fn_;
};

}

/****************************************************************************/
/* GUARDED FS LOCK                                                          */
/****************************************************************************/

GuardedFsLock::
GuardedFsLock(const string & filename)
    noexcept
  : lockname(filename + ".lock"), locked(false), mutex_(nullptr), fd_(-1)
{
}

GuardedFsLock::
GuardedFsLock(GuardedFsLock && other)
    noexcept
  : lockname(move(other.lockname)),
    locked(other.locked), mutex_(other.mutex_),
    fd_(other.fd_)
{
    other.locked = false;
    other.mutex_ = nullptr;
    other.fd_ = -1;
}

GuardedFsLock::
~GuardedFsLock()
    noexcept
{
    unlock();
    if (mutex_ != nullptr) {
        ::munmap(mutex_, sizeof(pthread_mutex_t));
    }
    if (fd_ != -1) {
        ::close(fd_);
    }
}

void
GuardedFsLock::
lock()
{
    if (!mutex_) {
        initMutex();
    }

    while (!locked) {
        int error = ::pthread_mutex_lock(mutex_);
        if (error == 0) {
            locked = true;
        }
        else if (error == EOWNERDEAD) {
            recoverMutex();
        }
        else {
            throw ML::Exception(error, "pthread_mutex_lock");
        }
    }
}

bool
GuardedFsLock::
tryLock()
{
    if (!mutex_) {
        initMutex();
    }

    if (!locked) {
    recovered:
        int error = ::pthread_mutex_trylock(mutex_);
        if (error == 0) {
            locked = true;
        }
        else if (error == EOWNERDEAD) {
            recoverMutex();
            goto recovered;
        }
        else if (error != EBUSY) {
            throw ML::Exception(error, "pthread_mutex_trylock");
        }
    }

    return locked;
}

void
GuardedFsLock::
unlock()
    noexcept
{
    if (locked) {
        int error = ::pthread_mutex_unlock(mutex_);
        if (error != 0) {
            perror("pthread_mutex_unlock");
            abort();
        }
        locked = false;
    }
}

/* Execute the sequence required to obtain the mutex. */
void
GuardedFsLock::
initMutex()
{
    loadMutexFile();
    if (!mutex_) {
        createMutexFile();
        if (!mutex_) {
            loadMutexFile();
            if (!mutex_) {
                throw ML::Exception("mutex could not be loaded nor"
                                    " created");
            }
        }
    }
}

/* Attempt to create the mutex atomically by creating a temporary file and
 * hardlinking it to the mutex file. If the "link" fails, our version is
 * cancelled and the other version is used instead. This guarantees that the
 * lock file is fully initialized when present. */
void
GuardedFsLock::
createMutexFile()
{
    bool success(false);

    /* create the file */
    string tmpLock(lockname + "-" + to_string(gettid()));
    int fd = ::open(tmpLock.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    if (fd == -1) {
        throw ML::Exception(errno, "open");
    }

    Call_Guard fdGuard([&] () {
        if (!success) {
            ::close(fd);
        }
        ::unlink(tmpLock.c_str());
    });

    /* mmap the file */
    if (::ftruncate(fd, sizeof(pthread_mutex_t)) == -1) {
        throw ML::Exception(errno, "ftruncate");
    }
    pthread_mutex_t * newMutex
        = (pthread_mutex_t *) ::mmap(nullptr, sizeof(pthread_mutex_t),
                                     PROT_READ | PROT_WRITE, MAP_SHARED,
                                     fd, 0);
    if (newMutex == MAP_FAILED) {
        throw ML::Exception(errno, "mmap");
    }

    Call_Guard mmapGuard([&] () {
        if (!success) {
            ::munmap(newMutex, sizeof(pthread_mutex_t));
        }
    });

    /* init the mutex */
    int error;
    pthread_mutexattr_t mutexattr;

    error = ::pthread_mutexattr_init(&mutexattr);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutexattr_init");
    }
    error = ::pthread_mutexattr_settype(&mutexattr, PTHREAD_MUTEX_DEFAULT);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutexattr_settype");
    }
    error = ::pthread_mutexattr_setpshared(&mutexattr, 1);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutexattr_setpshared");
    }
    error = ::pthread_mutexattr_setrobust(&mutexattr, PTHREAD_MUTEX_ROBUST);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutexattr_setrobust");
    }

    error = ::pthread_mutex_init(newMutex, &mutexattr);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutexattr_init");
    }

    Call_Guard mtxGuard([&] () {
        if (!success) {
            ::pthread_mutex_destroy(newMutex);
        }
    });

    /* take ownership */
    if (link(tmpLock.c_str(), lockname.c_str()) == 0) {
        success = true;
        mutex_ = newMutex;
        fd_ = fd;
    }
}

void
GuardedFsLock::
loadMutexFile()
{
    int fd = ::open(lockname.c_str(), O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH | S_IWOTH);
    if (fd == -1) {
        if (errno != ENOENT) {
            throw ML::Exception(errno, "open");
        }
    }
    else {
        void *mutex = ::mmap(nullptr, sizeof(pthread_mutex_t),
                             PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mutex == MAP_FAILED) {
            ::close(fd);
            throw ML::Exception(errno, "mmap");
        }
        fd_ = fd;
        mutex_ = (pthread_mutex_t *) mutex;
    }
}

/* Recover a mutex owned by a dead thread. */
void
GuardedFsLock::
recoverMutex()
{
    int error = ::pthread_mutex_consistent(mutex_);
    if (error != 0) {
        throw ML::Exception(error, "pthread_mutex_consistent");
    }
    error = ::pthread_mutex_unlock(mutex_);
    if (error != 0 && error != EPERM) {
        throw ML::Exception(error, "pthread_mutex_unlock");
    }
}
