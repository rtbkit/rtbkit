/* fs_lock.h                                                         -*-C++-*-
   Wolfgang Sourdeau, December 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include <pthread.h>
#include <string>


namespace ML {

/*****************************************************************************/
/* GUARDEDFSLOCK                                                             */
/*****************************************************************************/

struct GuardedFsLock
{
    explicit GuardedFsLock(const std::string & filename) noexcept;
    GuardedFsLock(GuardedFsLock && other) noexcept;

    ~GuardedFsLock() noexcept;

    /** obtain the lock and block until it is acquired */
    void lock();

    /** attempt to acquire the lock and returns whether it succeeded */
    bool tryLock();

    /** release the lock */
    void unlock() noexcept;

    /** the lock file name */
    std::string lockname;

    /** flag indicating whether we currently own the lock */
    bool locked;

    /* disabled functions */
    GuardedFsLock() = delete;
    GuardedFsLock(const GuardedFsLock & other) = delete;
    GuardedFsLock & operator = (GuardedFsLock & other) = delete;

private:
    void initMutex();
    void createMutexFile();
    void loadMutexFile();
    void recoverMutex();

    pthread_mutex_t *mutex_;
    int fd_;
};

}
