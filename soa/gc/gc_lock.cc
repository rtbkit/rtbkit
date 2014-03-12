/* gc_lock.cc
   Jeremy Barnes, 19 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.
*/

#include "soa/gc/gc_lock.h"
#include "jml/arch/tick_counter.h"
#include "jml/arch/spinlock.h"
#include "jml/arch/futex.h"
#include "jml/utils/exc_check.h"
#include "jml/utils/guard.h"

#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/static_assert.hpp>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <iostream>

using namespace std;
using namespace ML;
namespace ipc = boost::interprocess;

namespace Datacratic {


/*****************************************************************************/
/* Utility                                                                   */
/*****************************************************************************/

/** Externally visible initializer for the GcLock's epoch which can be used to
    test for overflows.
*/
int32_t gcLockStartingEpoch = 0;

int32_t SpeculativeThreshold = 5;

/** A safe comparaison of epochs that deals with potential overflows.
    \todo So many possible bit twiddling hacks... Must resist...
*/
template<typename T, size_t Bits = sizeof(T)*8>
static inline
int
compareEpochs (T a, T b)
{
    BOOST_STATIC_ASSERT(Bits >= 2);

    if (a == b) return 0;

    enum { MASK = (3ULL << (Bits - 2)) };

    // We use the last 2 bits for overflow detection.
    //   We don't use T to avoid problems with the sign bit.
    const uint64_t aMasked = a & MASK;
    const uint64_t bMasked = b & MASK;

    // Normal case.
    if (aMasked == bMasked) return a < b ? -1 : 1;

    // Check for overflow.
    else if (aMasked == 0) return bMasked == MASK ? 1 : -1;
    else if (bMasked == 0) return aMasked == MASK ? -1 : 1;

    // No overflow so just compare the masks.
    return aMasked < bMasked ? -1 : 1;
}


/*****************************************************************************/
/* GC LOCK BASE                                                              */
/*****************************************************************************/

struct DeferredEntry1 {
    DeferredEntry1(void (fn) (void *) = 0, void * data = 0)
        : fn(fn), data(data)
    {
    }

    void run()
    {
        fn(data);
    }
        
    void (*fn) (void *);
    void * data;
};

struct DeferredEntry2 {
    DeferredEntry2(void (fn) (void *, void *) = 0, void * data1 = 0,
                   void * data2 = 0)
        : fn(fn), data1(data1), data2(data2)
    {
    }
        
    void run()
    {
        fn(data1, data2);
    }


    void (*fn) (void *, void *);
    void * data1;
    void * data2;
};

struct DeferredEntry3 {
    DeferredEntry3(void (fn) (void *, void *, void *) = 0, void * data1 = 0,
                   void * data2 = 0, void * data3 = 0)
        : fn(fn), data1(data1), data2(data2), data3(data3)
    {
    }
        
    void run()
    {
        fn(data1, data2, data3);
    }


    void (*fn) (void *, void *, void *);
    void * data1;
    void * data2;
    void * data3;
};


/// Data about each epoch
struct GcLockBase::DeferredList {
    DeferredList()
    {
    }

    ~DeferredList()
    {
        //if (lock.locked())
        //    throw ML::Exception("deleting deferred in locked condition");
        if (size() != 0) {
            cerr << "deleting non-empty deferred with " << size()
                 << " entries" << endl;
            //throw ML::Exception("deleting non-empty deferred");
        }
    }

    void swap(DeferredList & other)
    {
        //ExcAssertEqual(lock.locked(), 0);
        //ExcAssertEqual(other.lock.locked(), 0);

        //boost::lock_guard<ML::Spinlock> guard(lock);
        //boost::lock_guard<ML::Spinlock> guard2(other.lock);

        deferred1.swap(other.deferred1);
        deferred2.swap(other.deferred2);
        deferred3.swap(other.deferred3);
    }

    std::vector<DeferredEntry1> deferred1;
    std::vector<DeferredEntry2> deferred2;
    std::vector<DeferredEntry3> deferred3;
    //mutable ML::Spinlock lock;

    bool addDeferred(int forEpoch, void (fn) (void *), void * data)
    {
        //boost::lock_guard<ML::Spinlock> guard(lock);
        deferred1.push_back(DeferredEntry1(fn, data));
        return true;
    }

    bool addDeferred(int forEpoch, void (fn) (void *, void *),
                     void * data1, void * data2)
    {
        //boost::lock_guard<ML::Spinlock> guard(lock);
        deferred2.push_back(DeferredEntry2(fn, data1, data2));
        return true;
    }

    bool addDeferred(int forEpoch, void (fn) (void *, void *, void *),
                     void * data1, void * data2, void * data3)
    {
        //boost::lock_guard<ML::Spinlock> guard(lock);
        deferred3.push_back(DeferredEntry3(fn, data1, data2, data3));
        return true;
    }
        
    size_t size() const
    {
        //boost::lock_guard<ML::Spinlock> guard(lock);
        return deferred1.size() + deferred2.size() + deferred3.size();
    }

    void runAll()
    {
        // Spinlock should be unnecessary...
        //boost::lock_guard<ML::Spinlock> guard(lock);

        for (unsigned i = 0;  i < deferred1.size();  ++i) {
            try {
                deferred1[i].run();
            } catch (...) {
            }
        }

        deferred1.clear();

        for (unsigned i = 0;  i < deferred2.size();  ++i) {
            try {
                deferred2[i].run();
            } catch (...) {
            }
        }

        deferred2.clear();

        for (unsigned i = 0;  i < deferred3.size();  ++i) {
            try {
                deferred3[i].run();
            } catch (...) {
            }
        }

        deferred3.clear();
    }
};

struct GcLockBase::Deferred {
    mutable ML::Spinlock lock;
    std::map<int32_t, DeferredList *> entries;
    std::vector<DeferredList *> spares;

    bool empty() const
    {
        boost::lock_guard<ML::Spinlock> guard(lock);
        return entries.empty();
    }
};

std::string
GcLockBase::ThreadGcInfoEntry::
print() const
{
    return ML::format("inEpoch: %d, readLocked: %d, writeLocked: %d",
                      inEpoch, readLocked, writeLocked);
}

inline GcLockBase::Data::
Data() :
    bits(0), bits2(0)
{
    epoch = gcLockStartingEpoch; // makes it easier to test overflows.
    visibleEpoch = epoch;
}

inline GcLockBase::Data::
Data(const Data & other)
{
    //ML::ticks();
    q = other.q;
    //ML::ticks();
}

inline GcLockBase::Data &
GcLockBase::Data::
operator = (const Data & other)
{
    //ML::ticks();
    this->q = other.q;
    //ML::ticks();
    return *this;
}

void
GcLockBase::Data::
validate() const
{
    try {
        // Visible is at most 2 behind current
        ExcAssertGreaterEqual(compareEpochs(visibleEpoch, epoch - 2), 0);

        // If nothing is in a critical section then only the current is
        // visible
        if (inOld() == 0 && inCurrent() == 0)
            ExcAssertEqual(visibleEpoch, epoch);

        // If nothing is in the old critical section then it's not visible
        else if (inOld() == 0)
            ExcAssertEqual(visibleEpoch, epoch - 1);

        else ExcAssertEqual(visibleEpoch, epoch - 2);
    } catch (const std::exception & exc) {
        cerr << "exception validating GcLock: " << exc.what() << endl;
        cerr << "current: " << print() << endl;
        throw;
    }
}

inline bool
GcLockBase::Data::
calcVisibleEpoch()
{
    Data old = *this;

    int oldValue = visibleEpoch;

    // Set the visible epoch
    if (inCurrent() == 0 && inOld() == 0)
        visibleEpoch = epoch;
    else if (inOld() == 0)
        visibleEpoch = epoch - 1;
    else visibleEpoch = epoch - 2;

    if (compareEpochs(visibleEpoch, oldValue) < 0) {
        cerr << "old = " << old.print() << endl;
        cerr << "new = " << print() << endl;
    }

    // Visible epoch must be monotonic increasing
    ExcAssertGreaterEqual(compareEpochs(visibleEpoch, oldValue), 0);

    return oldValue != visibleEpoch;
}
        
std::string
GcLockBase::Data::
print() const
{
    return ML::format("epoch: %d, in: %d, in-1: %d, visible: %d, exclusive: %d",
                      epoch, inCurrent(), inOld(), visibleEpoch, exclusive);
}

GcLockBase::
GcLockBase()
{
    deferred = new Deferred();
}

GcLockBase::
~GcLockBase()
{
    if (!deferred->empty()) {
        dump();
    }

    delete deferred;
}

bool
GcLockBase::
updateData(Data & oldValue, Data & newValue, RunDefer runDefer)
{
    bool wake;
    try {
        ExcAssertGreaterEqual(compareEpochs(newValue.epoch, oldValue.epoch), 0);
        wake = newValue.calcVisibleEpoch();
    } catch (...) {
        cerr << "update: oldValue = " << oldValue.print() << endl;
        cerr << "newValue = " << newValue.print() << endl;
        throw;
    }

    newValue.validate();

#if 0
    // Do an extra check before we assert lock
    Data upToDate = *data;
    if (upToDate != oldValue) {
        oldValue = upToDate;
        return false;
    }
#endif

    if (!ML::cmp_xchg(data->q, oldValue.q, newValue.q)) return false;

    if (wake) {
        // We updated the current visible epoch.  We can now wake up
        // anything that was waiting for it to be visible and run any
        // deferred handlers.
        futex_wake(data->visibleEpoch);
        if (runDefer) {
            runDefers();
        }
    }

    return true;
}

void
GcLockBase::
runDefers()
{
    std::vector<DeferredList *> toRun;
    {
        boost::lock_guard<ML::Spinlock> guard(deferred->lock);
        toRun = checkDefers();
    }

    for (unsigned i = 0;  i < toRun.size();  ++i) {
        toRun[i]->runAll();
        delete toRun[i];
    }
}

std::vector<GcLockBase::DeferredList *>
GcLockBase::
checkDefers()
{
    std::vector<DeferredList *> result;

    while (!deferred->entries.empty() &&
            compareEpochs(
                    deferred->entries.begin()->first,
                    data->visibleEpoch) <= 0)
    {
        result.reserve(deferred->entries.size());

        for (auto it = deferred->entries.begin(),
                 end = deferred->entries.end();
             it != end;  /* no inc */) {

            if (compareEpochs(it->first, data->visibleEpoch) > 0)
                break;  // still visible

            ExcAssert(it->second);
            result.push_back(it->second);
            //it->second->runAll();
            auto toDelete = it;
            it = boost::next(it);
            deferred->entries.erase(toDelete);
        }
    }

    return result;
}

void
GcLockBase::
enterCS(ThreadGcInfoEntry * entry, RunDefer runDefer)
{
    if (!entry) entry = &getEntry();
        
    ExcAssertEqual(entry->inEpoch, -1);

#if 0 // later...
    // Be optimistic...
    int optimisticEpoch = data->epoch;
    if (__sync_add_and_fetch(data->in + (optimisticEpoch & 1), 1) > 1
        && data->epoch == optimisticEpoch) {
        entry->inEpoch = optimisticEpoch & 1;
        return;
    }

    // undo optimism
    __sync_add_and_fetch(data->in + (optimisticEpoch & 1), -1);
#endif // optimistic

    Data current = *data;

    for (;;) {
        Data newValue = current;

        if (newValue.exclusive) {
            futex_wait(data->exclusive, 1);
            current = *data;
            continue;
        }

        if (newValue.inOld() == 0) {
            // We're entering a new epoch
            newValue.epoch += 1;
            newValue.setIn(newValue.epoch, 1);
        }
        else {
            // No new epoch as the old one isn't finished yet
            newValue.addIn(newValue.epoch, 1);
        }

        entry->inEpoch = newValue.epoch & 1;
            
        if (updateData(current, newValue, runDefer)) break;
    }
}

void
GcLockBase::
exitCS(ThreadGcInfoEntry * entry, RunDefer runDefer /* = true */)
{
    if (entry->inEpoch == -1)
        throw ML::Exception("not in a CS");

    ExcCheck(entry->inEpoch == 0 || entry->inEpoch == 1,
            "Invalid inEpoch");
    // Fast path
    if (__sync_fetch_and_add(data->in + entry->inEpoch, -1) > 1) {
        entry->inEpoch = -1;
        return;
    }

        
    // Slow path; an epoch may have come to an end
    
    Data current = *data;

    for (;;) {
        Data newValue = current;

        //newValue.addIn(entry->inEpoch, -1);

        if (updateData(current, newValue, runDefer)) break;
    }

    entry->inEpoch = -1;
}

void
GcLockBase::
enterCSExclusive(ThreadGcInfoEntry * entry)
{
    ExcAssertEqual(entry->inEpoch, -1);

    Data current = *data, newValue;

    for (;;) {
        if (current.exclusive) {
            futex_wait(data->exclusive, 1);
            current = *data;
            continue;
        }

        ExcAssertEqual(current.exclusive, 0);

        // TODO: single cmp/xchg on just exclusive rather than the whole lot?
        //int old = 0;
        //if (!ML::cmp_xchg(data->exclusive, old, 1))
        //    continue;

        newValue = current;
        newValue.exclusive = 1;
        if (updateData(current, newValue, RD_YES)) {
            current = newValue;
            break;
        }
    }

    ExcAssertEqual(data->exclusive, 1);

    // At this point, we have exclusive access... now wait for everything else
    // to exit.  This is kind of a critical section barrier.
    int startEpoch = current.epoch;
    
#if 1
    visibleBarrier();
#else

    for (unsigned i = 0;  ;  ++i, current = *data) {

        if (current.visibleEpoch == current.epoch
            && current.inCurrent() == 0 && current.inOld() == 0)
            break;
        
        long res = futex_wait(data->visibleEpoch, current.visibleEpoch);
        if (res == -1) {
            if (errno == EAGAIN) continue;
            throw ML::Exception(errno, "futex_wait");
        }
    }
#endif
    
    ExcAssertEqual(data->epoch, startEpoch);


#if 0
    // Testing
    for (unsigned i = 0;  i < 100;  ++i) {
        Data current = *data;

        try {
            ExcAssertEqual(current.exclusive, 1);
            ExcAssertEqual(current.inCurrent(), 0);
            ExcAssertEqual(current.inOld(), 0);
        } catch (...) {
            ThreadGcInfoEntry & entry = getEntry();
            cerr << "entry->inEpoch = " << entry->inEpoch << endl;
            cerr << "entry->readLocked = " << entry->readLocked << endl;
            cerr << "entry->writeLocked = " << entry->writeLocked << endl;
            cerr << "current: " << current.print() << endl;
            cerr << "data: " << data->print() << endl;
            throw;
        }
    }
#endif

    ExcAssertEqual(data->epoch, startEpoch);

    entry->inEpoch = startEpoch & 1;
}

void
GcLockBase::
exitCSExclusive(ThreadGcInfoEntry * entry)
{
    if (!entry) entry = &getEntry();
#if 0
    Data current = *data;

    try {
        ExcAssertEqual(current.exclusive, 1);
        ExcAssertEqual(current.inCurrent(), 0);
        ExcAssertEqual(current.inOld(), 0);
    } catch (...) {
        cerr << "entry->inEpoch = " << entry->inEpoch << endl;
        cerr << "entry->readLocked = " << entry->readLocked << endl;
        cerr << "entry->writeLocked = " << entry->writeLocked << endl;
        cerr << "current: " << current.print() << endl;
        cerr << "data: " << data->print() << endl;
        throw;
    }
#endif

    ML::memory_barrier();

    int old = 1;
    if (!ML::cmp_xchg(data->exclusive, old, 0))
        throw ML::Exception("error exiting exclusive mode");

    // Wake everything waiting on the exclusive lock
    futex_wake(data->exclusive);

    entry->inEpoch = -1;
}

void
GcLockBase::
visibleBarrier()
{
    ML::memory_barrier();
    
    ThreadGcInfoEntry & entry = getEntry();

    if (entry.inEpoch != -1)
        throw ML::Exception("visibleBarrier called in critical section will "
                            "deadlock");

    Data current = *data;
    int startEpoch = data->epoch;
    //int startVisible = data.visibleEpoch;
    
    // Spin until we're visible
    for (unsigned i = 0;  ;  ++i, current = *data) {
        
        //int i = startEpoch & 1;

        // Have we moved on?  If we're 2 epochs ahead we're surely not visible
        if (current.epoch != startEpoch && current.epoch != startEpoch + 1) {
            //cerr << "epoch moved on" << endl;
            return;
        }

        // If there's nothing in a critical section then we're OK
        if (current.inCurrent() == 0 && current.inOld() == 0)
            return;

        if (current.visibleEpoch == startEpoch)
            return;

        if (i % 128 == 127 || true) {
            long res = futex_wait(data->visibleEpoch, current.visibleEpoch);
            if (res == -1) {
                if (errno == EAGAIN) continue;
                throw ML::Exception(errno, "futex_wait");
            }
        }
    }
}

void
GcLockBase::
deferBarrier()
{
    // TODO: what is the interaction between a defer barrier and an exclusive
    // lock?

    ThreadGcInfoEntry & entry = getEntry();

    visibleBarrier();

    // Do it twice to make sure that everything is cycled over two different
    // epochs
    for (unsigned i = 0;  i < 2;  ++i) {
        
        // If we're in a critical section, we'll wait forever...
        ExcAssertEqual(entry.inEpoch, -1);
        
        // What does "defer barrier" mean?  It means that we wait until everything
        // that is currently enqueued to be deferred is finished.
        
        // TODO: this is a very inefficient implementation... we could do a lot
        // better especially in the non-contended case
        
        int lock = 0;
        
        defer(futex_unlock, &lock);
        
        ML::atomic_add(lock, -1);
        
        futex_wait(lock, -1);
    }

    // If certain threads aren't allowed to execute deferred work
    // then it's possible that not all deferred work will have been executed.
    // To be sure, we run any leftover work.
    runDefers();
}

/** Helper function to call an arbitrary boost::function passed through with a void * */
static void callFn(void * arg)
{
    boost::function<void ()> * fn
        = reinterpret_cast<boost::function<void ()> *>(arg);
    try {
        (*fn)();
    } catch (...) {
        delete fn;
        throw;
    }
    delete fn;
}

void
GcLockBase::
defer(boost::function<void ()> work)
{
    defer(callFn, new boost::function<void ()>(work));
}

template<typename... Args>
void
GcLockBase::
doDefer(void (fn) (Args...), Args... args)
{
    // INVARIANT
    // If there is another thread that is provably in a critical section at
    // this moment, then the function will only be run when all such threads
    // have exited the critical section.
    //
    // If there are no threads in either the current or the old epoch, then
    // we can run it straight away.
    //
    // If there are threads in the old epoch but not the current epoch, then
    // we need to wait until all threads have exited the old epoch.  In other
    // words, it goes on the old epoch's defer queue.
    //
    // If there are threads in the current epoch (irrespective of the old
    // epoch) then we need to wait until the current epoch is done.

    Data current = *data;

    int32_t newestVisibleEpoch = current.epoch;
    if (current.inCurrent() == 0) --newestVisibleEpoch;

#if 1
    // Nothing is in a critical section; we can run it inline
    if (current.inCurrent() + current.inOld() == 0) {
        fn(std::forward<Args>(args)...);
        return;
    }
#endif

    for (int i = 0; i == 0; ++i) {
        // Lock the deferred structure
        boost::lock_guard<ML::Spinlock> guard(deferred->lock);

#if 1
        // Get back to current again
        current = *data;

        // Find the oldest live epoch
        int oldestLiveEpoch = -1;
        if (current.inOld() > 0)
            oldestLiveEpoch = current.epoch - 1;
        else if (current.inCurrent() > 0)
            oldestLiveEpoch = current.epoch;
    
        if (oldestLiveEpoch == -1 || 
                compareEpochs(oldestLiveEpoch, newestVisibleEpoch) > 0)
        {
            // Nothing in a critical section so we can run it now and exit
            break;
        }
    
        // Nothing is in a critical section; we can run it inline
        if (current.inCurrent() + current.inOld() == 0)
            break;
#endif

        // OK, get the deferred list
        auto epochIt
            = deferred->entries.insert
            (make_pair(newestVisibleEpoch, (DeferredList *)0)).first;
        if (epochIt->second == 0) {
            // Create a new list
            epochIt->second = new DeferredList();
        }
        
        DeferredList & list = *epochIt->second;
        list.addDeferred(newestVisibleEpoch, fn, std::forward<Args>(args)...);

        // TODO: we only need to do this if the newestVisibleEpoch has
        // changed since we last calculated it...
        //checkDefers();

        return;
    }
    
    // If we got here we can run it straight away
    fn(std::forward<Args>(args)...);
    return;
}

void
GcLockBase::
defer(void (work) (void *), void * arg)
{
    doDefer(work, arg);
}

void
GcLockBase::
defer(void (work) (void *, void *), void * arg1, void * arg2)
{
    doDefer(work, arg1, arg2);
}

void
GcLockBase::
defer(void (work) (void *, void *, void *), void * arg1, void * arg2, void * arg3)
{
    doDefer(work, arg1, arg2, arg3);
}

void
GcLockBase::
dump()
{
    Data current = *data;
    cerr << "epoch " << current.epoch << " in " << current.inCurrent()
         << " in-1 " << current.inOld() << " vis " << current.visibleEpoch
         << " excl " << current.exclusive << endl;
    cerr << "deferred: ";
    {
        boost::lock_guard<ML::Spinlock> guard(deferred->lock);
        cerr << deferred->entries.size() << " epochs: ";
        
        for (auto it = deferred->entries.begin(), end = deferred->entries.end();
             it != end;  ++it) {
            cerr << " " << it->first << " (" << it->second->size()
                 << " entries)";
        }
    }
    cerr << endl;
}


/*****************************************************************************/
/* GC LOCK                                                                   */
/*****************************************************************************/

GcLock::
GcLock()
{
    data = &localData;
}

GcLock::
~GcLock()
{
    // Nothing to cleanup.
}

void
GcLock::
unlink()
{
    // Nothing to cleanup.
}


/*****************************************************************************/
/* SHARED GC LOCK                                                            */
/*****************************************************************************/

// We want to mmap the file so it has to be the size of a page.

namespace { size_t GcLockFileSize = 1ULL << 12; }


GcCreate GC_CREATE; ///< Open and initialize a new gcource.
GcOpen GC_OPEN;     ///< Open an existing gcource.

void
SharedGcLock::
doOpen(bool create)
{
    int flags = O_RDWR | O_CREAT;
    if (create) flags |= O_EXCL;

    ipc::named_mutex mutex(ipc::open_or_create, name.c_str());
    {
        // Lock is used to create and truncate the file atomically.
        ipc::scoped_lock<ipc::named_mutex> lock(mutex);

        // We don't want the locks to be persisted so an shm_open will do fine.
        fd = shm_open(name.c_str(), flags, 0644);
        ExcCheckErrno(fd >= 0, "shm_open failed");

        struct stat stats;
        int res = fstat(fd, &stats);
        ExcCheckErrno(!res, "failed to get the file size");

        if (stats.st_size != GcLockFileSize) {
            int res = ftruncate(fd, GcLockFileSize);
            ExcCheckErrno(!res, "failed to resize the file.");
        }
    }

    // Map the region so that all the processes can see the writes.
    addr = mmap(0, GcLockFileSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ExcCheckErrno(addr != MAP_FAILED, "failed to map the shm file");

    // Initialize and set the member used by GcLockBase.
    if (create) new (addr) Data();
    data = reinterpret_cast<Data*>(addr);
}

SharedGcLock::
SharedGcLock(GcCreate, const string& name) :
    name("gc." + name)
{
    doOpen(true);
}

SharedGcLock::
SharedGcLock(GcOpen, const string& name) :
    name("gc." + name)
{
    doOpen(false);
}

SharedGcLock::
~SharedGcLock()
{
    munmap(addr, GcLockFileSize);
    close(fd);
}

void
SharedGcLock::
unlink()
{
    shm_unlink(name.c_str());
    (void) ipc::named_mutex::remove(name.c_str());
}


} // namespace Datacratic
