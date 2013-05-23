/* worker_task.h                                                   -*- C++ -*-
   Jeremy Barnes, 30 October 2005
   Copyright (c) 2005 Jeremy Barnes  All rights reserved.
   $Source$

   Task list of work to do.
*/

#ifndef __boosting__worker_task_h__
#define __boosting__worker_task_h__

#include "jml/utils/guard.h"
#include "jml/arch/format.h"
#include "jml/arch/spinlock.h"
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <ace/Synch.h>
#include <ace/Token.h>
#include <ace/Task.h>
#include <list>
#include <map>
#include <mutex>
#include <set>
#include <thread>


namespace ML {

/** This task represents a job that is run on a thread somewhere.  It is
    simply a function with no arguments; the arguments need to be encoded
    elsewhere.
*/

typedef boost::function<void ()> Job;

/** The ACE semaphores are not useful, as two tryacquire operations performed
    in parallel with 2 free semaphores (eg, enough for both to acquire the
    semaphore) can cause only one to succeed.  This is because the tryacquire
    operation performs a tryacquire on the semaphore's mutex, which just
    protects the internal state, and doesn't retry afterwards.

    We create this class here with a useful tryacquire behaviour.

    TODO: this is suboptimal due to the calls to gettimeofday(); we should
    use something that directly implements semaphores using the pthread
    primitives, or attempt to re-write the ACE implementation.
*/
struct Semaphore : public ACE_Semaphore {
    Semaphore(int initial_count = 1)
        : ACE_Semaphore(initial_count)
    {
    }

    int acquire()
    {
        return ACE_Semaphore::acquire();
    }

    int tryacquire()
    {
        return ACE_Semaphore::tryacquire();

        ACE_Time_Value tv = ACE_OS::gettimeofday();
        int result = ACE_Semaphore::acquire(tv);
        if (result == 0) return result;
        if (result == -1 && errno == ETIME) return 0;
        return -1;
    }

    int release()
    {
        return ACE_Semaphore::release();
    }
};

struct Release_Sem {
    Release_Sem(Semaphore & sem)
        : sem(&sem)
    {
    }

    void operator () ()
    {
        sem->release();
    }

    Semaphore * sem;
};

extern const Job NO_JOB;

/** Return the number of threads that we run with.  If the NUM_THREADS
    environment variable is set, we use that.  Otherwise, we use the
    number of CPU cores that we have available.
*/
int num_threads();


/*****************************************************************************/
/* WORKER_TASK                                                               */
/*****************************************************************************/

/* This class is a generic class that works through a list of jobs.
   The jobs can be arranged in groups, with a job that gets run once the
   group is finished, and the groups can be arranged in a hierarchy.

   The jobs are scheduled such that with any group, jobs belonging to an
   earlier subgroup are all scheduled before any belonging to a later
   subgroup, regardless of the order in which they are scheduled.  (This
   corresponds to a depth first search through the group tree).  The effect
   is to guarantee that the average number of groups outstanding will be
   as small as possible.

   It works multithreaded, and deals with all locking and unlocking.
*/

class Worker_Task : public ACE_Task_Base {
public:
    typedef long long Id;  // 64 bits so no wraparound

    /** Return the instance.  Creates it with the number of threads given,
        or num_threads() if thr == -1.  If thr == 0, then only local work
        is done (no work is transferred to other threads). */
    static Worker_Task & instance(int thr = -1);

    Worker_Task(int threads);

    virtual ~Worker_Task();
    
    int threads() const { return threads_; }

    /** Allocate a new job group.  The given job will be called once the group
        is finished.  Note that if nothing is ever added to the group, it won't
        be finished automatically unless check_finished() is called.

        It will be inserted into the jobs list just after the last job with
        the same parent group, so that the children of parent groups will be
        completed in preference to their sisters.

        If lock is set to true, then it will not ever be automatically removed
        until it is unlocked.  This stops a newly-created group from being
        instantly removed.
    */
    Id get_group(const Job & group_finish,
                 const std::string & info,
                 Id parent_group = -1,
                 bool lock = true);

    /** Unlock the group so that it can be removed. */
    void unlock_group(int group);

    /** Run a set of jobs in multiple threads. */
    template<typename It, typename It2, typename Fn>
    void do_group(It first, It2 last, Fn doWork, int parent = -1,
                 std::string groupName = "", std::string jobName = "")
    {
        int group;
        {
            int parent = -1;  // no parent group
            group = get_group(NO_JOB, groupName, parent);
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         this,
                                         group));
            
            for (int i = 0; first != last;  ++first, ++i)
                add(boost::bind<void>(doWork, first),
                    jobName + ML::format("%d", i),
                    group);
        }

        run_until_finished(group);
    }

private:
    /** Add a job that belongs to the given group.  Jobs which are scheduled into
        the same group will be scheduled together.  If there is an exception or
        an error, the error job will be called.
    */
    Id add(const Job & job, const Job & error,
           const std::string & info, Id group = -1);
public:

    /** Add a job that belongs to the given group.  Jobs which are scheduled into
        the same group will be scheduled together. */
    Id add(const Job & job, const std::string & info, Id group = -1);

    /** Check if a group is finished, and if so call its finish job. */
    bool check_finished(Id group);

    void finish_all();
    
    void clear_all();

    /** Return the number of jobs currently waiting. */
    int queued() const;

    /** Return the number of jobs currently running. */
    int running() const;

    /** Return the number of jobs that have finished. */
    int finished() const;

    /** This function lends the calling thread to the worker task until the
        given semaphore is released.  The semaphore will be checked on each
        state change.  If the group argument is given, then this thread
        will only schedule jobs from the given group (or its children), which
        can be used to avoid the thread being used for other work in the
        task.

        If any of the jobs throw an exception, then another exception will
        be thrown from the given job.
    */
    void run_until_released(Semaphore & sem, int group = -1);

    /** Lend the calling thread to the worker task until the given group
        has finished.

        An exception in a group job is handled by throwing an exception from
        this function.
    */
    void run_until_finished(int group, bool unlock = false);

    /** Lend the calling thread to the worker task for a single job in the
        given group and then return.

        An exception in a group job is handled by throwing an exception from
        this function.
    */
    void lend_thread(int group);

    /** ACE_Task_Base methods. */
    virtual int open(void *args = 0);

    virtual int close(u_long flags = 0);
    
    virtual int svc();

private:
    int threads_;
    
    struct Job_Info;

    typedef std::list<Job_Info> Jobs;
    
    struct Job_Info {
        Job_Info() : id(-1), group(-1) {}
        Job_Info(const Job & job, const Job & error,
                 const std::string & info, Id id, Id group = -1)
            : job(job), error(error), id(id), group(group), info(info) {}
        Job job;
        Job error;
        Id id;    // if -1, this is a group end marker
        Id group;
        std::string info;
        void dump(std::ostream & stream, int indent = 0) const;
    };

    struct Group_Info {
        Group_Info()
            : jobs_outstanding(0), jobs_running(0),
              groups_outstanding(0), parent_group(0),
              locked(false), error(false)
        {
        }

        Job finished;
        int jobs_outstanding;      ///< Number of jobs waiting for
        int jobs_running;          ///< Number of jobs that are running
        int groups_outstanding;    ///< Number of groups waiting for
        Id parent_group;           ///< Group to notify when finished
        Jobs::iterator group_job;  ///< Job for the group; always last
        bool locked;
        bool error;                ///< No further jobs can be run
        std::string error_message; ///< Error message to throw
        std::string info;

        void dump(std::ostream & stream, int indent = 0) const;
    };

    /** Get a job. */
    Job_Info get_job(int group = -1);

    /** Tries to get a job, but doesn't fail if there isn't one. */
    bool try_get_job(Job_Info & info, int group = -1);

    /** Implementation of the get_job methods.  Requires that the jobs_sem
        be acquired. */
    Job_Info get_job_impl(int group);
    
    /** Unlocked implementation of the get_job methods.  Requires that the
        jobs_sem be acquired and that the lock already be held. */
    Job_Info get_job_impl_ul(int group);
    
    void finish_job(const Job_Info & info);

    void remove_job_ul(const Jobs::iterator & it);

    void add_state_semaphore(Semaphore & sem);

    void remove_state_semaphore(Semaphore & sem);

    void notify_state_changed();

    // Check_finished, bit without the lock held
    bool check_finished_ul(Id group);

    /** Is the given job in the group?  Searches up the parent hierarchy. */
    bool in_group(const Job_Info & info, int group);

    /** Removes all queued jobs in the group.  Running jobs are left alone. */
    void cancel_group(Group_Info & group_info, int group);

    /** Removes all queued jobs in the group.  Running jobs are left alone. 
        Lock must already be held. */
    void cancel_group_ul(Group_Info & group_info, int group);

    /** Removes all queued jobs in the group, and waits for running jobs to
        finish.
    */
    void force_finish_group(Group_Info & group_info, int group);

    typedef std::mutex Lock;
    //typedef Spinlock Lock;
    typedef std::unique_lock<Lock> Guard;

    Semaphore jobs_sem, finished_sem, state_change_sem, shutdown_sem;

    /** Jobs we are running. */
    Jobs jobs;
    Id next_group;
    Id next_job;
    int num_queued;
    int num_running;
    Lock lock;

    /** Groups that are currently running. */
    std::map<Id, Group_Info> groups;

    /** Semaphores that get released on each state change. */
    std::set<Semaphore *> state_semaphores;

    volatile bool force_finished;

    /* Dump everything to cerr; for debugging */
    void dump() const;
};

/** Run a set of jobs in multiple threads.  The iterator will be iterated
    through the range and the doWork function will be called with each
    value of the iterator in a different thread.
*/
template<typename It, typename It2, typename Fn>
void run_in_parallel(It first, It2 last, Fn doWork, int parent = -1,
                     std::string groupName = "", std::string jobName = "",
                     Worker_Task & worker
                         = Worker_Task::instance(num_threads() - 1))
{
    worker.do_group(first, last, doWork, parent, groupName, jobName);
}

template<typename RAIt, typename RAIt2, typename Fn>
void run_in_parallel_blocked(RAIt first, RAIt2 last,
                             Fn doWork,
                             int parent = -1,
                             std::string groupName = "",
                             std::string jobName = "",
                             Worker_Task & worker
                                 = Worker_Task::instance(num_threads() - 1))
{
    int numJobs = last - first;
    if (numJobs < 16 * num_threads()) {
        // less than 16 jobs per thread... do it directly
        worker.do_group(first, last, doWork, parent, groupName, jobName);
    }
    else {
        // Split into sub-groups such that we have roughly 16 jobs per
        // thread
        int numPerGroup = numJobs / num_threads() / 16;
        int numGroups
            = (numJobs / numPerGroup)
            + (numJobs % numPerGroup != 0); // extra if there is a remainder

        auto doGroup = [&] (int groupNum)
            {
                RAIt it = first + groupNum * numPerGroup;
                for (int i = 0;  i < numPerGroup && it != last;  ++i, ++it)
                    doWork(it);
            };
        
        worker.do_group<int>(0, numGroups, doGroup, parent, groupName, jobName);
    }
}


} // namespace ML



#endif /* __boosting__worker_task_h__ */
