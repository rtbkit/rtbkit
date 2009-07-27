/* worker_task.cc
   Jeremy Barnes, 30 October 2005
   Copyright (c) 2005 Jeremy Barnes  All rights reserved.
   $Source$

   Task to perform work.
*/

#include "worker_task.h"
#include "arch/exception.h"
#include "utils/string_functions.h"
#include <iostream>
#include <boost/utility.hpp>
#include "config_impl.h"
#include "utils/environment.h"
#include "utils/info.h"
#include "utils/guard.h"
#include <ace/OS.h>
#include <boost/bind.hpp>


using namespace std;


namespace ML {

Env_Option<int> NUM_THREADS("NUM_THREADS", -1);

int num_threads()
{
    static int num_cpus_saved = num_cpus();

    int result = NUM_THREADS;
    if (result <= 0) result = num_cpus_saved;

    return result;
}


const Job NO_JOB;


/*****************************************************************************/
/* WORKER_TASK                                                               */
/*****************************************************************************/

Worker_Task & Worker_Task::instance(int thr)
{
    static Worker_Task result(thr);
    return result;
}

Worker_Task::Worker_Task(int threads)
    : jobs_sem(0), finished_sem(1), state_change_sem(0),
      next_group(0), next_job(0), num_queued(0),
      num_running(0), force_finished(false)
{
    if (threads == -1)
        threads = num_cpus();

    threads_ = threads;

    //cerr << "creating worker task with " << threads << " threads" << endl;

    /* Create our threads */
    if (threads > 0) {
        int res = activate(THR_DETACHED | THR_NEW_LWP, threads);
        
        //cerr << "res from activate = " << res << endl;
        
        if (res == -1)
            throw Exception("ACE_Task::activate() returned "
                            + ostream_format(res));
    }
}

Worker_Task::~Worker_Task()
{
    /* TODO: finish all tasks */
    if (jobs.size() || groups.size())
        cerr << "at the end, there were " << jobs.size()
             << " jobs outstanding and "
             << groups.size() << " groups outstanding" << endl;
}

Worker_Task::Id
Worker_Task::
get_group(const Job & group_finish, const std::string & info_str,
          Id parent_group, bool locked)
{
    //cerr << "getting group with parent group " << parent_group << endl;

    if (!locked) cerr << "warning: creating unlocked group" << endl;

    Guard guard(lock);
    Id id = next_group++;
    groups[id].finished = group_finish;
    groups[id].parent_group = parent_group;
    groups[id].locked = locked;
    groups[id].info = info_str;

    /* Add a job for the group.  This allows us to keep track of where the child
       jobs get inserted. */
    Job_Info info(group_finish, Job(),
                  format("job for group %lld: ", id) + info_str, -1, id);
    
    Jobs::iterator where = jobs.end();

    /* Insert it at the end of its parent's jobs if it has one. */
    if (parent_group != -1) {
        groups[parent_group].groups_outstanding += 1;
        where = groups[parent_group].group_job;
    }
    
    Jobs::iterator it = jobs.insert(where, info);
    groups[id].group_job = it;
    
    notify_state_changed();
    
    return id;
}

void Worker_Task::unlock_group(int group)
{
    //cerr << "unlocked group " << group << endl;
    Guard guard(lock);
    if (!groups.count(group))
        throw Exception("Worker_Task::unlock_group(): group info has none");
    groups[group].locked = false;
    check_finished(group);

    notify_state_changed();
}

Worker_Task::Id
Worker_Task::
add(const Job & job, const Job & error, const std::string & job_info, Id group)
{
    /* Wait to manupulate */
    Guard guard(lock);
    Job_Info info(job, error, job_info, next_job++, group);

    /* Find where to put it. */

    if (group != -1) {
        if (!groups.count(group))
            throw Exception("Worker_Task::add(): group info has none");

        Group_Info & group_info = groups[group];
        ++group_info.jobs_outstanding;
        
        Jobs::iterator it = jobs.insert(group_info.group_job, info);
    }
    else jobs.push_back(info);
    
    ++num_queued;
    
    jobs_sem.release();  // release one to allow something to run the job
    
    /* If this is the first job running, then we are no longer finished so we
       acquire the finished semaphore. */
    if (num_queued + num_running == 1)
        finished_sem.acquire();

    notify_state_changed();
    
    return info.id;
}

Worker_Task::Id
Worker_Task::
add(const Job & job, const std::string & job_info, Id group)
{
    return add(job, Job(), job_info, group);
}

void Worker_Task::finish_all()
{
    /* Wait until we are finished */
    finished_sem.acquire();

    /* Let something else finish */
    finished_sem.release();
}

void Worker_Task::clear_all()
{
    throw Exception("Worker_Task::clear_all(): not implemented");
}

int Worker_Task::open(void *args)
{
    /* don't need to do anything. */
    return 0;
}

int Worker_Task::close(u_long flags)
{
    /* Again; do nothing. */
    return 1;
}

int Worker_Task::svc()
{
    //cerr << "worker function" << endl;
    
    /* This is the worker function.  We grab work while there is any until it
       is time to exit. */
    
    while (!force_finished) {
        
        Job_Info info = get_job();
        try {
            //cerr << "thread " << ACE_OS::thr_self() << " is running job "
            //     << info.id << " (" << info.info << ")" << endl;
            info.job();
            //cerr << "thread " << ACE_OS::thr_self() << " is finished job "
            //     << info.id << " (" << info.info << ")" << endl;
            finish_job(info);
        }
        catch (const std::exception & exc) {
            // TODO: make this exception go to the calling process
            //cerr << "thread " << ACE_OS::thr_self() << " running job "
            //     << info.id << " (" << info.info << "): " << endl;
            //cerr << "warning: job threw exception: "
            //     << exc.what() << endl;
            try {
                if (info.error) info.error();
            }
            catch (const std::exception & exc) {
                cerr << "warning: job error function throw exception: "
                     << exc.what() << endl;
            }

            /* Indicate that the job's group had an error. */
            if (info.group != -1) {
                Guard guard(lock);
                Group_Info & group_info = groups[info.group];
                group_info.error = true;
                group_info.error_message = exc.what();
                cancel_group(group_info, info.group);
            }

            finish_job(info);
        }
    }
    
    return 0;
}

void Worker_Task::notify_state_changed()
{
    // must be called with the lock held
    for (set<ACE_Semaphore *>::const_iterator it = state_semaphores.begin();
         it != state_semaphores.end();  ++it)
        (*it)->release();
}

void Worker_Task::add_state_semaphore(ACE_Semaphore & sem)
{
    Guard guard(lock);
    if (state_semaphores.count(&sem))
        throw Exception("same state semaphore added twice");
    state_semaphores.insert(&sem);
}

void Worker_Task::remove_state_semaphore(ACE_Semaphore & sem)
{
    Guard guard(lock);
    if (!state_semaphores.count(&sem))
        throw Exception("state semaphore was lost");
    state_semaphores.erase(&sem);
}

void Worker_Task::run_until_released(ACE_Semaphore & sem, int group)
{
    /* We check at every change in state for either a) the semaphore
       being free or b) a job being available. */

    ACE_Semaphore state_semaphore(0);
    add_state_semaphore(state_semaphore);

    /* Make sure we remove this semaphore at the end. */
    Call_Guard guard(boost::bind(&Worker_Task::remove_state_semaphore,
                                 this, boost::ref(state_semaphore)));
    
    while (sem.tryacquire() == -1) {

        /* Run a job, if there is one old job to finish. */
        
        Job_Info info;
        if (try_get_job(info, group)) {
            // release lock here

            try {
                info.job();
                finish_job(info);
            }
            catch (const std::exception & exc) {
                // TODO: make this exception go to the calling process
                cerr << "warning: job threw exception: "
                     << exc.what() << endl;
                try {
                    if (info.error) info.error();
                }
                catch (const std::exception & exc) {
                    cerr << "warning: job error function throw exception: "
                         << exc.what() << endl;
                }
                finish_job(info);
            }

            continue;
        }

        /* Wait for a state change. */
        state_semaphore.acquire();
    }
    
    sem.release();
}

void
Worker_Task::
cancel_group(Group_Info & group_info, int group)
{
    //cerr << "thread " << ACE_OS::thr_self() << " cancel_group() "
    //     << group_info.info << endl;
    
    /* We clean up the group by scanning through its list of tasks, removing
       those that haven't run yet, and calling all of the handlers. */
    
    {
        Guard guard(lock);
        
        Jobs::iterator it = jobs.begin();
        
        /* Iterate through this group's jobs. */
        
        while (it != group_info.group_job) {

            while (it != group_info.group_job
                   && (it->id == -1 || !in_group(*it, group))) ++it;
            
            if (it == group_info.group_job) break;
            
            /* Remove this job. */
            //cerr << "removing job " << it->id << " (" << it->info
            //     << ")" << endl;

            
            Jobs::iterator next = it;
            ++next;
            remove_job(it);
            it = next;
        }
    }
    
    //cerr << "finished clearing jobs for group " << group << endl;
}

void
Worker_Task::
force_finish_group(Group_Info & group_info, int group)
{
    cancel_group(group_info, group);

    /* Wait until everything has stopped running in this group. */

    ACE_Semaphore state_semaphore(0);
    add_state_semaphore(state_semaphore);

    /* Make sure we remove this semaphore at the end. */
    Call_Guard guard(boost::bind(&Worker_Task::remove_state_semaphore,
                                 this, boost::ref(state_semaphore)));
    
    while (group_info.jobs_running > 0)
        state_semaphore.acquire();

    /* Group should be finished. */
}

void
Worker_Task::
run_until_finished(int group, bool unlock)
{
    /* We check at every change in state for either a) the group being
       finished or b) an error from the group or c) a job being available.
    */
    
    map<Id, Group_Info>::iterator group_it;
    
    /* Lock the group so that it doesn't get removed. */
    {
        Guard guard(lock);
        group_it = groups.find(group);
        if (group_it == groups.end()) {
            if (!unlock) return;  // group must have finished
            throw Exception("Worker_Task::run_until_finished(): "
                            "group doesn't exist but should be locked");
        }
        if (group_it->second.locked && !unlock)
            throw Exception("Worker_Task::run_until_finished(): "
                            "group is locked; it won't ever finish");
        group_it->second.locked = true;
    }

    /* The group is locked for now; make sure it will be unlocked at the
       end. */
    Call_Guard unlock_guard(boost::bind(&Worker_Task::unlock_group,
                                        this, group));
    
    /* Since the group is locked, this object must remain in memory here
       and so we don't need to lock to access it.  (A map's iterators
       aren't invalidated unless the actual object itself is removed).
    */
    Group_Info & group_info = group_it->second;

    /* Create a semaphore so that we get notified of state changes. */
    ACE_Semaphore state_semaphore(0);
    add_state_semaphore(state_semaphore);
    
    /* Make sure we remove this semaphore at the end. */
    Call_Guard state_guard(boost::bind(&Worker_Task::remove_state_semaphore,
                                       this, boost::ref(state_semaphore)));
    
    for (;;) {
        //cerr << "thread " << ACE_OS::thr_self() << " is waiting for group "
        //     << group << " to finish" << endl;

        /* Is the group finished?  If so, we can get out of here. */
        {
            if (group_info.error) {
                //cerr << "thread " << ACE_OS::thr_self()
                //     << " had a group error" << endl;
                /* The group had an error.  Wait until there's nothing
                   running, then throw an exception. */

                /* Save the message, as we're about to remove the
                   group. */
                string message = group_info.error_message;
                
                //cerr << "finishing the group..." << endl;

                force_finish_group(group_info, group);

                //cerr << "done finishing group" << endl;

                /* Unlock the group to allow everything to finish. */
                unlock_guard.clear();
                unlock_group(group);

                /* Done; throw the exception. */
                
                throw Exception("Error in worker job: " + message);
            }
            else {
                if (group_info.jobs_outstanding
                    + group_info.jobs_running
                    + group_info.groups_outstanding == 0) {
                    /* We're finished */
                    //cerr << "thread " << ACE_OS::thr_self()
                    //     << " finished all jobs" << endl;
                    return;
                }
            }
        }

        /* Run a job if we can */
        Job_Info info;
        if (try_get_job(info, group)) {

            try {
                //cerr << "thread " << ACE_OS::thr_self() << " is running job "
                //     << info.id << " (" << info.info << ")" << endl;
                info.job();
                //cerr << "thread " << ACE_OS::thr_self() << " finished job "
                //     << info.id << " (" << info.info << ")" << endl;
                finish_job(info);
            }
            catch (const std::exception & exc) {
                // TODO: make this exception go to the calling process
                //cerr << "thread " << ACE_OS::thr_self() << " running job "
                //     << info.id << " (" << info.info << "):" << endl;
                cerr << "warning: job threw exception: "
                     << exc.what() << endl;
                try {
                    if (info.error) info.error();
                }
                catch (const std::exception & exc) {
                    cerr << "warning: job error function throw exception: "
                         << exc.what() << endl;
                }

                /* Indicate that the job's group had an error. */
                if (info.group != -1) {
                    Guard guard(lock);
                    Group_Info & group_info = groups[info.group];
                    group_info.error = true;
                    group_info.error_message = exc.what();
                    cancel_group(group_info, info.group);
                }

                finish_job(info);
            }
            
            continue;  // no state change needed
        }
        
        /* Wait for a state change. */
        state_semaphore.acquire();
    }
}

bool Worker_Task::in_group(const Job_Info & info, int group)
{
    if (group == -1) return true;
    
    /* Check all parents. */
    if (info.group == group) return true;
    else if (info.group == -1) return false;
    else {
        Group_Info * group_info = &groups[info.group];
        
        while (group_info) {
            Id parent = group_info->parent_group;

            if (parent == -1) group_info = 0;
            else {
                if (!groups.count(parent))
                    throw Exception("Worker_Task::in_group(): invalid "
                                    "group number");
                group_info = &groups[parent];
            }
        }
    }

    return false;
}

Worker_Task::Job_Info Worker_Task::get_job(int group)
{
    /* Block until we can acquire a semaphore to have jobs. */
    jobs_sem.acquire();

    return get_job_impl(group);
}

bool Worker_Task::try_get_job(Worker_Task::Job_Info & job, int group)
{
    if (jobs_sem.tryacquire() == -1) return false;
    
    job = get_job_impl(group);
    return true;
}

Worker_Task::Job_Info Worker_Task::get_job_impl(int group)
{
    //cerr << "thread " << ACE_OS::thr_self()
    //     << " is getting a job in group " << group
    //     << endl;

    Guard guard(lock);

    Jobs::iterator it;

    if (group == -1) {
        it = jobs.begin();

        /* Try to find one whose id isn't -1 but which is in the group
           anyway. */
        while (it != jobs.end() && it->id == -1) ++it;
        
        if (it == jobs.end())
            throw Exception("get_job(): internal error: "
                            "semaphore acquired with zero jobs");

    }
    else {
        map<Id, Group_Info>::const_iterator group_it
            = groups.find(group);
        if (group_it == groups.end()) {
            //throw Exception("iterator not found in group");
            cerr << "coldn't find group " << group << " in  list"
                 << endl;
            return get_job_impl(-1);
        }
        else if (group_it->second.error)
            return get_job_impl(-1);  // group has an error; we don't do it
        else it = jobs.begin();
        
        /* Try to find one whose id isn't -1 but which is in the group
           anyway. */
        while (it != group_it->second.group_job
               && (it->id == -1 || !in_group(*it, group)))
            ++it;
        
        /* If we didn't find one in our group, we select any job at all
           so that we make some progress. */
        if (it == group_it->second.group_job || !in_group(*it, group))
            return get_job_impl(-1);
    }
    
    Job_Info result = *it;
    ++num_running;

    if (result.group != -1) ++groups[result.group].jobs_running;
    
    --num_queued;
    jobs.erase(it);

    notify_state_changed();

    return result;
}

void Worker_Task::finish_job(const Job_Info & info)
{
    Guard guard(lock);
    --num_running;
    
    /* Finish off the group if we need to. */
    Id group = info.group;
    if (group != -1) {

        if (!groups.count(group))
            throw Exception("Worker_Task::finish_job(): invalid group number");

        Group_Info * group_info = &groups[group];

        --group_info->jobs_outstanding;
        --group_info->jobs_running;

        if (group_info->jobs_outstanding < 0)
            throw Exception("Worker_Task::finish_job(): "
                            "group has negative outstanding count");

        if (group_info->jobs_running < 0)
            throw Exception("Worker_Task::finish_job(): "
                            "group has negative running count");
        
        check_finished(group);
    }
    
    if (num_queued + num_running == 0) finished_sem.release();

    notify_state_changed();
}

void Worker_Task::remove_job(const Jobs::iterator & it)
{
    Guard guard(lock);
    --num_queued;
    
    if (jobs_sem.tryacquire() == -1) {
        cerr << "Worker_Task::remove_job(): couldn't acquire the job "
             << "semaphore" << endl;
        abort();
    }

    /* Finish off the group if we need to. */
    Id group = it->group;
    if (group != -1) {

        if (!groups.count(group))
            throw Exception("Worker_Task::finish_job(): invalid group number");

        Group_Info * group_info = &groups[group];

        --group_info->jobs_outstanding;

        if (group_info->jobs_outstanding < 0)
            throw Exception("Worker_Task::finish_job(): "
                            "group has negative outstanding count");
        
        check_finished(group);
    }

    jobs.erase(it);
    
    if (num_queued + num_running == 0) finished_sem.release();
    
    notify_state_changed();
}

bool Worker_Task::check_finished(Id group)
{
    Guard guard(lock);
    
    if (!groups.count(group))
        throw Exception("Worker_Task::finish_job(): invalid group number");
    
    Group_Info * group_info = &groups[group];

    /* Go through the list of parents and notify everywhere of what is
       finished. */
    while (group_info && group_info->jobs_outstanding == 0
           && group_info->groups_outstanding == 0) {
        //cerr << "  *** yes, group " << group << " is finished" << endl;

        if (group_info->locked) return false;

        //cerr << "finished group " << group << endl;

        try {
            if (!group_info->finished.empty() && !group_info->error)
                group_info->finished();
        }
        catch (const std::exception & exc) {
            cerr << "Worker_Task::check_finished(): " << exc.what() << endl;
        }
        
        /* Get rid of the job for the group. */
        jobs.erase(group_info->group_job);

        Id parent = group_info->parent_group;

        if (parent == -1) group_info = 0;
        else {
            if (!groups.count(parent))
                throw Exception("Worker_Task::finish_job(): invalid "
                                "group number");
            group_info = &groups[parent];
            --group_info->groups_outstanding;

            if (group_info->groups_outstanding < 0) {
                dump();
                cerr << "check_finished(" << group << "): "
                     << "groups_outstainding < 0" << endl;
                cerr << "parent = " << parent << endl;
            }
        }
        groups.erase(group);
        group = parent;

        notify_state_changed();
    }
    
    return false;
}

int Worker_Task::queued() const
{
    return num_queued;
}

int Worker_Task::running() const
{
    return num_running;
}

int Worker_Task::finished() const
{
    return next_job - num_running - num_queued;
}

void
Worker_Task::Job_Info::
dump(std::ostream & stream, int indent) const
{
    string i(indent, ' ');
    stream << i << "Job_Info @ " << this << endl;
    stream << i << "  id         = " << id << endl;
    stream << i << "  group      = " << group << endl;
    stream << i << "  info       = " << info << endl;
    stream << i << "  job set    = " << (bool)job << endl;
    stream << i << "  error set  = " << (bool)error << endl;
}

void
Worker_Task::Group_Info::
dump(std::ostream & stream, int indent) const
{
    string i(indent, ' ');
    stream << i << "Group_Info @ " << this << endl;
    stream << i << "  info               = " << info << endl;
    stream << i << "  jobs outstanding   = " << jobs_outstanding << endl;
    stream << i << "  jobs running       = " << jobs_running << endl;
    stream << i << "  groups outstanding = " << groups_outstanding << endl;
    stream << i << "  parent group       = " << parent_group << endl;
    stream << i << "  group job          = " << &(*group_job) << endl;
    stream << i << "  locked             = " << locked << endl;
    stream << i << "  error              = " << error << endl;
    stream << i << "  error message      = " << error_message << endl;
    stream << i << "  finished set       = " << (bool)finished << endl;
}

void
Worker_Task::
dump() const
{
    std::ostream & stream = cerr;

    stream << "Worker_Task @ " << this << endl;
    stream << "  number of jobs   = " << jobs.size() << endl;
    stream << "  next group       = " << next_group << endl;
    stream << "  next job         = " << next_job << endl;
    stream << "  num queued       = " << num_queued << endl;
    stream << "  num running      = " << num_running << endl;
    stream << "  number of groups = " << groups.size() << endl;
    stream << "  state semaphores = " << state_semaphores.size() << endl;
    stream << "  force finishned  = " << force_finished << endl;
    stream << endl;
    stream << "  jobs:" << endl;
    int i = 0;
    for (Jobs::const_iterator it = jobs.begin();  it != jobs.end();  ++it, ++i) {
        stream << "   " << i << ":" << endl;
        it->dump(cerr, 4);
    }
    stream << "  groups:" << endl;
    for (map<Id, Group_Info>::const_iterator it = groups.begin();
         it != groups.end();  ++it) {
        stream << "   group with ID " << it->first << ":" << endl;
        it->second.dump(cerr, 4);
    }
    stream << endl;
}

} // namespace ML
