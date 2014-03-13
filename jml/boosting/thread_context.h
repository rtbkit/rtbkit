/* thread_context.h                                                -*- C++ -*-
   Jeremy Barnes, 26 February 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2009 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Context for a thread.  Allows us to keep track of execution resources, etc
   as we go.
*/

#ifndef __boosting__thread_context_h__
#define __boosting__thread_context_h__

#include "jml/utils/worker_task.h"
#include <boost/random/mersenne_twister.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include "jml/arch/exception.h"

namespace ML {

template<class RNG>
struct RNG_Adaptor {

    RNG_Adaptor(RNG & rng)
        : rng(rng)
    {
    }

    RNG & rng;
    
    template<class T>
    T operator () (T val) const
    {
        return rng() % val;
    }
};

class Thread_Context {
public:
    Thread_Context(Worker_Task & worker
                       = Worker_Task::instance(num_threads() - 1),
                   int group = -1,
                   uint32_t rand_seed = 0,
                   int recursion = 0)
        : worker_(make_unowned_sp(worker)), group_(group),
          uniform01_(rng_), recursion_(recursion)
    {
        if (rand_seed != 0)
            rng_.seed(rand_seed);
    }

    Thread_Context(std::shared_ptr<Worker_Task> worker,
                   int group = -1,
                   uint32_t rand_seed = 0,
                   int recursion = 0)
        : worker_(worker), group_(group), uniform01_(rng_),
          recursion_(recursion)
    {
        if (rand_seed != 0)
            rng_.seed(rand_seed);
    }

    /** Return the worker task that should be used in order to perform any
        sub-tasks */
    Worker_Task & worker() const
    {
        if (!worker_)
            throw Exception("worker_ not initialized");
        return *worker_;
    }

    void seed(uint32_t value)
    {
        if (value == 0) value = 1;
        rng_.seed(value);
        uniform01_.base().seed(value);
    }

    /** Return the group that any parent groups should be under */
    int group() const
    {
        return group_;
    }

    /** Get a random number in a deterministic way */
    uint32_t random()
    {
        return rng_();
    }

    /** Get a uniform (0, 1) random number in a deterministic way */
    float random01()
    {
        return uniform01_();
    }

    typedef RNG_Adaptor<boost::mt19937> RNG_Type;
    RNG_Type rng() { return RNG_Type(rng_); }

    /** What level are we recursed to? */
    int recursion() const { return recursion_; }

    /** Create a new thread context for another thread, optionally with a
        child identifier. */
    Thread_Context child(int new_group = -1, bool local_thread_only = false)
    {
        std::shared_ptr<Worker_Task> new_worker = worker_;
        if (local_thread_only)
            new_worker.reset(new Worker_Task(0));
        return Thread_Context(new_worker, new_group == -1 ? group_ : new_group,
                              random(), recursion_ + 1);
    }

private:
    std::shared_ptr<Worker_Task> worker_;
    int group_;
    boost::mt19937 rng_;
    boost::uniform_01<boost::mt19937> uniform01_;
    int recursion_;
};

} // namespace ML

#endif /* __boosting__thread_context_h__ */
