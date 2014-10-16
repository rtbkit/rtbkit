/* rng.h                                                           -*- C++ -*-
   Jeremy Barnes, 12 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   RNG wrapper.
*/

#ifndef __jml__utils__rng_h__
#define __jml__utils__rng_h__

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/thread/tss.hpp>

namespace ML {

/*****************************************************************************/
/* RNG                                                                       */
/*****************************************************************************/

struct RNG {
    
    RNG()
        : uniform01_(rng_)
    {
        seed(random());
    }

    RNG(uint32_t seedValue)
        : uniform01_(rng_)
    {
        seed(seedValue);
    }

    void seed(uint32_t value)
    {
        if (value == 0) value = 1;
        rng_.seed(value);
        uniform01_.base().seed(value);
    }

    /** Get a random number in a deterministic way */
    uint32_t random()
    {
        return rng_();
    }

    /** Get a random number between 0 and max-1 */
    uint32_t random(uint32_t max)
    {
        return rng_() % max;
    }
    

    /** Get a uniform (0, 1) random number in a deterministic way */
    float random01()
    {
        return uniform01_();
    }

    struct StandardRng {

        StandardRng(RNG & rng)
            : rng(rng)
        {
        }
        
        RNG & rng;
        
        template<class T>
        T operator () (T val) const
        {
            return rng.random(val);
        }
    };

    // Return one that can be used in standard algorithms like random_shuffle
    StandardRng standard() { return StandardRng(*this); }

    static RNG & defaultRNG();

private:
    boost::mt19937 rng_;
    boost::uniform_01<boost::mt19937> uniform01_;

    static boost::thread_specific_ptr<RNG> defaultRNGs;
};



} // namespace ML


#endif /* __jml__utils__rng_h__ */
