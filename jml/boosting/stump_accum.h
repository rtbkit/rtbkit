/* stump_accum.h                                                   -*- C++ -*-
   Jeremy Barnes, 28 April 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   The stump_accum object.  Internal to the stump, but split off here to allow
   the testing programs to work.
*/

#include "jml/arch/threads.h"

namespace ML {

namespace {

/* Sort stumps by their Z values. */
struct Sort_Z {
    bool operator () (const Stump & s1, const Stump & s2) const
    {
        return s1.Z > s2.Z;
    }
};

} // file scope

struct No_Locks {
    typedef int Lock;
    typedef int Guard;
};

typedef Lock Lock_;
typedef Guard Guard_;

struct Locked {
    typedef Lock_ Lock;
    typedef Guard_ Guard;
};

/*****************************************************************************/
/* STUMP_ACCUM                                                               */
/*****************************************************************************/

/* An internal class to the stumps.  This is passed to the stump training core
   to accumulate the results into.  It keeps track of which is the best stump.
*/

template<class W, class Z, class C, class Tracer = No_Trace,
         class Locks = No_Locks>
struct Stump_Accum {

    Stump_Accum(const Feature_Space & fs, bool fair, int num_results = 0,
                const C & c = C(), const Tracer & tracer = Tracer())
        : calc_c(c), z_best(Z::worst), z_needed(Z::worst), fair(fair),
          num_results(num_results), fs(fs), tracer(tracer)
    {
        next_check = num_results * 2;
    }

    Z calc_z;
    C calc_c;

    typedef typename Locks::Lock Lock;
    typedef typename Locks::Guard Guard;
    Lock lock;

    /* Holds an entry that we are accumulating. */
    struct Entry {
        Entry(const W & w, float arg, float z, const Feature & feature)
            : w(w), arg(arg), z(z), feature(feature)
        {
            //if (!isfinite(z))
            //    throw Exception("Attempt to put non-finite Z "
            //                    + ostream_format(z) + " into accum");
        }

        W w;
        float arg;
        float z;
        Feature feature;

        bool operator < (const Entry & other) const
        {
            return z < other.z;
        }
    };

    float z_best;
    float z_needed;
    int next_check;

    std::vector<Entry> best;

    bool fair;
    int num_results;

    const Feature_Space & fs;

    Tracer tracer;

    /** Method that gets called when we start a new feature.  We use it to
        pre-cache part of the work from the Z calculation, as we are
        assured that the MISSING buckets of W will never change after this
        method is called.

        Return value is used to allow an early exit from the training process,
        due to it being impossible for this feature to have a high enough
        value to be included.
    */
    bool start(const Feature & feature, const W & w, double & missing)
    {
        bool optional = fs.info(feature).optional();
        missing = calc_z.missing(w, optional);
        bool keep_going = num_results > 1
            || calc_z.can_beat(w, missing, z_best);
        //cerr << "start: missing = " << missing << " z_best = "
        //     << z_best << " keep_going = " << keep_going << endl;
        //cerr << "non-missing = " << calc_z.non_missing(w, missing)
        //     << endl;
        return keep_going;
    }

    void cleanup_best()
    {
        Guard JML_UNUSED guard(lock);
        //cerr << "cleanup_best: best.size() = " << best.size()
        //     << " num_results = " << num_results << endl;
        //cerr << "fair = " << fair << endl;

        //for (unsigned i = 0;  i < best.size();  ++i)
        //    cerr << " " << best[i].z;
        //cerr << endl;

#if 0
        for (unsigned i = 0;  i < best.size();  ++i) {
            if (!finite(best[i].z))
                throw Exception("non-finite Z value");
        }
#endif

        if (fair) std::sort(best.begin(), best.end());
        else {
            size_t num_to_get = std::min<size_t>(num_results, best.size());
            std::partial_sort(best.begin(), best.begin() + num_to_get,
                              best.end());
        }
        //cerr << "finished sort." << endl;
        if (num_results > best.size()) return;  // not enough yet...

        typename std::vector<Entry>::iterator it
            = best.begin() + num_results;
        if (fair) {
            --it;  // get the last included one
            z_needed = it->z;
            while (it != best.end() && z_equal(it->z, z_needed)) ++it;
        }
        
        //cerr << "before erase" << endl;
        best.erase(it, best.end());
        //cerr << "finished erase" << endl;
    }

    /** Method that gets called when we have found a potential split point. */
    float add(const Feature & feature, const W & w, float arg, float z)
    {
        Guard JML_UNUSED guard(lock);

        if (tracer)
            tracer("stump accum", 3)
                << "  accum: feature " << feature << " arg " << arg
                << "  z " << z << "  " << fs.print(feature) << std::endl;
        //if (tracer)
        //    tracer("stump accum", 4)
        //        << "    accum: w = " << endl << w.print() << endl;
        
        if ((fair && z_equal(z, z_best)) || num_results > 1 || best.empty()) {
            /* Check if we can short circuit it.  This happens when the
               z is lower than the lowest necessary value. */
            if (!z_equal(z, z_needed) && z_needed < z) return z;

            // Another split with the same Z value
            best.push_back(Entry(w, arg, z, feature));
            z_best = std::min(z, z_best);

            if (num_results > 1 && best.size() == next_check) {
                /* Time to clean up the list, to stop it getting too long. */
                cleanup_best();
                next_check = std::min(std::max(512, num_results * 2),
                                      next_check * 2);
            }
        }
        else if (z < z_best) {
            // A better one.  This replaces whatever we had accumulated so
            // far.
            z_best = z;
            best[0].w = w;
            best[0].arg = arg;
            best[0].z = z;
            best[0].feature = feature;
            
            if (fair) {
                // Erase the rest of the ones with the old (inferior) Z value
                best.erase(best.begin() + 1, best.end());
            }
        }

        return z;
    }
    
    /** Method that gets called when we have found a potential split point. */
    float add(const Feature & feature, const W & w, float arg, double missing)
    {
        float z = calc_z.non_missing(w, missing);
        return add(feature, w, arg, z);
    }

    /** Method that gets called when we have found a potential split point. */
    float add_presence(const Feature & feature, const W & w, float arg,
                       double missing)
    {
        float z = calc_z.non_missing_presence(w, missing);
        return add(feature, w, arg, z);
    }
    
    void finish(const Feature & feature)
    {
        // nothing to do here, at the moment
    }

    /* Extract a list of stumps from the results. */
    std::vector<Stump>
    results(const Training_Data & data, const Feature & predicted)
    {
        Guard JML_UNUSED guard(lock);

        cleanup_best();

        unsigned nx = data.example_count();
        unsigned nc = data.label_count(predicted);
        float epsilon = 1.0 / (nx * nc);
        
        std::vector<Stump> result;
        for (unsigned i = 0;  i < best.size();  ++i) {
            std::vector<distribution<float> > c
                = calc_c(best[i].w, epsilon,
                         fs.info(best[i].feature).optional());
            Stump added(predicted, best[i].feature,
                        best[i].arg,
                        c[true], c[false], c[MISSING],
                        calc_c.update_alg(),
                        data.feature_space(), best[i].z);
            result.push_back(added);
            //cerr << "added " << added.print() << endl;
        }
        
        return result;
    }
};


/*****************************************************************************/
/* BIAS_ACCUM                                                               */
/*****************************************************************************/

/* Class that accumulates to keep track of the bias. */

template<class W, class Z, class C, class Tracer = No_Trace>
struct Bias_Accum {

    Bias_Accum(const Feature_Space & fs, int nl, const C & c = C(),
               const Tracer & tracer = Tracer())
        : calc_c(c), saved_w(nl), fs(fs), tracer(tracer)
    {
    }
    
    Z calc_z;
    C calc_c;
    
    double z;
    W saved_w;

    const Feature_Space & fs;

    Tracer tracer;

    /** Method that gets called when we start a new feature.  We use it to
        pre-cache part of the work from the Z calculation, as we are
        assured that the MISSING buckets of W will never change after this
        method is called.

        Return value is used to allow an early exit from the training process,
        due to it being impossible for this feature to have a high enough
        value to be included.
    */
    bool start(const Feature & feature, const W & w, double & missing)
    {
        saved_w = w;
        missing = calc_z.missing(w, false);
        bool keep_going = true;
        return keep_going;
    }
    
    /** Method that gets called when we have found a potential split point. */
    float add(const Feature & feature, const W & w, float arg, double missing)
    {
        return z = calc_z.non_missing(w, missing);
    }

    /** Method that gets called when we have found a potential split point. */
    float add_presence(const Feature & feature, const W & w, float arg,
                       double missing)
    {
        return z = calc_z.non_missing_presence(w, missing);
    }

    void finish(const Feature & feature)
    {
        // nothing to do here, at the moment
    }

    /* Extract a list of stumps from the results. */
    Stump result(const Training_Data & data, const Feature & predicted)
    {
        unsigned nx = data.example_count();
        unsigned nc = data.label_count(predicted);
        float epsilon = 1.0 / (nx * nc);
        
        std::vector<distribution<float> > c
            = calc_c(saved_w, epsilon, false);
        //cerr << "c[true] = " << c[true] << endl;
        //cerr << "c[false] = " << c[false] << endl;
        //cerr << "c[MISSING] = " << c[MISSING] << endl;
        //cerr << "bias z = " << z << endl;
        Stump result(predicted, MISSING_FEATURE, -INFINITY, c[true], c[false],
                     c[MISSING], calc_c.update_alg(), data.feature_space(),
                     1.0);
        return result;
    }
};

} // namespace ML


   
