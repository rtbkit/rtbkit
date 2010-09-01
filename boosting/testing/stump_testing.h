/* stump_testing.h                                                 -*- C++ -*-
   Jeremy Barnes, 22 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Routines to allow testing of the stump code.
*/

#include "jml/arch/exception.h"



namespace ML {


/*****************************************************************************/
/* W_TESTING                                                                 */
/*****************************************************************************/

/** This is a test harness for a W class.  It wraps two W implementations
    and makes sure that the same results are always returned by both of
    them.
*/

template<class W1, class W2>
struct W_testing {
private:
    W1 w1;
    W2 w2;

public:
    W_testing(size_t nl)
        : w1(nl), w2(nl)
    {
        check("construct");
    }
    
    W_testing(const W_testing & other)
        : w1(other.w1), w2(other.w2)
    {
        check("copy construct");
    }
    
    W_testing & operator = (const W_testing & other)
    {
        w1 = other.w1;
        w2 = other.w2;
        check("assign");
        return *this;
    }

    double operator () (int l, int cat, bool corr) const
    {
        double r1 = w1(l, cat, corr);
        double r2 = w2(l, cat, corr);
        compare("operator ()", r1, r2);
        return r1;
    }
    
    /* Structure to assign to both w1 and w2 when something is assigned
       to here. */
    struct Assigner {
        Assigner(W_testing * obj, int l, int cat, bool corr)
            : obj(obj), l(l), cat(cat), corr(corr)
        {
        }

        W_testing * obj;
        int l;
        int cat;
        bool corr;
        
        /* Write. */
        double operator = (double d) const
        {
            obj->w1.operator () (l, cat, corr) = d;
            obj->w2.operator () (l, cat, corr) = d;
            obj->check("assign");
        }

        /* Read. */
        operator double () const { return ((const W_testing *)obj)
                                       ->operator () (l, cat, corr); }
    };
    friend class Assigner;
    
    Assigner operator () (int l, int cat, bool corr)
    {
        return Assigner(this, l, cat, corr);
    }
    
    std::string print() const
    {
        return w1.print();
    }
    
    size_t nl() const { return w1.nl(); }

    template<class Iterator>
    void add(int correct_label, int bucket, Iterator it, int advance)
    {
        w1.add(correct_label, bucket, it, advance);
        w2.add(correct_label, bucket, it, advance);
        check("add");
    }

    template<class Iterator>
    void add(int correct_label, int bucket, double wt, Iterator it, int advance)
    {
        w1.add(correct_label, bucket, wt, it, advance);
        w2.add(correct_label, bucket, wt, it, advance);
        check("add");
    }
    
    template<class Iterator>
    void transfer(int correct_label, int from, int to, float weight,
                  Iterator it, int advance)
    {
        w1.transfer(correct_label, from, to, weight, it, advance);
        w2.transfer(correct_label, from, to, weight, it, advance);
        check("transfer");
    }

    void clip(int bucket)
    {
        w1.clip(bucket);
        w2.clip(bucket);
        check("clip");
    }
    
    void swap_buckets(int b1, int b2)
    {
        w1.swap_buckets(b1, b2);
        w2.swap_buckets(b1, b2);
        check("swap_buckets");
    }

    void compare(const char * routine, double v1, double v2) const
    {
        if (abs(v1 - v2) > 0.0/*1e-5*/)
            throw Exception(format("W_testing: %s: values differ: "
                                   "%20.15f vs %20.15f\n",
                                   routine, v1, v2)
                            + w1.print() + "\n" + w2.print());
    }
    
    void check(const char * routine) const
    {
        for (unsigned l = 0;  l < nl();  ++l) {
            for (unsigned j = 0;  j < 3;  ++j) {
                compare(routine, w1(l, j, false), w2(l, j, false));
                compare(routine, w1(l, j, true),  w2(l, j, true));
            }
        }
    }
};


/*****************************************************************************/
/* Z_TESTING                                                                 */
/*****************************************************************************/

template<class Z1, class Z2>
struct Z_testing {
    static const double worst   = Z1::worst;
    static const double none    = Z1::none;
    static const double perfect = Z1::perfect;

    /* Return the constant missing part. */
    template<class W>
    double missing(const W & w, bool optional) const
    {
        double r1 = z1.missing(w, optional);
        double r2 = z2.missing(w, optional);
        compare("missing", r1, r2);
        return r1;
    }

    /* Return the non-missing part. */
    template<class W>
    double non_missing(const W & w, double missing) const
    {
        double r1 = z1.non_missing(w, missing);
        double r2 = z2.non_missing(w, missing);
        //cerr << "w = " << endl << w.print() << endl;
        //cerr << "missing = " << missing << endl;
        compare("non_missing", r1, r2);
        return r1;
    }

    template<class W>
    double operator () (const W & w) const
    {
        double r1 = z1(w);
        double r2 = z2(w);
        compare("()", r1, r2);
        return r1;
    }

    void compare(const char * routine, double v1, double v2) const
    {
        if (abs(v1 - v2) > 1e-5) {
            throw Exception(format("Z_testing: %s: values differ: "
                                   "%20.15f vs %20.15f\n",
                                   routine, v1, v2));
        }
    }

    template<class W>
    bool can_beat(const W & w, double missing, double z_best) const
    {
        bool r1 = z1.can_beat(w, missing, z_best);
        bool r2 = z2.can_beat(w, missing, z_best);
        compare("can_beat", r1, r2);
        return r1;
    }

    Z1 z1;
    Z2 z2;
};

} // namespace ML

