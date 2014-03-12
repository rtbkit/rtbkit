/* simd_sqrt_test.cc
   Jeremy Barnes, 30 August 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Test of square root functionality and accuracy using SIMD instructions.
*/

#include "jml/boosting/config_impl.h"
#include "jml/arch/sse.h"
#include <boost/timer.hpp>
#include <vector>
#include <cmath>
#include "jml/arch/tick_counter.h"


using namespace std;



void vec_sqrt_simd(const float * x, float * r, int n)
{
    using namespace Arch::SSE;

    pre_prefetch(x);
    pre_prefetch_write(r);

    v4sf hhalf = load_const(0.5);
    v4sf ttwo  = load_const(2.0);
    v4sf nn, aa, rr, xx, yy;

    while (n >= 4) {
        /* Load our x and y arrays. */
        nn      = loadups(x);
        loop_prefetch(x);
        rr      = VEC_INSN(rsqrtps, (nn));

#if 1
        aa   = VEC_INSN(rcpps,   (rr));  // approx
        xx   = mulps(rr, aa);
        yy   = subps(ttwo, xx);
        aa   = mulps(aa, yy);
#else
        aa      = accurate_recip(rr);
#endif

        rr      = mulps(rr, nn);
        rr      = addps(aa, rr);
        aa      = mulps(hhalf, rr);
        storeups(r, aa);
        loop_prefetch_write(r);
        x += 4;
        r += 4;
        n -= 4;
    }
}

void vec_sqrt_simd2(const float * x, float * r, int n)
{
    using namespace Arch::SSE;

    //pre_prefetch(x);
    //pre_prefetch_write(r);

    v4sf hhalf = load_const(0.5);
    v4sf ttwo  = load_const(2.0);
    v4sf nn0, nn1, aa0, aa1, rr0, rr1, xx0, xx1, yy0, yy1;

    while (n >= 8) {
        /* Load our x and y arrays. */
        nn0     = loadaps(x);
        loop_prefetch(x);
        nn1     = loadaps(x + 4);
        rr0     = VEC_INSN(rsqrtps, (nn0));
        rr1     = VEC_INSN(rsqrtps, (nn1));
        aa0     = VEC_INSN(rcpps,   (rr0));  // approx
        aa1     = VEC_INSN(rcpps,   (rr1));  // approx
        xx0     = mulps(rr0, aa0);
        yy0     = subps(ttwo, xx0);
        xx1     = mulps(rr1, aa1);
        x += 8;
        aa0     = mulps(aa0, yy0);
        yy1     = subps(ttwo, xx1);
        rr0     = mulps(rr0, nn0);
        rr0     = addps(aa0, rr0);
        aa1     = mulps(aa1, yy1);
        n -= 8;
        aa0     = mulps(hhalf, rr0);
        rr1     = mulps(rr1, nn1);
        storeaps(r, aa0);
        rr1     = addps(aa1, rr1);
        loop_prefetch_write(r);
        aa1     = mulps(hhalf, rr1);
        storeaps(r + 4, aa1);
        r += 8;
    }
}

void vec_sqrt_simd3(const float * x, float * r, int n)
{
    using namespace Arch::SSE;

    //pre_prefetch(x);
    //pre_prefetch_write(r);

    v4sf nn0, nn1, rr0, rr1;
    
    while (n >= 8) {
        /* Load our x and y arrays. */
        nn0     = loadaps(x);
        loop_prefetch(x);
        nn1     = loadaps(x + 4);
        rr0     = VEC_INSN(sqrtps, (nn0));
        rr1     = VEC_INSN(sqrtps, (nn1));
        x += 8;
        n -= 8;
        storeaps(r, rr0);
        loop_prefetch_write(r);
        storeaps(r + 4, rr1);
        r += 8;
    }
}

void vec_sqrt_scalar(const float * p, float * r, int n)
{
    for (unsigned i = 0;  i < n;  ++i)
        *r++ = std::sqrt(*p++);
}

void test1()
{
    size_t SIZE = 1000;
    size_t TRIALS = 100000;
    vector<float> numbers(SIZE);
    for (unsigned i = 0;  i < SIZE;  ++i)
        numbers[i] = rand();

    // test scalar version
    {
        vector<float> x = numbers;

        double before = ticks();

        for (unsigned i = 0;  i < TRIALS;  ++i)
            vec_sqrt_scalar(&x[0], &x[0], SIZE);

        double after = ticks() - ticks_overhead;

        cerr << "scalar: " << (after - before) / (TRIALS * SIZE)
             << " ticks/call" << endl;
    }

    // test vector version
    {
        vector<float> x = numbers;

        double before = ticks();

        for (unsigned i = 0;  i < TRIALS;  ++i)
            vec_sqrt_simd(&x[0], &x[0], SIZE);

        double after = ticks() - ticks_overhead;

        cerr << "vector: " << (after - before) / (TRIALS * SIZE)
             << " ticks/call" << endl;
    }

    // test vector version
    {
        vector<float> x = numbers;

        double before = ticks();

        for (unsigned i = 0;  i < TRIALS;  ++i)
            vec_sqrt_simd2(&x[0], &x[0], SIZE);

        double after = ticks() - ticks_overhead;

        cerr << "vector2: " << (after - before) / (TRIALS * SIZE)
             << " ticks/call" << endl;
    }

    // test vector version
    {
        vector<float> x = numbers;

        double before = ticks();

        for (unsigned i = 0;  i < TRIALS;  ++i)
            vec_sqrt_simd3(&x[0], &x[0], SIZE);

        double after = ticks() - ticks_overhead;

        cerr << "vector3: " << (after - before) / (TRIALS * SIZE)
             << " ticks/call" << endl;
    }
}

int main(int argc, char ** argv)
{
    using namespace Arch::SSE;

#if 0
    double num = 2.09384923048;

    double correct = std::sqrt(num);

    int iter = 6;

    float approx = num / 2;

    for (unsigned i = 0;  i < iter;  ++i) {
        float z = approx * approx - num;
        float new_approx = 0.5 * (approx + num / approx);
        cerr << "iter " << i << " approx = " << approx
             << " new_approx = " << new_approx << " z = " << z << endl;
        approx = new_approx;
        cerr << "after " << i + 1 << " iterations, error = "
             << correct - approx << endl;
    }

    v4sf nn    = load_const(num);
    v4sf hhalf = load_const(0.5);
    v4sf cc    = load_const(correct);
    //v4sf aa    = mulps(hhalf, nn);
    v4sf rr    = VEC_INSN(rsqrtps, (nn));
    v4sf aa    = accurate_recip(rr);//VEC_INSN(rcpps, (rr));
    //v4sf aa    = VEC_INSN(rcpps, (rr));
    //v4sf aa    = VEC_INSN(rcpps,   (VEC_INSN(rsqrtps, (nn))));  // approx


    for (unsigned i = 0;  i < iter;  ++i) {
        cerr << "iter " << i << " approx " << aa 
             << " error " << subps(cc, aa) << endl;
        
        //rr      = accurate_recip(aa);
        rr      = mulps(rr, nn);
        rr      = addps(aa, rr);
        aa      = mulps(hhalf, rr);
        //rr      = VEC_INSN(rcpps, (aa));
        rr      = accurate_recip(aa);//VEC_INSN(rcpps, (aa));

        //v4sf ee = subps(cc, aa);
        //cerr << "iter " << i << " error " << ee << endl;
    }
#endif

    test1();
}
