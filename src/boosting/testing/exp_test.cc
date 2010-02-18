/* exp_test.cc
   Jeremy Barnes, 29 November 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Test to find the maximum value we can take an exp of.
*/

#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "jml/arch/demangle.h"


using namespace std;


template<class Float>
void test()
{
    Float minv = std::numeric_limits<Float>::min();
    Float maxv = std::numeric_limits<Float>::max();
    Float v = (minv + maxv) * (Float)0.5;

    while (minv != maxv) {
        Float e = std::exp(v);
        if (isfinite(e)) minv = v;
        else maxv = v;

        Float oldv = v;
        v = (minv + maxv) * (Float)0.5;

        if (v== oldv) break;

    }

    cout << "maximum exp arg for " << demangle(typeid(Float).name()) << " is "
         << std::setprecision(std::numeric_limits<Float>::digits10 + 2)
         << v << endl;
}

int main(int argc, char ** argv)
{
    test<float>();
    test<double>();
    test<long double>();
}
