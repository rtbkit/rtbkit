/* orthogonal_test.cc
   Jeremy Barnes, 22 March 2006
   Copyright (c) Jeremy Barnes, 2006.  All rights reserved.
   $Source$

   Test of the orthogonal array.
*/

#include <iostream>
#include "jml/boosting/config_impl.h"
#include <boost/multi_array.hpp>
#include "jml/stats/distribution.h"

using namespace std;


typedef boost::multi_array<unsigned char, 2> Array;

/** Calculate the Hamming distance between two points. */

int distance(const Array & array, int n1, int n2)
{
    int result = 0;
    int l = array.shape()[1];

    for (unsigned i = 0;  i < l;  ++i)
        result += (array[n1][i] != array[n2][i]);

    return result;
}

void test(const Array & array, const distribution<float> & weights)
{
    int n = array.shape()[0];
    int l = array.shape()[1];
    
    int min_dist = 1000;

    for (unsigned n1 = 0;  n1 < n;  ++n1) {
        for (unsigned n2 = 0;  n2 < n;  ++n2) {
            int dist = distance(array, n1, n2);
            if (dist < min_dist) min_dist = dist;
            
        }
    }
}

/** Algorithm to generate a maximally distant encoding to expand a set
    of values into a higher coded representation.

    We start with a part of the best hamming code.
*/


int main(int argc, char ** argv)
{
    return 0;
}
