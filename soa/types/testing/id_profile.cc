/* id_profile.cc
   Jeremy Barnes, 17 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include <iostream>
#include "soa/types/id.h"
#include "soa/types/date.h"

using namespace ML;
using namespace std;
using namespace Datacratic;

int hexToDec(int c)
{
    int d = c & 0x1f;
    int i = (c & 0x60) >> 5;
    d += (i == 1) * -16;
    d += (i >= 2) * 9;
    bool h = isxdigit(c);
    return h * d - (!h);
}

int hexToDec2(int c)
{
    int v;

    if (c >= '0' && c <= '9')
        v = c - '0';
    else if (c >= 'a' && c <= 'f')
        v = c + 10 - 'a';
    else if (c >= 'A' && c <= 'F')
        v = c + 10 - 'A';
    else
        v = -1;

    return v;
}

static const signed char lookups[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};

inline int hexToDec3(int c)
{
    return lookups[c & 0x1f];
}

void profile1()
{
    Date before = Date::now();
    
    int n = 100000000;

    for (unsigned i = 0;  i < n;  ++i) {
        (void)hexToDec(i % 127);
    }

    Date after = Date::now();
    double elapsed = after.secondsSince(before);
    
    cerr << "processed " << n << " in " << elapsed << "s ("
         << 1.0 * n / elapsed << " per second)" << endl;
}

void profile2()
{
    Date before = Date::now();
    
    int n = 100000000;

    for (unsigned i = 0;  i < n;  ++i) {
        (void)hexToDec2(i % 127);
    }

    Date after = Date::now();
    double elapsed = after.secondsSince(before);
    
    cerr << "processed " << n << " in " << elapsed << "s ("
         << 1.0 * n / elapsed << " per second)" << endl;
}

void profile3()
{
    Date before = Date::now();
    
    int n = 100000000;

    for (unsigned i = 0;  i < n;  ++i) {
        (void)hexToDec3(i % 127);
    }

    Date after = Date::now();
    double elapsed = after.secondsSince(before);
    
    cerr << "processed " << n << " in " << elapsed << "s ("
         << 1.0 * n / elapsed << " per second)" << endl;
}

int main(int argc, char ** argv)
{
    //profile1();
    //profile2();
    //profile3();

    string ids[5] = {
        "2fa07c3c-1ac1-4001-15e8-42e6000003a1",
        "a78e802f-1ac1-4001-15e8-c6b0000003a0",
        "f8ece33b-1ac1-4001-15e8-42e6000003a1",
        "e46ead3d-1ac1-4001-15e8-ade2000003a1",
        "7081463e-1ac1-4001-15e8-01e8000003a0"
    };

    int nids = 5;

    Date before = Date::now();
    
    int n = 10000000;
    
    for (unsigned i = 0;  i < n;  ++i) {
        Id id(ids[i % nids]);
        //if (id.type != Id::UUID)
        //    cerr << "bad type" << endl;
        //if (id.toString() != ids[i % nids])
        //    cerr << "bad parse" << endl;
    }

    Date after = Date::now();
    double elapsed = after.secondsSince(before);
    
    cerr << "processed " << n << " in " << elapsed << "s ("
         << 1.0 * n / elapsed << " per second)" << endl;
}
