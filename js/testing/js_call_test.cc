/* js_call_test.cc
   Jeremy Barnes, 19 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/js/js_call.h"

using namespace Datacratic::JS;
using namespace Datacratic;

void test_that_it_compiles()
{
    JS::JSArgs & args = *(JS::JSArgs *)0;
    
    typedef boost::function<void ()> Fn1;

    Fn1 fn1;
    callfromjs<Fn1>::call(fn1, args);

    typedef boost::function<int ()> Fn2;

    Fn2 fn2;
    int i JML_UNUSED = callfromjs<Fn2>::call(fn2, args);

    typedef boost::function<int (int)> Fn3;

    Fn3 fn3;
    i = callfromjs<Fn3>::call(fn3, args);

    typedef boost::function<int (int, std::string, int, int, std::string)> Fn4;

    Fn4 fn4;
    i = callfromjs<Fn4>::call(fn4, args);
}

int main()
{
}
