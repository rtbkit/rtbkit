/** lapack_detect.cc
    Jeremy Barnes, 11 June 2010
    Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

    Code to detect the LAPACK version.
*/

#include "jml/arch/exception.h"

extern "C" {

    /* Dummy (weak) implementation */
    static void slarfp_dummy(const int * n, float * alpha, float * X, const int * incx,
                      float * tau)
    {
        if (*n != -234)
            throw ML::Exception("wrong version of slarfp called; this weak "
                                "version should have been overridden");
        
        *tau = 1.0;
    }

    /* Elementary reflector.  Used to detect version 3.2 of the LAPACK.  Most
       important thing is that if n < 0, it will return zero in tau. */
    void slarfp_(const int * n, float * alpha, float * X, const int * incx,
                 float * tau) __attribute__ ((__weak__, __alias__ ("slarfp_dummy")));

} // extern "C"

