/* gpgpu.cc                                                         -*- C++ -*-
   Jeremy Barnes, 10 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Delayed initialization of GPGPU runtimes.
*/

#ifndef __arch__gpgpu_h__
#define __arch__gpgpu_h__

#include <dlfcn.h>
#include <iostream>
#include "jml/utils/environment.h"

using namespace std;

namespace ML {

Env_Option<bool> use_cuda("USE_CUDA", false);
Env_Option<bool> use_cal("USE_CAL", false);

struct Load_CUDA {
    void * handle;

    Load_CUDA()
    {
        handle = 0;
        if (!use_cuda) return;

        handle = dlopen("libarch_cuda.so", RTLD_NOW);
        if (!handle) {
            cerr << "CUDA not initialized: " << dlerror() << endl;
        }
    }

    ~Load_CUDA()
    {
        if (!handle) return;
        
        int res = dlclose(handle);
        if (res != 0)
            cerr << "Unloading CUDA: " << res << endl;
    }

} load_cuda;

struct Load_CAL {
    void * handle;

    Load_CAL()
    {
        handle = 0;
        if (!use_cal) return;

        handle = dlopen("libarch_cal.so", RTLD_NOW);
        if (!handle) {
            cerr << "CAL not initialized: " << dlerror() << endl;
        }
    }

    ~Load_CAL()
    {
        if (!handle) return;
        
        int res = dlclose(handle);
        if (res != 0)
            cerr << "Unloading CAL: " << res << endl;
    }

} load_cal;

} // namespace ML

#endif /* __arch__cuda_h__ */
