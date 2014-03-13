/* cuda.cc
   Jeremy Barnes, 10 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   CUDA implementation.  dlopens and initializes CUDA if it's available on the
   system.
*/

#include "cuda.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include "format.h"
#include "jml/utils/environment.h"

using namespace std;


namespace ML {

namespace {

Env_Option<bool> debug("DEBUG_CUDA_INIT", false);

} // file scope

struct Register_CUDA {
    
    Register_CUDA()
    {
        if (debug)
            cerr << "registering CUDA" << endl;
        int num_devices = 0;
        cudaError_t res = cudaGetDeviceCount(&num_devices);
        if ((int)res != CUDA_SUCCESS) {
            if (debug)
                cerr << "error calling cudaGetDeviceCount(): "
                     << cudaGetErrorString(res) << endl;
            return;
        }

        if (debug)
            cerr << num_devices << " CUDA devices" << endl;

        for (unsigned i = 0; i < num_devices; ++i) {
            cudaDeviceProp deviceProp;
            res = cudaGetDeviceProperties(&deviceProp, i);
            if ((int)res != CUDA_SUCCESS) {
                if (debug)
                    cerr << "error calling cudaGetDeviceProperties for device "
                         << i << ": " << cudaGetErrorString(res) << endl;
                continue;
            }

            if (i == 0
                && deviceProp.major == 9999
                && deviceProp.minor == 9999) {
                if (debug)
                    cerr << "no device really supports CUDA" << endl;
                return;
            }

            if (!debug) continue;

            cerr << format("\nDevice %d: \"%s\"\n", i, deviceProp.name);
            cerr << format("  Major revision number:                         %d\n",
                   deviceProp.major);
            cerr << format("  Minor revision number:                         %d\n",
                   deviceProp.minor);
            cerr << format("  Total amount of global memory:                 %u bytes\n",
                   deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
            cerr << format("  Number of multiprocessors:                     %d\n",
                   deviceProp.multiProcessorCount);
            cerr << format("  Number of cores:                               %d\n",
                   8 * deviceProp.multiProcessorCount);
#endif
            cerr << format("  Total amount of constant memory:               %u bytes\n",
                   deviceProp.totalConstMem); 
            cerr << format("  Total amount of shared memory per block:       %u bytes\n",
                   deviceProp.sharedMemPerBlock);
            cerr << format("  Total number of registers available per block: %d\n",
                   deviceProp.regsPerBlock);
            cerr << format("  Warp size:                                     %d\n",
                   deviceProp.warpSize);
            cerr << format("  Maximum number of threads per block:           %d\n",
                   deviceProp.maxThreadsPerBlock);
            cerr << format("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
                   deviceProp.maxThreadsDim[0],
                   deviceProp.maxThreadsDim[1],
                   deviceProp.maxThreadsDim[2]);
            cerr << format("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
                   deviceProp.maxGridSize[0],
                   deviceProp.maxGridSize[1],
                   deviceProp.maxGridSize[2]);
            cerr << format("  Maximum memory pitch:                          %u bytes\n",
                   deviceProp.memPitch);
            cerr << format("  Texture alignment:                             %u bytes\n",
                   deviceProp.textureAlignment);
            cerr << format("  Clock rate:                                    %.2f GHz\n",
                   deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
            cerr << format("  Concurrent copy and execution:                 %s\n",
                   deviceProp.deviceOverlap ? "Yes" : "No");
#endif
        }
    }
    
    ~Register_CUDA()
    {
        if (debug)
            cerr << "unregistering CUDA" << endl;
    }
} register_cuda;

} // namespace ML
