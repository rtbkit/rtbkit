/* device_data.h                                                   -*- C++ -*-
   Jeremy Barnes, 1 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Structure to deal with host/device transfers.
*/

#ifndef __cuda__device_data_h__
#define __cuda__device_data_h__

#include "jml/compiler/compiler.h"
#include "jml/arch/demangle.h"
#include <typeinfo>

#if (! defined(JML_COMPILER_NVCC) ) || (! JML_COMPILER_NVCC)
# warning "This file should only be included for CUDA"
#endif

#include <iostream>

namespace ML {
namespace CUDA {

using namespace std;

template<typename D>
struct DeviceData {
    DeviceData()
        : size_(0), devicePtr_(0)
    {
    }

    DeviceData(const D * hostData, size_t size)
        : size_(0), devicePtr_(0)
    {
        init(hostData, size);
    }

    ~DeviceData()
    {
        free();
    }

    void free()
    {
        if (!devicePtr_) return;

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            cerr << "error before calling cudaFree for " << num_bytes()
                 << " bytes at "
                 << devicePtr_ << ": " << cudaGetErrorString(err);

        err = cudaFree(devicePtr_);

        // We don't throw, as this normally occurs within destructors
        if (err != cudaSuccess)
            cerr << "error calling cudaFree for " << num_bytes()
                 << " bytes at "
                 << devicePtr_ << ": " << cudaGetErrorString(err);
        
        devicePtr_ = 0;
    }

    void init(const D * hostData, size_t size)
    {
        free();

        size_ = size;

        if (hostData) {
            cudaError_t err
                = cudaMalloc((void **) &devicePtr_, size * sizeof(D));
            
            if (err != cudaSuccess)
                throw Exception(cudaGetErrorString(err));

            err = cudaMemcpy(devicePtr_, hostData, size * sizeof(D),
                             cudaMemcpyHostToDevice);

            if (err != cudaSuccess) {
                cerr << "failed to copy " << size * sizeof(D)
                     << " bytes from host "
                     << hostData << " to " << devicePtr_ << " type "
                     << demangle(typeid(D).name()) << endl;
                throw Exception(cudaGetErrorString(err));
            }
            
#if 0
            cerr << "copied " << size * sizeof(D) << " bytes from host "
                 << hostData << " to " << devicePtr_ << " type "
                 << demangle(typeid(D).name()) << endl;
#endif
        }
    }

    void init_zeroed(size_t size)
    {
        free();
        size_ = size;

        cudaError_t err
            = cudaMalloc((void **) &devicePtr_, size * sizeof(D));
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
        
        err = cudaMemset(devicePtr_, 0, size * sizeof(D));

        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
    }

    void init(size_t size)
    {
        free();
        size_ = size;

        cudaError_t err
            = cudaMalloc((void **) &devicePtr_, size * sizeof(D));
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
    }


    operator D * () const
    {
        return devicePtr_;
    }

    D * getDevice() const
    {
        return devicePtr_;
    }

    void sync(D * hostData) const
    {
        if (!devicePtr_)
            throw Exception("couldn't sync device data");

#if 0
        cerr << "syncing back " << size_ * sizeof(D) << " bytes"
             << " from " << devicePtr_ << " to " << hostData
             << endl;
#endif

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            cerr << "error before calling cudaMemcpy(): "
                 << cudaGetErrorString(err)
                 << endl;;
        
        
        err = cudaMemcpy(hostData, devicePtr_, size_ * sizeof(D),
                         cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess)
            throw Exception(cudaGetErrorString(err));
    }

    size_t num_bytes() const { return sizeof(D) * size_; }

    D * devicePtr_;
    size_t size_;
};

} // namespace CUDA
} // namespace ML


#endif /* __cuda__device_data_h__ */
