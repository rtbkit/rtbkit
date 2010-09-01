/* single_sm_matrix_vector_multiply.h                              -*- C++ -*-
   Jeremy Barnes, 29 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Matrix-vector multiplication routine on a SINGLE Stream Multiprocessor.
*/

#ifndef __cuda__single_sm_matrix_vector_multiply_h__
#define __cuda__single_sm_matrix_vector_multiply_h__

#if 0
/** Given a matrix[ni][no] with stride stride, multiply by the vector[ni] and
    overwrite the vector[no] with the output.  The size of the vector should
    be at least max(no, ni).  Splits the work over the given number of
    threads.  Works best if the vector is in shared memory; global memory will
    be slow.

    It always goes through the matrix in such a way that the unit stride is
    respected.
*/
__device__ void
matrix_vector_multiply_left(const float * matrix,
                            const float * vector,
                            int ni, int no, int stride,
                            int num_threads_on_multiprocessor)
{
    /* We want to have each thread working here, even if no is much less
       than the number of threads.  To do so, we assign each thread to
       a certain o value and a certain subset of the i values, and then
       accumulate the updates, broadcasting them at the end.
       
       For example:
       32 threads
       2 outputs
       
       So we have 16 threads working on each example
       
       100 threads
       16 outputs
       
       So we have 7 threads on the first 4 examples, and 6 threads on
       the rest.
    
       32 threads
       100 outputs

       So most threads do 3 examples but the first 4 do 4.
*/
    
    int nt = num_threads_on_multiprocessor;
    
    for (unsigned o_start = 0;  o_start < no;  o_start += nt) {

        int nob = std::min(nt, no - o_start);  // no for the block

        int min_threads = nt / no;  // min num threads for the block
        int left_over   = nt % no;  // left over to help with extra work
        int max_threads = min_threads + (left_over > 0);
        
        int o = tid % no;    // which o value are we working on?
        int idx = tid / no;  // which thread in that block?
        int o_threads = min_threads + (o < left_over);
        
        double accum = 0.0;
        
        for (unsigned i = idx;  i < ni;  i += o_threads) {
            // warning: bank conflicts...
            float inval = vector[i];
            float weight = matrix[i * w_stride + o];
            
            accum += weight * inval;
        }
        
        if (max_threads > 1) {
        
            __syncthreads();
        
            if (tid < no) vector[tid] = biases[l][tid];
            
            __syncthreads();
            
            /* Now we accumulate them, allowing each thread to increment in its
               turn. */
            for (unsigned i = 0;  i < max_threads;  ++i, __syncthreads())
                if (i == idx) vector[o] += accum;
        }
    }
    else {
        /* More outputs than threads.  We break up into blocks. */
        
        __syncthreads();
        
        if (__any(tid < no)) {
            
            if (tid < no)
                this_layer_outputs[tid]
                    = vector[tid]
                    = transform(vector[tid], activation);
        }
    }
    else {
        accum += biases[l][o];
        this_layer_outputs[o]
            = vector[o]
            = transform(accum, activation);
            
    }
}

#endif

#endif /* __cuda__single_sm_matrix_vector_multiply_h__ */
