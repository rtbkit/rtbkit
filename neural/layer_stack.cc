/* layer_stack.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of layers.
*/

#include "layer_stack.h"
#include "dense_layer.h"
#include "layer_stack_impl.h"
#include "jml/boosting/registry.h"

namespace ML {

template class Layer_Stack<Layer>;
template class Layer_Stack<Dense_Layer<float> >;
template class Layer_Stack<Dense_Layer<double> >;

namespace {

Register_Factory<Layer, Layer_Stack<Layer> >
LAYER_STACK_REGISTER("Layer_Stack");

} // file scope


#if 0

void
DNAE_Stack::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)1; // version
    store << compact_size_t(size());
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].serialize(store);
}

void
DNAE_Stack::
reconstitute(ML::DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1) {
        cerr << "version = " << (int)version << endl;
        throw Exception("DNAE_Stack::reconstitute(): invalid version");
    }
    compact_size_t sz(store);
    resize(sz);

    for (unsigned i = 0;  i < sz;  ++i)
        (*this)[i].reconstitute(store);
}

#endif

} // namespace ML

