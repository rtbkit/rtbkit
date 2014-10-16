/* decoder.cc
   Jeremy Barnes, 22 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of decoder classes.
*/

#include "decoder.h"
#include "registry.h"
#include "classifier.h"



namespace ML {


/*****************************************************************************/
/* DECODER_IMPL                                                              */
/*****************************************************************************/

Decoder_Impl::~Decoder_Impl()
{
}


/*****************************************************************************/
/* DECODER                                                                   */
/*****************************************************************************/


Decoder::Decoder()
{
}

Decoder::Decoder(DB::Store_Reader & store)
{
    reconstitute(store);
}

Decoder::Decoder(const Decoder_Impl & impl)
    : impl_(impl.make_copy())
{
}

Decoder::Decoder(const std::shared_ptr<Decoder_Impl> & impl)
    : impl_(impl)
{
}

Decoder::Decoder(const Decoder & other)
{
    if (other.impl_) impl_.reset(other.impl_->make_copy());
}

Decoder & Decoder::operator = (const Decoder & other)
{
    Decoder new_me(other);
    swap(new_me);
    return *this;
}

void Decoder::serialize(DB::Store_Writer & store) const
{
    Registry<Decoder_Impl>::singleton().serialize(store, impl_.get());
}

void Decoder::reconstitute(DB::Store_Reader & store)
{
    impl_ = Registry<Decoder_Impl>::singleton().reconstitute(store);
}

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Decoder & decoder)
{
    decoder.serialize(store);
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Decoder & decoder)
{
    decoder.reconstitute(store);
    return store;
}


} // namespace ML

