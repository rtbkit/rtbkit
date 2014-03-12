/* null_decoder.h                                                  -*- C++ -*-
   Jeremy Barnes, 6 July 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   A decoder that does nothing to its input.
*/

#include "null_decoder.h"
#include "jml/db/persistent.h"
#include "registry.h"
#include "config_impl.h"


using namespace std;
using namespace DB;


namespace ML {


/*****************************************************************************/
/* NULL_DECODER                                                              */
/*****************************************************************************/

Null_Decoder::Null_Decoder()
{
}

Null_Decoder::Null_Decoder(DB::Store_Reader & store)
{
    reconstitute(store);
}

Null_Decoder::~Null_Decoder()
{
}

distribution<float>
Null_Decoder::apply(const distribution<float> & input) const
{
    return input;
}

std::string Null_Decoder::class_id() const
{
    return "NULL_DECODER";
}

Null_Decoder * Null_Decoder::make_copy() const
{
    return new Null_Decoder(*this);
}
    
size_t Null_Decoder::domain() const
{
    return (size_t)-1;
}

size_t Null_Decoder::range() const
{
    return (size_t)-1;
}

Output_Encoding Null_Decoder::output_encoding(Output_Encoding input) const
{
    return input;
}

void Null_Decoder::serialize(DB::Store_Writer & store) const
{
    store << string("NULL_DECODER") << compact_size_t(1);
}

void Null_Decoder::reconstitute(DB::Store_Reader & store)
{
    string name;
    store >> name;
    if (name != "NULL_DECODER")
        throw Exception("Null_Decoder::reconstitute(): tried to "
                        "reconstitue, got a " + name + " instead.");

    compact_size_t version(store);
    if (version > 1)
        throw Exception("Null_Decoder::reconstitute(): got unknown version");
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Decoder_Impl, Null_Decoder> REGISTER("NULL_DECODER");

} // file scope


} // namespace ML

