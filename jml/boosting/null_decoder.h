/* null_decoder.h                                                  -*- C++ -*-
   Jeremy Barnes, 6 July 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   A decoder that does nothing to its input.
*/

#ifndef __boosting__null_decoder_h__
#define __boosting__null_decoder_h__

#include "decoder.h"


namespace ML {


/*****************************************************************************/
/* NULL_DECODER                                                              */
/*****************************************************************************/

/** The null decoder. */

class Null_Decoder : public Decoder_Impl {
public:
    Null_Decoder();
    Null_Decoder(DB::Store_Reader & store);
    virtual ~Null_Decoder();

    virtual distribution<float>
    apply(const distribution<float> & input) const;

    virtual std::string class_id() const;
    virtual Null_Decoder * make_copy() const;
    
    virtual size_t domain() const;
    virtual size_t range() const;

    virtual Output_Encoding output_encoding(Output_Encoding input) const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
};


} // namespace ML



#endif /* __boosting__null_decoder_h__ */
