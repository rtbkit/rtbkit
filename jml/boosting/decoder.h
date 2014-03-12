/* decoder.h                                                       -*- C++ -*-
   Jeremy Barnes, 21 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   This is the base class for a decoder.  A decoder is used to turn the output
   of a particular classifier into another format (which may have a different
   number of outputs, etc).
*/

#ifndef __boosting__decoder_h__
#define __boosting__decoder_h__

#include "config.h"
#include "jml/stats/distribution.h"
#include <string>
#include "jml/db/persistent.h"
#include "classifier.h"


namespace ML {

class Classifier;
class Classifier_Impl;
class Training_Data;


/*****************************************************************************/
/* DECODER_IMPL                                                              */
/*****************************************************************************/

/** The implementation of the decoder class. */

class Decoder_Impl {
public:
    virtual ~Decoder_Impl();

    virtual distribution<float>
    apply(const distribution<float> & input) const = 0;

    virtual std::string class_id() const = 0;
    virtual Decoder_Impl * make_copy() const = 0;
    
    virtual size_t domain() const = 0;
    virtual size_t range() const = 0;

    virtual Output_Encoding output_encoding(Output_Encoding input) const = 0;

protected:
    virtual void serialize(DB::Store_Writer & store) const = 0;
    virtual void reconstitute(DB::Store_Reader & store) = 0;
    template<class Base, class Derived> friend class Object_Factory;
    template<class Base> friend class Registry;
};

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const std::shared_ptr<const Decoder_Impl> & prob_ptr);

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             std::shared_ptr<Decoder_Impl> & prob_ptr);


/*****************************************************************************/
/* DECODER                                                                   */
/*****************************************************************************/

/** The bridge class for the decoder.  Delegates all of its methods to the
    implementation class. */

class Decoder {
public:
    Decoder();
    Decoder(DB::Store_Reader & store);
    explicit Decoder(const Decoder_Impl & impl);
    Decoder(const std::shared_ptr<Decoder_Impl> & impl);
    Decoder(const Decoder & other);
    Decoder & operator = (const Decoder & other);

    void swap(Decoder & other)
    {
        impl_.swap(other.impl_);
    }

    operator bool () const { return !!impl_; }

    distribution<float> apply(const distribution<float> & input) const
    {
        return impl().apply(input);
    }

    size_t domain() const
    {
        return impl().domain();
    }

    size_t range() const
    {
        return impl().range();
    }

    Output_Encoding output_encoding(Output_Encoding input) const
    {
        return impl().output_encoding(input);
    }

    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);

    Decoder_Impl & impl() { return *impl_; }
    const Decoder_Impl & impl() const { return *impl_; }

private:
    std::shared_ptr<Decoder_Impl> impl_;
};

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Decoder & decoder);

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Decoder & decoder);


} // namespace ML



#endif /* __boosting__decoder_h__ */
