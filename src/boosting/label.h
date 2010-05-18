/* label.h                                                         -*- C++ -*-
   Jeremy Barnes, 18 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Label class.
*/

#ifndef __jml__boosting__label_h__
#define __jml__boosting__label_h__

#include "jml/compiler/compiler.h"
#include "jml/db/persistent_fwd.h"
#include "jml/arch/exception.h"

namespace ML {

/*****************************************************************************/
/* LABEL                                                                     */
/*****************************************************************************/

/** A label.  Designed to work for both classification and regression
    problems.  This can be:
      - A single integer, giving a single class number;
      - A single float, giving a regression value;
      - A set of up to 30 labels;
      - (later) a pointer to a set structure (for >30 labels in a set).
*/
struct Label {
    Label() { label_ = 0; }
    Label(int lab) { label_ = lab; }
    Label(unsigned lab) { label_ = lab; }
    Label(float val) { val_ = val; }
    Label(double val) { val_ = val; }
    
    operator int () const { return label_; }
    operator int & () { return label_; }
    
    union {
        int label_;
        float val_;
    };
    float value() const { return val_; }
    float & value() { return val_; }
    
    int label() const { return label_; }
    int & label() { return label_; }
    
#ifndef JML_COMPILER_NVCC
    bool is(int lab) const
    {
        if (label_ & 0x80000000)
            return label_ & ((1 << lab) & 0x3fffffff);
        else if (label_ & 0x40000000)
            throw Exception("Training_Data::Label::is_label(): "
                            "label sets not done yet");
        else return (label_ == lab);
    }
    
    /** Serialize to a store. */
    void serialize(DB::Store_Writer & store) const;
    
    /** Reconstitute from a store. */
    void reconstitute(DB::Store_Reader & store);
#endif // JML_COMPILER_NVCC
};



} // namespace ML

#endif /* __jml__boosting__label_h__ */
