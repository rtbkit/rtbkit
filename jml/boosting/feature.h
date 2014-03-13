/* feature.h                                                       -*- C++ -*-
   Jeremy Barnes, 29 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Definition of a feature.
*/

#ifndef __boosting__feature_h__
#define __boosting__feature_h__

#include <ostream>
#include "jml/compiler/compiler.h"

namespace ML {

/*****************************************************************************/
/* FEATURE                                                                   */
/*****************************************************************************/

/** This structure identifies a single feature.  This is a simple structure
    with a static type, into which the definition for a "feature" is encoded
    by each feature space.

    It maintains three fields: one for the type, and two for the
    arguments.  This should be enough for bigrams, etc, without making
    it too heavy.  Note that these fields <i>mean</i> nothing at all to the
    machine learning algorithms.  The names (type, arg1, arg2) are simply
    meant to be mnemonic, and you can use each of them as you see fit.

    Three of these plus a float value (label) fit nicely into 16 bytes.

    Note that different features with the same value will be interpreted
    differently by different feature sets.  Thus, its meaning depends upon
    its feature set.
*/

typedef int Feature_Id;

struct Feature {
    typedef Feature_Id id_type;
    
    Feature()
    {
        set_type(0);
        set_arg1(0);
        set_arg2(0);
    }
    
    explicit Feature(id_type type, id_type arg1 = 0, id_type arg2 = 0)
    {
        args_[0] = type;
        args_[1] = arg1;
        args_[2] = arg2;
    }

    id_type args_[3];

    const id_type & type() const { return args_[0]; }
    void set_type(id_type new_type) { args_[0] = new_type; }

    const id_type & arg1() const { return args_[1]; }
    void set_arg1(id_type new_arg1) { args_[1] = new_arg1; }

    const id_type&  arg2() const { return args_[2]; }
    void set_arg2(id_type new_arg2) { args_[2] = new_arg2; }

    JML_ALWAYS_INLINE size_t hash() const
    {
        id_type id1 = type(), id2 = arg1(), id3 = arg2();
        size_t result
            = (3 * id1 + 31 * id2 + 71 * id3)
            * (103 * id1 + 1411 * id2 + 3179 * id3);
        return result;
    }

    JML_ALWAYS_INLINE bool operator == (const Feature & other) const
    {
        return args_[0] == other.args_[0] && args_[1] == other.args_[1]
            && args_[2] == other.args_[2];
    }

    JML_ALWAYS_INLINE bool operator != (const Feature & other) const
    {
        return ! operator == (other);
    }

    JML_ALWAYS_INLINE bool operator < (const Feature & other) const
    {
        return (args_[0] < other.args_[0]
                || (args_[0] == other.args_[0]
                    && (args_[1] < other.args_[1]
                        || (args_[1] == other.args_[1]
                            && (args_[2] < other.args_[2])))));
    }

    std::string print() const;
};


/** This is the feature used when we want one we are sure will never be used
    as a real feature code. */
extern const Feature MISSING_FEATURE;

std::ostream & operator << (std::ostream & stream, const Feature & feature);

} // namespace ML


#endif /* __boosting__feature_h__ */
