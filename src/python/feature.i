/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Set class.
*/

%module jml 
%{
#include "boosting/feature.h"
%}

%include "std_string.i"

namespace ML {

/*****************************************************************************/
/* FEATURE                                                                   */
/*****************************************************************************/

typedef int Feature_Id;

struct Feature {
    typedef Feature_Id id_type;
    
    Feature();
    
    explicit Feature(id_type type, id_type arg1 = 0, id_type arg2 = 0);

    const id_type & type() const;
    void set_type(id_type new_type);

    const id_type & arg1() const;
    void set_arg1(id_type new_arg1);

    const id_type&  arg2() const;
    void set_arg2(id_type new_arg2);

    size_t hash() const;

    bool operator == (const Feature & other) const;

    bool operator != (const Feature & other) const;

    bool operator < (const Feature & other) const;

    // Add a __str__ method for Python
    %extend {
        std::string __str__() const
        {
            return $self->print();
        }
    }
};


/** This is the feature used when we want one we are sure will never be used
    as a real feature code. */
extern const Feature MISSING_FEATURE;

} // namespace ML
