/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Set class.
*/

%module jml 
%{
#include "jml/boosting/feature.h"
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

        std::string __repr__() const
        {
            return $self->print();
        }

        size_t __hash__() const
        {
            return $self->hash();
        }

        // Export these as variables, ignoring the getter/setter functions
        // (these are implemented below).
        id_type type;
        id_type arg1;
        id_type arg2;
    }
};


/** This is the feature used when we want one we are sure will never be used
    as a real feature code. */
extern const Feature MISSING_FEATURE;


} // namespace ML

// Implementation of accessor functions for type, arg1 and arg2
%{
    void ML_Feature_type_set(ML::Feature * f, ML::Feature::id_type val) { f->set_type(val); }
    void ML_Feature_arg1_set(ML::Feature * f, ML::Feature::id_type val) { f->set_arg1(val); }
    void ML_Feature_arg2_set(ML::Feature * f, ML::Feature::id_type val) { f->set_arg2(val); }
    
    ML::Feature::id_type ML_Feature_type_get(ML::Feature * f) { return f->type(); }
    
    ML::Feature::id_type ML_Feature_arg1_get(ML::Feature * f) { return f->arg1(); }
    ML::Feature::id_type ML_Feature_arg2_get(ML::Feature * f) { return f->arg2(); }
%}

