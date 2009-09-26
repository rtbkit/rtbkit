/* feature_space.i                                                 -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Space class.
*/

%module jml 
%{
#include "boosting/feature_space.h"
%}

%include "std_vector.i"

namespace ML {



/*****************************************************************************/
/* FEATURE_SPACE                                                             */
/*****************************************************************************/

/** This is a class that provides information on a space of features. */

class Feature_Space {
public:
    virtual ~Feature_Space();

    virtual Feature_Info info(const Feature & feature) const = 0;
    virtual std::string print(const Feature & feature) const;
    virtual std::string print(const Feature & feature, float value) const;
    //virtual bool parse(Parse_Context & context, Feature & feature) const;
    //virtual void parse(const std::string & name, Feature & feature) const;
    //virtual void expect(Parse_Context & context, Feature & feature) const;

    //virtual void serialize(DB::Store_Writer & store,
    //                       const Feature & feature) const;

    //virtual void reconstitute(DB::Store_Reader & store,
    //                          Feature & feature) const;
    //virtual void serialize(DB::Store_Writer & store,
    //                       const Feature & feature,
    //                       float value) const;

    //virtual void reconstitute(DB::Store_Reader & store,
    //                          const Feature & feature,
    //                          float & value) const;


    //virtual std::string print(const Feature_Set & fs) const;
    //virtual void serialize(DB::Store_Writer & store,
    //                       const Feature_Set & fs) const;
    //virtual void reconstitute(DB::Store_Reader & store,
    //                          boost::shared_ptr<Feature_Set> & fs) const;


    virtual std::string class_id() const = 0;

    enum Type {
        DENSE,   ///< Dense feature space
        SPARSE   ///< Sparse feature space
    };

    virtual Type type() const = 0;

    //virtual void serialize(DB::Store_Writer & store) const;
    //virtual void reconstitute(DB::Store_Reader & store,
    //                        const boost::shared_ptr<const Feature_Space> & fs);
    
    virtual Feature_Space * make_copy() const = 0;

    virtual std::string print() const;

    virtual boost::shared_ptr<Training_Data>
    training_data(const boost::shared_ptr<const Feature_Space> & fs) const;

    virtual void freeze();
};


} // namespace ML
