/* null_feature_space.h                                            -*- C++ -*-
   Jeremy Barnes, 21 July 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A feature space with no features at all.  Useful primarily when you need
   to fill in.
*/

#ifndef __boosting__null_feature_space_h__
#define __boosting__null_feature_space_h__


#include "feature_space.h"



namespace ML {


/*****************************************************************************/
/* NULL_FEATURE_SPACE                                                        */
/*****************************************************************************/

/** A feature space with no features at all.  Useful primarily when you need
    to fill in a feature space where in fact you are not using one.
*/

class Null_Feature_Space : public Feature_Space {
public:
    Null_Feature_Space();
    Null_Feature_Space(DB::Store_Reader & store);

    virtual ~Null_Feature_Space();

    virtual Feature_Info info(const Feature & feature) const;

    virtual Type type() const { return DENSE; }

    using Feature_Space::print;
    using Feature_Space::parse;
    using Feature_Space::serialize;
    using Feature_Space::reconstitute;

    /* Methods to deal with features. */
    virtual std::string print(const Feature & feature) const;
    virtual bool parse(Parse_Context & context, Feature & feature) const;
    virtual void expect(Parse_Context & context, Feature & feature) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature & feature) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              Feature & feature) const;

    /** Methods to deal with a feature set. */
    virtual std::string print(const Feature_Set & fs) const;
    virtual void serialize(DB::Store_Writer & store,
                           const Feature_Set & fs) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              Feature_Set & fs) const;

    /** Serialization and reconstitution. */
    virtual std::string class_id() const;
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);
    void reconstitute(DB::Store_Reader & store);

    /* Methods to deal with the thing as a whole. */
    virtual Null_Feature_Space * make_copy() const;
    virtual std::string print() const;
};

} // namespace ML


#endif /* __boosting__null_feature_space_h__ */
