/* classifier_persist_impl.h                                       -*- C++ -*-
   Jeremy Barnes, 21 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of persistence functions for the classifier.  Only needs to
   be included by subclasses of the classifier, so split off from here in order
   to avoid everyone having to include it.
*/

#ifndef __boosting__classifier_persist_impl_h__
#define __boosting__classifier_persist_impl_h__

#include "classifier.h"
#include "registry.h"


namespace ML {


/*****************************************************************************/
/* REGISTRY FUNCTIONS                                                        */
/*****************************************************************************/

/* Functions for the registry of objects. */

template<>
class Factory_Base<Classifier_Impl> {
public:
    virtual ~Factory_Base();

    virtual std::shared_ptr<Classifier_Impl> create() const = 0;

    virtual std::shared_ptr<Classifier_Impl>
    reconstitute(const std::shared_ptr<const Feature_Space> & feature_space,
                 DB::Store_Reader & store) const = 0;
};

template<class Derived>
class Object_Factory<Classifier_Impl, Derived>
    : public Factory_Base<Classifier_Impl> {
public:
    virtual ~Object_Factory() {}
    virtual std::shared_ptr<Classifier_Impl>
    create() const
    {
        std::shared_ptr<Classifier_Impl> result(new Derived());
        return result;
    }

    virtual std::shared_ptr<Classifier_Impl>
    reconstitute(const std::shared_ptr<const Feature_Space> & feature_space,
                 DB::Store_Reader & store) const
    {
        std::shared_ptr<Classifier_Impl> result = create();
        //cerr << "created: predicted = " << result->predicted() << endl;
        result->reconstitute(store, feature_space);
        //cerr << "reconstituted: predicted = " << result->predicted() << endl;
        return result;
    }
};


} // namespace ML



#endif /* __boosting__classifier_persist_impl_h__ */
