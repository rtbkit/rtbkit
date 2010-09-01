/* classifier_mlp.h                                                -*- C++ -*-
   Jeremy Barnes, 2 September 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Classifier class using an MLP interface.
*/

#ifndef __boosting__classifier_mlp_h__
#define __boosting__classifier_mlp_h__

#include "config.h"
#include "classifier.h"


namespace ML {


/*****************************************************************************/
/* CLASSIFIER_MLP                                                            */
/*****************************************************************************/

/** Class that has an MLP interface but uses a classifier underneath. */

class Classifier_MLP {
public:
    //----------------------------------------
    // Constructor, Destructor
    //----------------------------------------
    Classifier_MLP         ();
    Classifier_MLP         ( DB::Store_Reader& store );
    ~Classifier_MLP        ();

    //----------------------------------------
    // Test:  Returns the decoded output
    //----------------------------------------
    std::vector<double> test(const std::vector<double>& fv) const;
    
    //----------------------------------------
    // Read - Write Network
    //----------------------------------------
    void                read       ( const std::string & fName );

    void                reconstitute(DB::Store_Reader & store);
    
    // Check if the network has been initialized
    bool                valid      () const;
    
    //----------------------------------------
    // Test:  Returns the non decoded output (for those who might want it)
    //----------------------------------------
    std::vector<double> compute(const std::vector<double> & fv) const;
    
private:
    ML::Classifier classifier;
};


} // namespace ML


#endif /* __boosting__classifier_mlp_h__ */
