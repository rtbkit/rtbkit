/* classifier_mlp_averager.h                                    -*- C++ -*-
   Jeremy Barnes, 2 September 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Classifier class using an MlpAverager interface.
*/

#ifndef __boosting__classifier_mlp_averager_h__
#define __boosting__classifier_mlp_averager_h__

#include "config.h"
#include "classifier.h"


namespace ML {


/*****************************************************************************/
/* Classifier_MLP_Averager                                                    */
/*****************************************************************************/

/** Class that has an MlpAverager interface but uses a classifier underneath. */

class Classifier_MLP_Averager {

  public:
    //----------------------------------------
    // Constructor, Destructor
    //----------------------------------------
    Classifier_MLP_Averager         ();
    Classifier_MLP_Averager         ( DB::Store_Reader& store );
    ~Classifier_MLP_Averager        ();

    //----------------------------------------
    // computeScore:  Returns the decoded output
    //----------------------------------------
    std::vector<double> computeScore(const std::vector<double>& fv) const;
    
    //----------------------------------------
    // Read - Write the averager
    //----------------------------------------
    void                read        ( const std::string & fName );

    void                reconstitute(DB::Store_Reader & store);
    
    // Check if the averager has been initialized
    bool                valid       () const;
    
  private:
    ML::Classifier classifier;
};


} // namespace ML


#endif /* __boosting__classifier_mlp_averager_h__ */
