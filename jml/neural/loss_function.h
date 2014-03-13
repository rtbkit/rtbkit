/* loss_function.h                                                 -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Loss functions for discriminative training.
*/

#ifndef __jml__neural__loss_function_h__
#define __jml__neural__loss_function_h__

#include "jml/stats/distribution.h"

namespace ML {


/*****************************************************************************/
/* LOSS_FUNCTION_TYPE                                                        */
/*****************************************************************************/

enum Loss_Function_Type {
    LF_MSE,
    LF_CUSTOM  ///< Non-standard loss function type
};

/*****************************************************************************/
/* LOSS_FUNCTION                                                             */
/*****************************************************************************/

/** A loss function; used principally to generate an error gradient for
    gradient descent training. */

struct Loss_Function {


    /*************************************************************************/
    /* LOSS                                                                  */
    /*************************************************************************/
    
    virtual float loss(const distribution<float> & input,
                       const distribution<float> & target) const = 0;

    virtual double loss(const distribution<double> & input,
                        const distribution<double> & target) const = 0;


    /*************************************************************************/
    /* DLOSS                                                                 */
    /*************************************************************************/

    virtual
    distribution<float>
    dloss(const distribution<float> & input,
          const distribution<float> & target) const = 0;

    virtual
    distribution<double>
    dloss(const distribution<double> & input,
          const distribution<double> & target) const = 0;
};


/*****************************************************************************/
/* MSE_LOSS_FUNCTION                                                         */
/*****************************************************************************/

/** Mean squared error loss function. */

struct MSE_Loss {

    /*************************************************************************/
    /* LOSS                                                                  */
    /*************************************************************************/
    
    virtual float loss(const distribution<float> & input,
                       const distribution<float> & target) const;

    virtual double loss(const distribution<double> & input,
                        const distribution<double> & target) const;


    /*************************************************************************/
    /* DLOSS                                                                 */
    /*************************************************************************/

    virtual
    distribution<float>
    dloss(const distribution<float> & input,
          const distribution<float> & target) const;

    virtual
    distribution<double>
    dloss(const distribution<double> & input,
          const distribution<double> & target) const;
};


/*****************************************************************************/
/* FACTORY                                                                   */
/*****************************************************************************/

// ...

} // namespace ML

#endif /* __jml__neural__loss_function_h__ */
