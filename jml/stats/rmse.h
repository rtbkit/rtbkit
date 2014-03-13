/* rmse.h                                                          -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Root Mean Squared Error calculation routines.
*/

#ifndef __jml__stats__rmse_h__
#define __jml__stats__rmse_h__

#include "jml/stats/distribution.h"
#include "jml/stats/distribution_ops.h"

namespace ML {

template<typename Float1, typename Float2>
double
calc_rmse(const distribution<Float1> & outputs,
          const distribution<Float2> & targets)
{
    return sqrt(sqr((targets - outputs)).total()
                * (1.0 / outputs.size()));
}

template<typename Float1, typename Float2, typename Float3>
double
calc_rmse(const distribution<Float1> & outputs,
          const distribution<Float2> & targets,
          const distribution<Float3> & weights)
{
    return sqrt((sqr((targets - outputs)) * weights).total()
                / weights.total());
}


} // namespace ML


#endif /* __jml__stats__rmse_h__ */
