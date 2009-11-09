/* auc.h                                                           -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Functionality to calculate area under the curve statistics.
*/

#ifndef __jml__stats__auc_h__
#define __jml__stats__auc_h__

#include <vector>
#include "arch/exception.h"

namespace ML {

struct AUC_Entry {
    AUC_Entry(float model = 0.0, float target = 0.0, float weight = 1.0)
        : model(model), target(target), weight(weight)
    {
    }

    float model;
    float target;
    float weight;

    bool operator < (const AUC_Entry & other) const
    {
        return model < other.model;
    }
};

double do_calc_auc(std::vector<AUC_Entry> & entries);


template<typename Float1, typename Float2>
double
calc_auc(const std::vector<Float1> & outputs,
         const std::vector<Float2> & targets)
{
    if (targets.size() != outputs.size())
        throw Exception("targets and predictions don't match");
    
    std::vector<AUC_Entry> entries;
    entries.reserve(outputs.size());
    for (unsigned i = 0;  i < outputs.size();  ++i)
        entries.push_back(AUC_Entry(outputs[i], targets[i]));
    
    return do_calc_auc(entries);
}

template<typename Float1, typename Float2, typename Float3>
double
calc_auc(const std::vector<Float1> & outputs,
         const std::vector<Float2> & targets,
         const std::vector<Float3> & weights)
{
    if (targets.size() != outputs.size())
        throw Exception("targets and predictions don't match");
    if (weights.size() != outputs.size())
        throw Exception("targets and weights don't match");
    
    std::vector<AUC_Entry> entries;
    entries.reserve(outputs.size());
    for (unsigned i = 0;  i < outputs.size();  ++i)
        entries.push_back(AUC_Entry(outputs[i], targets[i], weights[i]));
    
    return do_calc_auc(entries);
}

} // namespace ML

#endif /* __jml__stats__auc_h__ */
