/* auc.h                                                           -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Functionality to calculate area under the curve statistics.
*/

#ifndef __jml__stats__auc_h__
#define __jml__stats__auc_h__

#include <vector>
#include "jml/arch/exception.h"
#include <iostream>

namespace ML {

struct AUC_Entry {
    AUC_Entry(float model = 0.0, bool target = false, float weight = 1.0)
        : model(model), target(target), weight(weight)
    {
    }

    float model;
    bool target;
    float weight;

    bool operator < (const AUC_Entry & other) const
    {
        return model < other.model;
    }
};

double do_calc_auc(std::vector<AUC_Entry> & entries);


template<typename Float1, typename Float2, typename Float3>
double
calc_auc(const std::vector<Float1> & outputs,
         const std::vector<Float2> & targets,
         Float3 neg_val, Float3 pos_val)
{
    if (targets.size() != outputs.size())
        throw Exception("targets and predictions don't match");
    
    std::vector<AUC_Entry> entries;
    entries.reserve(outputs.size());
    for (unsigned i = 0;  i < outputs.size();  ++i) {
        bool target;
        if (targets[i] == neg_val) target = false;
        else if (targets[i] == pos_val) target = true;
        else {
            using namespace std;
            cerr << "i = " << i << " of " << outputs.size() << endl;
            cerr << "targets[i] = " << targets[i] << " not weighted" << endl;
            throw Exception("calc_auc(): "
                            "target value %f wasn't neg %f or pos %f value",
                            targets[i], neg_val, pos_val);
        }
        entries.push_back(AUC_Entry(outputs[i], target));
    }
    
    return do_calc_auc(entries);
}

template<typename Float1, typename Float2, typename Float3, typename Float4>
double
calc_auc(const std::vector<Float1> & outputs,
         const std::vector<Float2> & targets,
         const std::vector<Float3> & weights,
         Float4 neg_val, Float4 pos_val)
{
    if (targets.size() != outputs.size())
        throw Exception("targets and predictions don't match");
    if (weights.size() != outputs.size())
        throw Exception("targets and weights don't match");
    
    std::vector<AUC_Entry> entries;
    entries.reserve(outputs.size());
    for (unsigned i = 0;  i < outputs.size();  ++i) {
        bool target;
        if (targets[i] == neg_val) target = false;
        else if (targets[i] == pos_val) target = true;
        else {
            using namespace std;
            cerr << "i = " << i << " of " << outputs.size() << endl;
            cerr << "targets[i] = " << targets[i] << " weighted" << endl;
            throw Exception("calc_auc(): "
                              "target value wasn't neg or pos value");
        }

        entries.push_back(AUC_Entry(outputs[i], target, weights[i]));
    }
    
    return do_calc_auc(entries);
}

} // namespace ML

#endif /* __jml__stats__auc_h__ */
