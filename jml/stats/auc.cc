/* auc.cc
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Area under the curve code.
*/

#include "auc.h"
#include <algorithm>


using namespace std;


namespace ML {

double do_calc_auc(std::vector<AUC_Entry> & entries)
{
    // 1.  Total number of positive and negative
    int num_neg = 0, num_pos = 0;

    for (unsigned i = 0;  i < entries.size();  ++i) {
        if (entries[i].weight == 0.0) continue;
        if (entries[i].target == false) ++num_neg;
        else ++num_pos;
    }

    // 2.  Sort
    std::sort(entries.begin(), entries.end());
    
    // 3.  Get (x,y) points and calculate the AUC
    int total_pos = 0, total_neg = 0;

    float prevx = 0.0, prevy = 0.0;

    double total_area = 0.0, total_weight = 0.0, current_weight = 0.0;

    for (unsigned i = 0;  i < entries.size();  ++i) {
        if (entries[i].weight > 0.0) {
            if (entries[i].target == false) ++total_neg;
            else ++total_pos;
        }
        
        current_weight += entries[i].weight;
        total_weight += entries[i].weight;

        if (i != entries.size() - 1
            && entries[i].model == entries[i + 1].model)
            continue;
        
        if (entries[i].weight == 0.0) continue;

        float x = total_pos * 1.0 / num_pos;
        float y = total_neg * 1.0 / num_neg;

        double area = (x - prevx) * (y + prevy) * 0.5;

        total_area += /* current_weight * */ area;

        prevx = x;
        prevy = y;
        current_weight = 0.0;
    }

    // TODO: get weighted working properly...

    //cerr << "total_area = " << total_area << " total_weight = "
    //     << total_weight << endl;

    if (total_pos != num_pos || total_neg != num_neg)
        throw Exception("bad total pos or total neg");

    // 4.  Convert to gini
    //double gini = 2.0 * (total_area - 0.5);

    // 5.  Final score is absolute value.  Since we want an error, we take
    //     1.0 - the gini
    //return 1.0 - fabs(gini);
    return 1.0 - total_area;
}

} // namespace ML
