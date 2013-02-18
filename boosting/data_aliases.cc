/* data_aliases.cc
   Jeremy Barnes, 18 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

*/

#include "data_aliases.h"
#include <vector>
#include <utility>
#include <algorithm>
#include "jml/utils/sgi_numeric.h"


using namespace std;


namespace ML {

namespace {

/** Internal structure to compare two feature sets at the end of pointer to
    an Example_Data structure.
*/
struct Compare_Example_Data {
    Compare_Example_Data(const Feature & to_ignore)
        : to_ignore(to_ignore)
    {
    }

    typedef const pair<int, std::shared_ptr<const ML::Feature_Set> >
          arg_type;
    bool operator() (const arg_type & p1, const arg_type & p2) const
    {
        return p1.second->compare(*p2.second, to_ignore) == -1;
    }

    Feature to_ignore;
};

} // file scope

std::vector<Alias>
aliases(const ML::Training_Data & dataset, const Feature & predicted)
{
    //cerr << "aliases" << endl;
    vector<pair<int, std::shared_ptr<const ML::Feature_Set> > > sorted;
    sorted.reserve(dataset.example_count());
    for (unsigned i = 0;  i < dataset.example_count();  ++i)
        sorted.push_back(make_pair(i, dataset.get(i)));
    std::sort(sorted.begin(), sorted.end(), Compare_Example_Data(predicted));
    
    vector<Alias> result;
    int x = 0;
    while (x < sorted.size()) {
        int b = x++;  // beginning of equal range
        while (x < sorted.size()
               && sorted[x].second->compare(*sorted[b].second, predicted) == 0)
            ++x;
        
        if (x == b + 1) continue;  // only one in the range

        Alias alias;
        alias.homogenous = true;
        int label = (int)(*sorted[b].second)[predicted];
        for (unsigned i = b;  i < x;  ++i) {
            alias.examples.insert(sorted[i].first);
            if ((*sorted[i].second)[predicted] != label)
                alias.homogenous = false;
        }

        result.push_back(alias);
    }

    return result;
}

int
remove_aliases(ML::Training_Data & dataset,
               const std::vector<Alias> & aliases,
               bool homogenous,
               std::vector<int> * mapping)
{
    //cerr << "remove_aliases" << endl;
    set<int> removed;

    for (unsigned i = 0;  i < aliases.size();  ++i) {
        if (homogenous == false && aliases[i].homogenous) continue;
        removed.insert(aliases[i].examples.begin(), aliases[i].examples.end());
    }

    if (removed.empty()) {
        if (mapping) {
            mapping->resize(dataset.example_count());
            std::iota(mapping->begin(), mapping->end(), 0);
        }
        return 0;  // no work; don't do the expensive stuff
    }

    if (mapping) mapping->clear();

    Training_Data new_data(dataset.feature_space());
    for (int x = 0;  x < dataset.example_count();  ++x) {
        if (removed.count(x)) continue;
        new_data.add_example(dataset.share(x));
        if (mapping) mapping->push_back(x);
    }
    
    dataset.swap(new_data);
    
    return removed.size();
}

std::vector<Alias>
remove_aliases(ML::Training_Data & dataset, const Feature & predicted,
               bool homogenous,
               std::vector<int> * mapping)
{
    vector<Alias> aliased = aliases(dataset, predicted);
    remove_aliases(dataset, aliased, homogenous, mapping);
    return aliased;
}

} // namespace ML

