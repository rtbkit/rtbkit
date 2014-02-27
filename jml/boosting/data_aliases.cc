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
#include "jml/utils/floating_point.h"
#include "jml/utils/worker_task.h"
#include "jml/arch/timers.h"
#include <mutex>

using namespace std;


namespace ML {

std::vector<Alias>
aliases(const ML::Training_Data & dataset, const Feature & predicted)
{
    Timer timer;

    auto hashFeatureSet = [&] (const ML::Feature_Set & features) -> uint64_t
        {
            std::string printed;
            printed.reserve(4096);

            uint64_t result = 1231223;
            for (auto f: features) {
                Feature feat;
                float value;
                std::tie(feat, value) = f;
                
                if (feat == predicted)
                    continue;

                //printed += dataset.feature_space()->print(feat, value);

                uint64_t featHash = feat.hash();
                uint64_t valueHash = float_hasher()(value);

                if (featHash == 0)
                    featHash = 1;
                if (valueHash == 0)
                    valueHash = 1;
                
                result = chain_hash(chain_hash(featHash, valueHash), result);
            }

            return result; // ^ std::hash<string>()(printed);
        };

    //cerr << "aliases" << endl;
    vector<tuple<int, uint64_t, float> > sorted[32];
    ML::Spinlock sortedLocks[32];
    //sorted.resize(dataset.example_count());

    auto doHash = [&] (int exampleNum)
        {
            auto example = dataset.get(exampleNum);
            uint64_t hash = hashFeatureSet(*example);
            float label = (*example)[predicted];

            //cerr << "example " << exampleNum << " hash " << hash << endl;

            std::unique_lock<ML::Spinlock> guard(sortedLocks[hash % 32]);
            sorted[hash % 32].emplace_back(exampleNum, hash, label);
        };

    // Hash the examples in parallel
    run_in_parallel_blocked(0, dataset.example_count(), doHash);

    cerr << "hashed examples in " << timer.elapsed() << endl;

    uint64_t comparisons = 0, nontrivialComparisons = 0;

    auto compareExamples = [&] (const std::tuple<int, uint64_t, float> & ex1,
                                const std::tuple<int, uint64_t, float> & ex2)
        -> int
        {
            ML::atomic_inc(comparisons);

            uint64_t hash1 = std::get<1>(ex1);
            uint64_t hash2 = std::get<1>(ex2);

            if (hash1 < hash2)
                return -1;
            else if (hash1 > hash2)
                return 1;

            ML::atomic_inc(nontrivialComparisons);

            // Same hash; compare the contents
            auto fv1 = dataset.get(std::get<0>(ex1));
            auto fv2 = dataset.get(std::get<0>(ex2));

            int res = fv1->compare(*fv2, predicted);

            if (res == -1)
                return -1;
            else if (res == 1)
                return 1;
            else if (res != 0)
                throw ML::Exception("unexpected compare result");

            return 0;  // equal
        };

    auto lessExamples = [&] (const std::tuple<int, uint64_t, float> & ex1,
                             const std::tuple<int, uint64_t, float> & ex2)
        {
            return compareExamples(ex1, ex2) == -1;
        };

    vector<Alias> bucketResults[32];

    // Check for aliases within the bucket
    auto processBucket = [&] (int i)
        {
            auto & bucket = sorted[i];

            // Now sort them
            std::sort(bucket.begin(), bucket.end(), lessExamples);

            std::vector<Alias> & bucketResult = bucketResults[i];

            int x = 0;
            while (x < bucket.size()) {
                int b = x++;  // beginning of equal range
                while (x < bucket.size()
                       && compareExamples(bucket[x], bucket[b]) == 0)
                    ++x;
        
                if (x == b + 1) continue;  // only one in the range

                Alias alias;
                alias.homogenous = true;
                int label = (int)(std::get<2>(bucket[b]));
                for (unsigned i = b;  i < x;  ++i) {
                    alias.examples.insert(std::get<0>(bucket[i]));
                    if ((int)std::get<2>(bucket[i]) != label)
                        alias.homogenous = false;
                }
                
                
                bucketResult.push_back(alias);
            }

        };

    run_in_parallel(0, 32, processBucket);

    // Assemble a final result
    vector<Alias> result;
    for (auto & r: bucketResults)
        result.insert(result.end(), r.begin(), r.end());

    cerr << "comparisons " << comparisons << " non-trivial " << nontrivialComparisons
         << " " << 100.0 * nontrivialComparisons / comparisons << "%" << endl;

    cerr << "alias detection took " << timer.elapsed() << endl;

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

