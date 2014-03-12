/* feature_map_test.cc
   Jeremy Barnes, 22 February 2005
   Copyright (c) Jeremy Barnes 2005.  All rights reserved.
   $Source$

   Test of various implementations of the Feature_Map class.
*/

#include "jml/boosting/feature_map.h"
#include "jml/utils/string_functions.h"
#include <iostream>
#include <vector>
#include <map>
#include <boost/timer.hpp>
#include "jml/utils/hash_map.h"

using namespace std;
using namespace ML;


template<class Array>
void do_timed_test(Array & array, const vector<Feature> & features_,
                   const string & name)
{
    cerr << "testing " << name << " with " << features_.size()
         << " entries" << endl;

    array.clear();

    vector<Feature> features = features_;

    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()),
                   features.end());
    std::random_shuffle(features.begin(), features.end());

    int NUM = features.size();
    double total = 0.0;

    boost::timer t;
    for (unsigned i = 0;  i < NUM;  ++i)
        array[features[i]] = i;
    total += t.elapsed();
    cerr << "random insert: " << format("%6.2fs", t.elapsed()) << endl;

    t.restart();
    for (typename Array::const_iterator it = array.begin();
         it != array.end();  ++it) {
    }
    total += t.elapsed();
    cerr << "iterate:       " << format("%6.2fs", t.elapsed()) << endl;
    
    t.restart();
    array.clear();
    total += t.elapsed();
    cerr << "clear:         " << format("%6.2fs", t.elapsed()) << endl;
    
    t.restart();
    std::sort(features.begin(), features.end());
    //cerr << "sort: " << format("%6.2fs", t.elapsed()) << endl;
    
    t.restart();
    for (unsigned i = 0;  i < NUM;  ++i)
        array[features[i]] = i;
    total += t.elapsed();
    cerr << "sorted insert: " << format("%6.2fs", t.elapsed()) << endl;

    for (unsigned i = 0;  i < features.size();  ++i)
        if (array[features[i]] != i) {
            cerr << "i = " << i << " features[i] = " << features[i].print()
                 << endl;
            cerr << "array.size() = " << array.size() << endl;

            throw Exception("error: insert not working: element "
                            + ostream_format(i) + " (feature " 
                            + features[i].print() + ") = "
                            + ostream_format(array[features[i]]));
        }

    t.restart();
    std::random_shuffle(features.begin(), features.end());
    //cerr << "shuffle: " << format("%6.2fs", t.elapsed()) << endl;

    t.restart();
    for (unsigned i = 0;  i < NUM;  ++i)
        array[features[i]] += 1;
    total += t.elapsed();
    cerr << "random lookup: " << format("%6.2fs", t.elapsed()) << endl;

    cerr << "TOTAL:         " << format("%6.2fs", total) << endl;
    cerr << "MEM:           "
         << format("%6.4fM", memusage(array) / 1024.0 / 1024.0) << endl;
    cerr << endl;
}

#if 0
    /* Finalize the data structures. */
    for (Itl::index_type::iterator it = itl->index.begin();
         it != itl->index.end();  ++it) {
        Feature feature = it.key();

        //cerr << "finalizing feature " << itl->feature_space->print(feature)
        //     << endl;

        it->finalize(data.example_count(), feature, itl->feature_space);
        itl->all_features.push_back(feature);

        //cerr << "  " << it->print_info() << endl;
        //cerr << "  examples = " << it->examples.size() << endl;
        //cerr << "  values   = " << it->values.size() << endl;
    }

    std::sort(itl->all_features.begin(), itl->all_features.end());
#endif


void profile(const vector<Feature> & features, string name)
{
    cerr << "******************************" << endl;
    cerr << "testing " << name << endl;
    typedef std::map<Feature, unsigned long> array1_type;
    array1_type array1;

    typedef std::hash_map<Feature, unsigned long> array2_type;
    array2_type array2;

    typedef Feature_Map<unsigned long> array3_type;
    array3_type array3;

    do_timed_test(array1, features, "map");
    do_timed_test(array2, features, "hash_map");
    do_timed_test(array3, features, "feature_map");
    cerr << endl << endl;
}

void profile1()
{
    int NUM = 1000000;
    
    /* Create the features. */
    vector<Feature> features(NUM);

    for (unsigned i = 0;  i < NUM;  ++i)
        features[i] = Feature(rand(), rand(), rand());

    profile(features, "all random features");
}

void profile2()
{
    int NUM = 1000000;
    
    /* Create the features. */
    vector<Feature> features(NUM);

    for (unsigned i = 0;  i < NUM;  ++i)
        features[i] = Feature(rand(), 0, 0);

    profile(features, "first random features");
}

void profile3()
{
    int NUM = 1000000;
    
    /* Create the features. */
    vector<Feature> features(NUM);

    for (unsigned i = 0;  i < NUM;  ++i)
        features[i] = Feature(i, 0, 0);

    std::random_shuffle(features.begin(), features.end());

    profile(features, "first sequential features");
}

void profile4()
{
    int NUM = 1000000;
    
    /* Create the features. */
    vector<Feature> features(NUM);

    for (unsigned i = 0;  i < NUM / 2;  ++i)
        features[i] = Feature(i, 0, 0);

    for (unsigned i = NUM / 2;  i < NUM;  ++i)
        features[i] = Feature(i, rand(), rand());

    std::random_shuffle(features.begin(), features.end());

    profile(features, "half sequential half random");
}

void sanity1()
{
    Feature_Map<unsigned long> array;

    int NUM = 32;
    
    /* Create the features. */
    vector<Feature> features(NUM);
    
    for (unsigned i = 0;  i < NUM;  ++i)
        features[i] = Feature(rand(), rand(), rand());
    
    std::sort(features.begin(), features.end());

    features.erase(std::unique(features.begin(), features.end()),
                   features.end());
    std::random_shuffle(features.begin(), features.end());

    NUM = features.size();
    
    cerr << "testing insert" << endl;
    for (unsigned i = 0;  i < NUM;  ++i) {
        array[features[i]] = i;

        if (!array.count(features[i])) {
            for (Feature_Map<unsigned long>::const_iterator it 
                     = array.begin();  it != array.end();
                 ++it) {
                cerr << it.key().print() << " " << *it << endl;
            }
            throw Exception("error: insert not working: element "
                            + ostream_format(i) + " (feature " 
                            + features[i].print() + ") doesn't exist");
        }
            
        if (array[features[i]] != i)
            throw Exception("error: insert not working: element "
                            + ostream_format(i) + " (feature " 
                            + features[i].print() + ") = "
                            + ostream_format(array[features[i]]));
    }
    cerr << "done testing insert" << endl;
    
    array.clear();
    
    std::sort(features.begin(), features.end());
    
    cerr << "testing sorted insert" << endl;
    for (unsigned i = 0;  i < NUM;  ++i)
        array[features[i]] = i;

    for (unsigned i = 0;  i < features.size();  ++i) {
        if (array[features[i]] != i) {
            cerr << "i = " << i << " features[i] = " << features[i].print()
                 << endl;
            cerr << "array.size() = " << array.size() << endl;
            for (Feature_Map<unsigned long>::const_iterator it 
                     = array.begin();  it != array.end();
                 ++it) {
                cerr << it.key().print() << " " << *it << endl;
            }

            throw Exception("error: insert not working: element "
                            + ostream_format(i) + " (feature " 
                            + features[i].print() + ") = "
                            + ostream_format(array[features[i]]));
        }
    }

    std::random_shuffle(features.begin(), features.end());
    for (unsigned i = 0;  i < NUM;  ++i)
        array[features[i]] += 1;

    cerr << "MEM:           "
         << format("%6.4fM", memusage(array) / 1024.0 / 1024.0) << endl;
    cerr << endl;
}

void test1()
{
    Feature_Map<unsigned long> array;
    int NUM = 1000;
    
    for (unsigned i = 0;  i < NUM;  ++i)
        array[Feature(rand(), rand(), rand())] = i;
}

void test2()
{
    int NUM1 = 10;
    int NUM2 = 10;
    
    /* Create the features. */
    vector<Feature> features(NUM1 * NUM2);

    for (unsigned i = 0;  i < NUM1;  ++i) {
        for (unsigned j = 0;  j < NUM2;  ++j)
            features[i * NUM2 + j] = Feature(i, rand(), rand());
    }

    std::sort(features.begin(), features.end());
    features.erase(std::unique(features.begin(), features.end()),
                   features.end());

    typedef Feature_Map<int> array_type;
    array_type array;

    for (unsigned i = 0;  i < features.size();  ++i)
        array[features[i]] = i;

    if (array.size() != features.size())
        throw Exception("sizes don't match");

    cerr << "features.size() = " << features.size()
         << "  array.size() = " << array.size() << endl;

    vector<Feature> features2;

    for (array_type::iterator it = array.begin(); it != array.end();
         ++it) {
        Feature feature = it.key();
        cerr << "feature: " << feature << endl;
        features2.push_back(feature);
    }

    std::sort(features2.begin(), features2.end());

    if (features2.size() != features.size())
        throw Exception("features.size() != features.size()");

    for (unsigned i = 0;  i < features.size();  ++i) {
        if (features2[i] != features[i])
            throw Exception("iterated feature "
                            + ostream_format(i) + " doesn't match: current = "
                            + features2[i].print() + " supposed to be: "
                            + features[i].print());
    }

    //std::random_shuffle(features.begin(), features.end());

    for (unsigned i = 0;  i < features.size();  ++i)
        array[features[i]] = i;

    for (unsigned i = 0;  i < features.size();  ++i)
        if (array[features[i]] != i)
            throw Exception("error: insert not working: element "
                            + ostream_format(i) + " (feature " 
                            + features[i].print() + ") = "
                            + ostream_format(array[features[i]]));
}


int main(int argc, char ** argv)
try
{
    sanity1();
    test1();
    test2();
    profile1();
    profile2();
    profile3();
    profile4();
}
catch (const std::exception & exc) {
    cerr << "caught exception: " << exc.what() << endl;
    exit(1);
}
