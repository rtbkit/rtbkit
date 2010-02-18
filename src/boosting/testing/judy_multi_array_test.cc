/* judy_multi_array_test.cc
   Jeremy Barnes, 21 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Test of the Judy multi array class.
*/

#include "jml/boosting/judy_multi_array.h"
#include "jml/utils/minivec.h"
#include "jml/utils/string_functions.h"
#include <iostream>
#include <vector>
#include <map>
#include <boost/timer.hpp>



using namespace ML;
using namespace std;


struct Minivec_Extractor {
    template<size_t idx>
    static unsigned long get(const minivec<long, 3> & vec)
    {
        return vec[idx];
    }

    template<size_t idx>
    static void put(minivec<long, 3> & vec, unsigned long val)
    {
        vec.resize(3);
        vec[idx] = val;
    }
};

template<typename T, unsigned char L>
ostream & operator << (ostream & stream, const minivec<T, L> & vec)
{
    stream << "[ ";
    for (unsigned i = 0;  i < vec.size();  ++i)
        stream << vec[i] << " ";
    stream << "]";
    return stream;
}

void test1()
{
    vector<minivec<long, 3> > indexes(10);
    
    for (unsigned i = 0;  i < indexes.size();  ++i) {
        indexes[i].push_back(rand());
        indexes[i].push_back(rand());
        indexes[i].push_back(rand());
    }
    
    typedef judy_multi_array<minivec<long, 3>, int, Minivec_Extractor, 3>
        array_type;
    array_type array;

    for (unsigned i = 0;  i < indexes.size();  ++i) {
        array[indexes[i]] = i;
    }

    for (unsigned i = 0;  i < indexes.size();  ++i) {
        cerr << "array[" << indexes[i] << "] = " << array[indexes[i]]
             << endl;
    }

    for (array_type::iterator it = array.begin();  it != array.end();  ++it) {
        cerr << it.key() << " = " << *it << endl;
    }

    array_type array2 = array;

    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " = " << *it << endl;
    }
}

template<class Array>
void do_timed_test(Array & array, vector<minivec<long, 3> > & indexes)
{
    int NUM = indexes.size();

    double total = 0.0;

    boost::timer t;
    for (unsigned i = 0;  i < NUM;  ++i)
        array[indexes[i]] = i;
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
    std::sort(indexes.begin(), indexes.end());
    //cerr << "sort: " << format("%6.2fs", t.elapsed()) << endl;
    
    t.restart();
    for (unsigned i = 0;  i < NUM;  ++i)
        array[indexes[i]] = i;
    total += t.elapsed();
    cerr << "sorted insert: " << format("%6.2fs", t.elapsed()) << endl;

    t.restart();
    std::random_shuffle(indexes.begin(), indexes.end());
    //cerr << "shuffle: " << format("%6.2fs", t.elapsed()) << endl;
    
    t.restart();
    for (unsigned i = 0;  i < NUM;  ++i)
        array[indexes[i]] += 1;
    total += t.elapsed();
    cerr << "random lookup: " << format("%6.2fs", t.elapsed()) << endl;

    cerr << "TOTAL:         " << format("%6.2fs", total) << endl;
    cerr << "MEM:           "
         << format("%6.4fM", memusage(array) / 1024.0 / 1024.0) << endl;
    cerr << endl;
    cerr << endl;
}

template<class Array>
void timed_test1(Array & array)
{
    array.clear();

    int NUM = 2000000;

    /* Create the vectors. */
    vector<minivec<long, 3> > indexes(NUM);

    for (unsigned i = 0;  i < NUM;  ++i) {
        indexes[i].push_back(rand());
        indexes[i].push_back(rand());
        indexes[i].push_back(rand());
    }

    do_timed_test(array, indexes);
}

template<class Array>
void timed_test2(Array & array)
{
    array.clear();

    int NUM = 2000000;

    /* Create the vectors. */
    vector<minivec<long, 3> > indexes(NUM);

    for (unsigned i = 0;  i < NUM;  ++i) {
        indexes[i].push_back(0);
        indexes[i].push_back(0);
        indexes[i].push_back(rand());
    }

    do_timed_test(array, indexes);
}

template<class Array>
void timed_test3(Array & array)
{
    array.clear();

    int NUM = 2000000;

    /* Create the vectors. */
    vector<minivec<long, 3> > indexes(NUM);

    for (unsigned i = 0;  i < NUM;  ++i) {
        indexes[i].push_back(rand());
        indexes[i].push_back(0);
        indexes[i].push_back(0);
    }

    do_timed_test(array, indexes);
}

void profile1()
{
    map<minivec<long, 3>, int> map_array;
    cerr << "map all random:" << endl;
    timed_test1(map_array);
    cerr << endl << endl;
}

void profile2()
{
    typedef judy_multi_array<minivec<long, 3>, long, Minivec_Extractor, 3>
        array_type;
    array_type array;
    cerr << "judy_multi_array all random:" << endl;
    timed_test1(array);
    cerr << endl << endl;
}

void profile3()
{
    map<minivec<long, 3>, int> map_array;
    cerr << "map last random:" << endl;
    timed_test2(map_array);
    cerr << endl << endl;
}

void profile4()
{
    typedef judy_multi_array<minivec<long, 3>, long, Minivec_Extractor, 3>
        array_type;
    array_type array;
    cerr << "judy_multi_array last random:" << endl;
    timed_test2(array);
    cerr << endl << endl;
}

void profile5()
{
    map<minivec<long, 3>, int> map_array;
    cerr << "map first random:" << endl;
    timed_test3(map_array);
    cerr << endl << endl;
}

void profile6()
{
    typedef judy_multi_array<minivec<long, 3>, long, Minivec_Extractor, 3>
        array_type;
    array_type array;
    cerr << "judy_multi_array first random:" << endl;
    timed_test3(array);
    cerr << endl << endl;
}

int main(int argc, char ** argv)
try
{
    test1();
    profile1();
    profile2();
    profile3();
    profile4();
    profile5();
    profile6();
}
catch (const std::exception & exc) {
    cerr << "caught exception: " << exc.what() << endl;
    exit(1);
}
