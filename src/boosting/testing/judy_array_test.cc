/* judy_array_test.cc
   Jeremy Barnes, 18 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Test of a judy array.
*/

#include "jml/boosting/judy_array.h"
#include <iostream>
#include "jml/utils/string_functions.h"
#include <vector>
#include <map>
#include <algorithm>
#include "jml/utils/hash_map.h"
#include <boost/timer.hpp>



using namespace ML;
using namespace std;


void test1()
{
    typedef judyl_base array_type;
    array_type array;
    cerr << "sizeof(array) = " << sizeof(array) << endl;
    
    cerr << "array.size() == " << array.size() << endl;
    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting..." << endl;
    for (unsigned i = 0;  i < 10;  ++i) {
        cerr << "memusage(array) = " << memusage(array) << endl;
        array[i] = i;
    }
    cerr << "done inserting" << endl;

    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;
    if (array.begin() == array.end())
        throw Exception("array begin and end are equal");


    cerr << "inserting again..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i] = i + 1;
    cerr << "done inserting" << endl;

    cerr << "array.size() = " << array.size() << endl;

    if (array.size() != 10)
        throw Exception("array size is incorrect");

    cerr << "memusage(array) = " << memusage(array) << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "iterating..." << endl;
    for (array_type::iterator it = array.begin();
         it != array.end();  ++it) {
        *it = 100 -*it;
    }
    cerr << "done iterating" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "copying..." << endl;
    array_type array2 = array;
    cerr << "done copying" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "erasing..." << endl;
    array2.erase(1);
    array2.erase(11);
    array2.erase(3);
    cerr << "done erasing" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
}

void test2()
{
    typedef judyl_typed<unsigned> array_type;
    array_type array;
    cerr << "sizeof(array) = " << sizeof(array) << endl;
     
    cerr << "array.size() == " << array.size() << endl;
    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i] = i;
    cerr << "done inserting" << endl;

    cerr << "inserting again..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i] = i + 1;
    cerr << "done inserting" << endl;

    cerr << "array.size() = " << array.size() << endl;

    cerr << "memusage(array) = " << memusage(array) << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "iterating..." << endl;
    for (array_type::iterator it = array.begin();
         it != array.end();  ++it) {
        *it = 100 -*it;
    }
    cerr << "done iterating" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "copying..." << endl;
    judyl_typed<unsigned> array2 = array;
    cerr << "done copying" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "erasing..." << endl;
    array2.erase(1);
    array2.erase(11);
    array2.erase(3);
    cerr << "done erasing" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
}

void test3()
{
    typedef judyl_typed<std::string> array_type;
    array_type array;
    cerr << "sizeof(array) = " << sizeof(array) << endl;

    cerr << "array.size() == " << array.size() << endl;
    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i] = format("string%d", i);
    cerr << "done inserting" << endl;

    cerr << "inserting again..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i] += " plus some more";
    cerr << "done inserting" << endl;
    
    cerr << "array.size() = " << array.size() << endl;
    
    cerr << "memusage(array) = " << memusage(array) << endl;
    
    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "iterating..." << endl;
    for (array_type::iterator it = array.begin();
         it != array.end();  ++it) {
        *it = uppercase(*it);
    }
    cerr << "done iterating" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "copying..." << endl;
    judyl_typed<std::string> array2 = array;
    cerr << "done copying" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "erasing..." << endl;
    array2.erase(1);
    array2.erase(11);
    array2.erase(3);
    cerr << "done erasing" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
}

ostream & operator << (ostream & stream, const vector<int> & vec)
{
    stream << "[ ";
    for (unsigned i = 0;  i < vec.size();  ++i)
        stream << vec[i] << " ";
    stream << "]";
    return stream;
}

void test4()
{
    typedef judyl_typed<std::vector<int> > array_type;
    array_type array;
    cerr << "sizeof(array) = " << sizeof(array) << endl;
    
    cerr << "array.size() == " << array.size() << endl;
    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i].push_back(i);
    cerr << "done inserting" << endl;

    cerr << "inserting again..." << endl;
    for (unsigned i = 0;  i < 10;  ++i)
        array[i].push_back(100 - i);
    cerr << "done inserting" << endl;
    
    cerr << "array.size() = " << array.size() << endl;
    
    cerr << "memusage(array) = " << memusage(array) << endl;
    
    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "iterating..." << endl;
    for (array_type::iterator it = array.begin();
         it != array.end();  ++it) {
        it->push_back(it->front());
    }
    cerr << "done iterating" << endl;
    
    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
    
    cerr << "copying..." << endl;
    array_type array2 = array;
    cerr << "done copying" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "erasing..." << endl;
    array2.erase(1);
    array2.erase(11);
    array2.erase(3);
    cerr << "done erasing" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
}

void test5()
{
    typedef judyl_base array_type;
    array_type array;
    cerr << "sizeof(array) = " << sizeof(array) << endl;
    
    cerr << "array.size() == " << array.size() << endl;
    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting..." << endl;
    for (unsigned long i = 0;  i < 10;  ++i)
        array[(unsigned long)-i] = i;
    cerr << "done inserting" << endl;

    cerr << "array.begin() == array.end() = " << (array.begin() == array.end())
         << endl;

    cerr << "inserting again..." << endl;
    for (unsigned long i = 0;  i < 10;  ++i)
        array[(unsigned long)-i] = i + 1;
    cerr << "done inserting" << endl;

    cerr << "array.size() = " << array.size() << endl;

    cerr << "memusage(array) = " << memusage(array) << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "iterating..." << endl;
    for (array_type::iterator it = array.begin();
         it != array.end();  ++it) {
        *it = 100 -*it;
    }
    cerr << "done iterating" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array.begin();
         it != array.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "copying..." << endl;
    array_type array2 = array;
    cerr << "done copying" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;

    cerr << "erasing..." << endl;
    array2.erase((unsigned long)-1L);
    array2.erase((unsigned long)-11L);
    array2.erase((unsigned long)-3L);
    cerr << "done erasing" << endl;

    cerr << "printing..." << endl;
    for (array_type::const_iterator it = array2.begin();
         it != array2.end();  ++it) {
        cerr << it.key() << " " << *it << endl;
    }
    cerr << "done printing" << endl;
}

template<class Array>
void do_timed_test(Array & array, vector<unsigned long> & indexes)
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
void timed_test1(Array & array, int NUM, const std::string & name)
{
    array.clear();

    /* Create the vectors. */
    vector<unsigned long> indexes(NUM);

    for (unsigned i = 0;  i < NUM;  ++i)
        indexes[i] = ((unsigned long)rand() << 32) + rand();
    
    cerr << "testing " << name << " with " << NUM << " entries" << endl;
    do_timed_test(array, indexes);
}

void profile1()
{
    typedef std::map<unsigned long, unsigned long> array_type;
    array_type array;

    timed_test1(array, 2000000, "map");
}

void profile2()
{
    typedef std::hash_map<unsigned long, unsigned long> array_type;
    array_type array;

    timed_test1(array, 2000000, "hash_map");
}

void profile3()
{
    typedef judyl_base array_type;
    array_type array;
    
    timed_test1(array, 2000000, "judyl_base");
}

void profile4()
{
    typedef std::hash_map<unsigned long, unsigned long> array_type;
    array_type array;

    timed_test1(array, 20000000, "hash_map");
}

void profile5()
{
    typedef judyl_base array_type;
    array_type array;
    
    timed_test1(array, 20000000, "judyl_base");
}

int main(int argc, char ** argv)
try
{
    test1();
    test2();
    test3();
    test4();
    test5();
    profile1();
    profile2();
    profile3();
    //profile4();
    //profile5();
}
catch (const std::exception & exc) {
    cerr << "caught exception: " << exc.what() << endl;
    exit(1);
}
