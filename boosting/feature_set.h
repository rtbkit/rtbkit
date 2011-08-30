/* feature_set.h                                                   -*- C++ -*-
   Jeremy Barnes, 10 May 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A feature set.
*/

#ifndef __boosting__feature_set_h__
#define __boosting__feature_set_h__


#include "config.h"
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/tuple/tuple_io.hpp>
#include "jml/db/persistent.h"
#include "jml/stats/distribution.h"
#include <vector>
#include "jml/utils/floating_point.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/sgi_algorithm.h"
#include "jml/utils/hash_specializations.h"
#include <cmath>
#include "jml/utils/parse_context.h"
#include "feature.h"

namespace ML {

class Training_Data;
class Feature_Set;
class Feature_Info;


/*****************************************************************************/
/* FEATURE_SET_CONST_ITERATOR                                                */
/*****************************************************************************/

struct Feature_Set_Const_Iterator {
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::pair<Feature, float> value_type;
    typedef value_type & reference;
    typedef value_type * pointer;
    typedef ssize_t difference_type;

    Feature_Set_Const_Iterator() : feat(0), val(0), feat_stride(0), val_stride(0) {}
    Feature_Set_Const_Iterator(const Feature * feat, const float * val,
                   int feat_stride, int val_stride)
        : feat(feat), val(val),
          feat_stride(feat_stride), val_stride(val_stride) {}

    const Feature * feat;
    const float * val;
    int feat_stride;
    int val_stride;

    void advance(int num)
    {
        size_t i_f = (size_t)feat;
        size_t i_v = (size_t)val;
        i_f += feat_stride * num;
        i_v += val_stride * num;
        feat = (const Feature *)i_f;
        val = (const float *)i_v;
    }

    Feature_Set_Const_Iterator & operator ++ ()
    {
        advance(1);
        return *this;
    }

    Feature_Set_Const_Iterator operator ++ (int)
    {
        Feature_Set_Const_Iterator result = *this;
        advance(1);
        return result;
    }
        
    Feature_Set_Const_Iterator & operator += (int num)
    {
        advance(num);
        return *this;
    }
        
    Feature_Set_Const_Iterator operator + (int num) const
    {
        Feature_Set_Const_Iterator result(*this);
        result.advance(num);
        //using namespace std;
        //cerr << "opterator + (" << num << "): feat = " << feat
        //     << " result.feat = " << result.feat << endl;
        return result;
    }

    Feature_Set_Const_Iterator & operator -- ()
    {
        advance(-1);
        return *this;
    }

    Feature_Set_Const_Iterator operator -- (int)
    {
        Feature_Set_Const_Iterator result = *this;
        advance(-1);
        return result;
    }
        
    Feature_Set_Const_Iterator & operator -= (int num)
    {
        advance(-num);
        return *this;
    }
        
    Feature_Set_Const_Iterator operator - (int num) const
    {
        Feature_Set_Const_Iterator result(*this);
        result.advance(-num);
        return result;
    }

    int operator - (const Feature_Set_Const_Iterator & other) const
    {
        size_t i1 = (size_t)feat;
        size_t i2 = (size_t)other.feat;
        return (i1 - i2) / feat_stride;
    }

    std::pair<Feature, float> operator * () const
    {
        return std::make_pair(*feat, *val);
    }

    const Feature & feature() const { return *feat; }
    float value() const { return *val; }

#define ITERATOR_COMP(op)                                       \
    bool operator op (const Feature_Set_Const_Iterator & other) const       \
    {                                                           \
        return feat op other.feat;                              \
    }

    ITERATOR_COMP(==);
    ITERATOR_COMP(!=);
    ITERATOR_COMP(> );
    ITERATOR_COMP(< );
    ITERATOR_COMP(>=);
    ITERATOR_COMP(<=);
#undef ITERATOR_COMP
};



/*****************************************************************************/
/* FEATURE_SET                                                               */
/*****************************************************************************/

/** This provides a set of features.  It is used as the interface between
    the learning algorithms and the feature extractors.
*/

class Feature_Set {
public:
    Feature_Set() {}
    virtual ~Feature_Set() {}

    /** Main virtual method.  Returns the information necessary to access the
        data.

        \param need_sorted      if true, the arrays must be sorted before
                                returning them.

        Return values:

        - Feature *:       pointer to the first feature
        - float *:         pointer to the first value
        - int:             bytes to advance Feature * to get to the next one
        - int:             bytes to advance value * to get to the next one
        - size_t:          the number of values in the array
    */
    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const = 0;
    
    virtual size_t size() const
    {
        return get_data(false).get<4>();
    }

    std::pair<Feature, float> at(int index) const
    {
        return operator [] (index);
    }

    std::pair<Feature, float> operator [] (int index) const
    {
        return *(begin() + index);
    }

    /** Return the value associated with the feature.  Will throw an exception
        if it is missing or if there are multiple values. */
    float operator [] (const Feature & feature) const
    {
        const_iterator it = feature_begin(feature);
        if (it == end() || (*it).first != feature)
            throw Exception("Feature_Set::operator []: feature not found");
        else {
            const_iterator next = it;  ++next;
            if (next != end() && (*next).first == feature)
                throw Exception("Feature_Set::operator []: feature occurs more "
                                "than once");
        }
        return (*it).second;
    }

    float value(const Feature & feature) const
    {
        return operator [] (feature);
    }

    typedef Feature_Set_Const_Iterator const_iterator;

    const_iterator begin() const
    {
        const Feature * feat;
        const float * val;
        int feat_stride;
        int val_stride;
        size_t size;
        boost::tie(feat, val, feat_stride, val_stride, size) = get_data(true);
        return const_iterator(feat, val, feat_stride, val_stride);
    }

    const_iterator end() const
    {
        const Feature * feat;
        const float * val;
        int feat_stride;
        int val_stride;
        size_t size;
        boost::tie(feat, val, feat_stride, val_stride, size) = get_data(true);
        return const_iterator(feat, val, feat_stride, val_stride) + size;
    }    

    /** Number of times a given feature occurs in this feature set. */
    size_t count(const Feature & feat) const
    {
        return feature_end(feat) - feature_begin(feat);
    }

    /** Inquire whether a given feature occurs within the feature set. */
    bool contains(const Feature & feat) const
    {
        const_iterator loc = feature_begin(feat);
        return loc != end() && (*loc).first == feat;
    }

    /** Returns a range of iterators for the given feature. */
    std::pair<const_iterator, const_iterator> find(const Feature & feat) const
    {
        return std::make_pair(feature_begin(feat), feature_end(feat));
    }

    /** Returns the first entry for a given feature. */
    const_iterator feature_begin(const Feature & feature) const
    {
        using namespace std;
        return lower_bound(begin(), end(), make_pair(feature, -INFINITY));
    }

    /** Returns the one-past-the-last entry for a given feature. */
    const_iterator feature_end(const Feature & feature) const
    {
        using namespace std;
        return upper_bound(begin(), end(), make_pair(feature, INFINITY));
    }

    virtual void sort() = 0;

    int compare(const Feature_Set & other) const
    {
        return std::lexicographical_compare_3way(begin(), end(),
                                                 other.begin(), other.end());
    }

    /** Compare the two feature sets, ignoring the values or presence of the
        given feature.

        Equivalent to filtering the given feature out of both feature sets
        and then calling the normal compare() routine.
    */
    int compare(const Feature_Set & other, const Feature & to_ignore) const
    {
        const_iterator it1 = begin(), e1 = end();
        const_iterator it2 = other.begin(), e2 = other.end();

        while (it1 != e1 && it2 != e2) {
            /* Skip any that match the ignored feature. */
            if ((*it1).first == to_ignore) { ++it1;  continue; }
            if ((*it2).first == to_ignore) { ++it2;  continue; }
            
            if (*it1 < *it2) return -1;
            else if (*it2 < *it1) return 1;
            
            ++it1;
            ++it2;
        }

        /* Skip any examples at the end. */
        while (it1 != e1 && (*it1).first == to_ignore) ++it1;
        while (it2 != e2 && (*it2).first == to_ignore) ++it2;
        
        if (it2 == e2) return it1 != e1;
        
        return -1;
    }

    virtual Feature_Set * make_copy() const = 0;
};


/*****************************************************************************/
/* MUTABLE_FEATURE_SET                                                       */
/*****************************************************************************/

/** This provides a set of features.  It is used as the interface between
    the learning algorithms and the feature extractors.
*/

class Mutable_Feature_Set : public Feature_Set {
public:
    Mutable_Feature_Set() : is_sorted(true), locked(false) {}

    /** Construct from a range of pair<Feature, float>. */
    template<class InputIterator>
    Mutable_Feature_Set(const InputIterator & first,
                        const InputIterator & last)
        : features(first, last),
          is_sorted(false), locked(false)//is_sorted(std::is_sorted(first, last))
    {
        do_sort();
    }

    virtual ~Mutable_Feature_Set() {}

    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const
    {
        if (need_sorted && !is_sorted) sort();

        return boost::make_tuple
            (&features[0].first, &features[0].second,
             sizeof(std::pair<Feature, float>),
             sizeof(std::pair<Feature, float>),
             features.size());
    }

    virtual void sort();

    void sort() const
    {
        if (is_sorted) return;
        do_sort();
    }

    void do_sort() const;

    typedef std::vector<std::pair<Feature, float> > features_type;
    mutable features_type features;
    mutable bool is_sorted;
    bool locked;

    /** Add the given feature onto the end.  Will need to be sorted once the
        values are all done. */
    void add(const Feature & feat, float val = 1.0);

    /** Replace all instances of the feature with the new value.  May need to
        be sorted at the end. */
    void replace(const Feature & feat, float val = 1.0);

    void reserve(size_t num)
    {
        if (locked) throw Exception("mutating locked feature set");
        features.reserve(num);
    }

    void clear()
    {
        if (locked) throw Exception("mutating locked feature set");
        features.clear();
        is_sorted = true;
    }

    typedef features_type::iterator iterator;
    typedef features_type::const_iterator const_iterator;

    iterator begin() { return features.begin(); }
    iterator end() { return features.end(); }
    const_iterator begin() const { return features.begin(); }
    const_iterator end() const { return features.end(); }

    features_type::value_type & at(int index) { return features.at(index); }
    const features_type::value_type & at(int index) const
    {
        return features.at(index);
    }

    using Feature_Set::operator [];

    features_type::value_type & operator [] (int index)
    { 
        return features[index];
    }

    const features_type::value_type & operator [] (int index) const
    {
        return features[index];
    }

    virtual Mutable_Feature_Set * make_copy() const;

private:
    struct compare_feature;
};


/*****************************************************************************/
/* MISCELLANEOUS                                                             */
/*****************************************************************************/

/** This function escapes the name of a feature before being written to a
    file, to allow ones with special characters to be used.

    \param feature      the name of the feature to escape
    \returns            the escaped feature.  This will be either enclosed in
                        quotes, have backslashes added, or be unchanged
                        depending upon which is the shortest way to write it.
*/
    
std::string escape_feature_name(const std::string & feature);

/** Parse a feature name escaped by the escape_feature_name function. */
std::string expect_feature_name(Parse_Context & c);

} // namespace ML


namespace JML_HASH_NS {

template<>
struct hash<ML::Feature> {
    size_t operator () (const ML::Feature & f) const
    {
        return f.hash();
    }
};

} // namespace JML_HASH_NS


#endif /* __boosting__feature_set_h__ */
