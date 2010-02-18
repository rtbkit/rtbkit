/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Set class.
*/

%module jml 
%{
#include "jml/boosting/feature_set.h"
%}

%include "std_vector.i"

namespace ML {

class Training_Data;
class Feature_Set;
class Feature_Info;
class Feature;

/*****************************************************************************/
/* FEATURE_SET_CONST_ITERATOR                                                */
/*****************************************************************************/

struct Feature_Set_Const_Iterator {
    typedef std::random_access_iterator_tag iterator_category;
    typedef std::pair<Feature, float> value_type;
    typedef value_type & reference;
    typedef value_type * pointer;
    typedef ssize_t difference_type;

    Feature_Set_Const_Iterator();
    Feature_Set_Const_Iterator(const Feature * feat, const float * val,
                               int feat_stride, int val_stride);

    const Feature * feat;
    const float * val;
    int feat_stride;
    int val_stride;

    void advance(int num);

    Feature_Set_Const_Iterator & operator ++ ();
    Feature_Set_Const_Iterator operator ++ (int);
    Feature_Set_Const_Iterator & operator += (int num);
    Feature_Set_Const_Iterator operator + (int num) const;
    Feature_Set_Const_Iterator & operator -- ();
    Feature_Set_Const_Iterator operator -- (int);
    Feature_Set_Const_Iterator & operator -= (int num);
    Feature_Set_Const_Iterator operator - (int num) const;
    int operator - (const Feature_Set_Const_Iterator & other) const;

    std::pair<Feature, float> operator * () const;

    const Feature & feature() const;
    float value() const;

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
    Feature_Set();
    virtual ~Feature_Set();

    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const = 0;
    
    virtual size_t size() const;

    std::pair<Feature, float> operator [] (int index) const;

    float operator [] (const Feature & feature) const;

    float value(const Feature & feature) const;

    typedef Feature_Set_Const_Iterator const_iterator;

    const_iterator begin() const;

    const_iterator end() const;

    /** Number of times a given feature occurs in this feature set. */
    size_t count(const Feature & feat) const;

    /** Inquire whether a given feature occurs within the feature set. */
    bool contains(const Feature & feat) const;

    /** Returns a range of iterators for the given feature. */
    std::pair<const_iterator, const_iterator> find(const Feature & feat) const;

    /** Returns the first entry for a given feature. */
    const_iterator feature_begin(const Feature & feature) const;

    /** Returns the one-past-the-last entry for a given feature. */
    const_iterator feature_end(const Feature & feature) const;

    virtual void sort() = 0;

    int compare(const Feature_Set & other) const;

    int compare(const Feature_Set & other, const Feature & to_ignore) const;

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
    Mutable_Feature_Set();

    /** Construct from a range of pair<Feature, float>. */
    //template<class InputIterator>
    //Mutable_Feature_Set(const InputIterator & first,
    //                    const InputIterator & last);

    virtual ~Mutable_Feature_Set();

    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const;

    virtual void sort();

    //void sort() const;

    //void do_sort() const;

    typedef std::vector<std::pair<Feature, float> > features_type;
    mutable features_type features;
    mutable bool is_sorted;

    /** Add the given feature onto the end.  Will need to be sorted once the
        values are all done. */
    void add(const Feature & feat, float val = 1.0);

    /** Replace all instances of the feature with the new value.  May need to
        be sorted at the end. */
    void replace(const Feature & feat, float val = 1.0);

    void reserve(size_t num);

    void clear();

#if 0
    typedef features_type::iterator iterator;
    typedef features_type::const_iterator const_iterator;

    iterator begin() { return features.begin(); }
    iterator end() { return features.end(); }
    const_iterator begin() const { return features.begin(); }
    const_iterator end() const { return features.end(); }
#endif

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
};


} // namespace ML
