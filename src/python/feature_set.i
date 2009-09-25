/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Set class.
*/

%module jml 
%{
#include "boosting/features.h"
%}


namespace ML {

class Training_Data;
class Feature_Set;
class Feature_Info;

/*****************************************************************************/
/* FEATURE                                                                   */
/*****************************************************************************/

typedef int Feature_Id;

struct Feature {
    typedef Feature_Id id_type;
    
    Feature();
    
    explicit Feature(id_type type, id_type arg1 = 0, id_type arg2 = 0);

    id_type args_[3];

    id_type type() const;
    id_type & type();

    id_type arg1() const;
    id_type & arg1();

    id_type arg2() const;
    id_type & arg2();

    JML_ALWAYS_INLINE size_t hash() const;

    JML_ALWAYS_INLINE bool operator == (const Feature & other) const;

    JML_ALWAYS_INLINE bool operator != (const Feature & other) const;

    JML_ALWAYS_INLINE bool operator < (const Feature & other) const;

    std::string print() const;
};


/** This is the feature used when we want one we are sure will never be used
    as a real feature code. */
extern const Feature MISSING_FEATURE;



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
    Feature_Set();
    virtual ~Feature_Set();

    virtual boost::tuple<const Feature *, const float *, int, int, size_t>
    get_data(bool need_sorted = false) const = 0;
    
    virtual size_t size() const;

    std::pair<Feature, float> operator [] (int index) const;

    float operator [] (const Feature & feature) const;

    float value(const Feature & feature) const;

    struct const_iterator {
        typedef std::random_access_iterator_tag iterator_category;
        typedef std::pair<Feature, float> value_type;
        typedef value_type & reference;
        typedef value_type * pointer;
        typedef ssize_t difference_type;

        const_iterator() : feat(0), val(0), feat_stride(0), val_stride(0) {}
        const_iterator(const Feature * feat, const float * val,
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

        const_iterator & operator ++ ()
        {
            advance(1);
            return *this;
        }

        const_iterator operator ++ (int)
        {
            const_iterator result = *this;
            advance(1);
            return result;
        }
        
        const_iterator & operator += (int num)
        {
            advance(num);
            return *this;
        }
        
        const_iterator operator + (int num) const
        {
            const_iterator result(*this);
            result.advance(num);
            //using namespace std;
            //cerr << "opterator + (" << num << "): feat = " << feat
            //     << " result.feat = " << result.feat << endl;
            return result;
        }

        const_iterator & operator -- ()
        {
            advance(-1);
            return *this;
        }

        const_iterator operator -- (int)
        {
            const_iterator result = *this;
            advance(-1);
            return result;
        }
        
        const_iterator & operator -= (int num)
        {
            advance(-num);
            return *this;
        }
        
        const_iterator operator - (int num) const
        {
            const_iterator result(*this);
            result.advance(-num);
            return result;
        }

        int operator - (const const_iterator & other) const
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

        #define ITERATOR_COMP(op) \
        bool operator op (const const_iterator & other) const \
        { \
            return feat op other.feat; \
        }

        ITERATOR_COMP(==);
        ITERATOR_COMP(!=);
        ITERATOR_COMP(> );
        ITERATOR_COMP(< );
        ITERATOR_COMP(>=);
        ITERATOR_COMP(<=);
    };

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
