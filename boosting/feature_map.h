/* feature_map.h                                                   -*- C++ -*-
   Jeremy Barnes, 18 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   A map-like class used to hold information about a bunch of features.

   Uses the judy array internally.
*/

#ifndef __boosting__feature_map_h__
#define __boosting__feature_map_h__


#include "judy_multi_array.h"
#include "jml/boosting/feature_set.h"
#include "jml/utils/hash_map.h"


namespace ML {

#if 1

#if defined(__amd64)

struct Feature_Extractor {

    enum {
        DEPTH = 2
    };

    // Allow different rules to access each element
    template<size_t idx>
    struct access {
    };

    template<size_t Idx>
    static unsigned long get(const Feature & feat)
    {
        return access<Idx>::get(feat);
    }

    template<size_t Idx>
    static void put(Feature & feat, unsigned long val)
    {
        access<Idx>::set(feat, val);
    }
};

template<>
struct Feature_Extractor::access<0> {
    static void set(Feature & feat, unsigned long val)
    {
        feat.args_[1] = (val >> 32UL);
        feat.args_[2] = val;
    }
    
    static const unsigned long get(const Feature & feat)
    {
        return (unsigned long)feat.args_[1] << 32UL | feat.args_[2];
    }
};

template<>
struct Feature_Extractor::access<1> {
    static void set(Feature & feat, unsigned long val)
    {
        feat.args_[0] = val;
    }
    
    static const unsigned long get(const Feature & feat)
    {
        return feat.args_[0];
    }
};
    

#else

struct Feature_Extractor {
    enum {
        DEPTH = 3
    };

    // Allow different rules to access each element
    template<size_t idx>
    struct access {
    };

    template<size_t Idx>
    static unsigned long get(const Feature & feat)
    {
        return access<Idx>::get(feat);
    }
    
    template<size_t Idx>
    static void put(Feature & feat, unsigned long val)
    {
        access<Idx>::set(feat, val);
    }
};

template<>
struct Feature_Extractor::access<0> {
    static void set(Feature & feat, unsigned long val)
    {
        feat.set_arg2(val);
    }
    
    static const unsigned long get(const Feature & feat)
    {
        return feat.arg2();
    }
};

template<>
struct Feature_Extractor::access<1> {
    static void set(Feature & feat, unsigned long val)
    {
        feat.set_arg1(val);
    }
    
    static const unsigned long get(const Feature & feat)
    {
        return feat.arg1();
    }
};

template<>
struct Feature_Extractor::access<2> {
    static void set(Feature & feat, unsigned long val)
    {
        feat.set_type(val);
    }
    
    static const unsigned long get(const Feature & feat)
    {
        return feat.type();
    }
};
    

#endif /* 32/64 bit */

template<class Data>
class Feature_Map
    : public judy_multi_array<Feature, Data, Feature_Extractor,
                              Feature_Extractor::DEPTH> {
};

template<class Data>
size_t memusage(const Feature_Map<Data> & fm)
{
    return fm.memusage_();
}

#else

template<class Data>
class Feature_Map
    : public std::hash_map<Feature, Data> {
public:    
    typedef std::hash_map<Feature, Data> base_type;
    typedef typename base_type::iterator base_iterator;
    typedef typename base_type::const_iterator base_const_iterator;

    using base_type::operator [];

    const Data & operator [] (const Feature & feature) const
    {
        static const Data NONE;
        const_iterator it = this->find(feature);
        if (it == this->end())
            return NONE;
        else return it->second;
    }

    struct iterator : public base_iterator {
        iterator() {}
        
        iterator(const base_iterator & it)
            : base_iterator(it)
        {
        }

        const Feature & key() const
        {
            return base_iterator::operator -> ()->first;
        }

        Data & operator * () const
        {
            return base_iterator::operator -> ()->second;
        }

        Data * operator -> () const
        {
            return &base_iterator::operator -> ()->second;
        }
    };

    struct const_iterator : public base_const_iterator {
        const_iterator() {}

        const_iterator(const base_const_iterator & it)
            : base_const_iterator(it)
        {
        }

        const Feature & key() const
        {
            return base_const_iterator::operator -> ()->first;
        }

        const Data & operator * () const
        {
            return base_const_iterator::operator -> ()->second;
        }

        const Data * operator -> () const
        {
            return &base_const_iterator::operator -> ()->second;
        }
    };
};

#endif

} // namespace ML

#endif /* __boosting__feature_map_h__ */
