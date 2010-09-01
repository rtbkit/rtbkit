/* judy_multi_array.h                                              -*- C++ -*-
   Jeremy Barnes, 21 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Multi array when the length of the key is known.
*/

#ifndef __boosting__judy_multi_array_h__
#define __boosting__judy_multi_array_h__


#include "judy_array.h"
#include <algorithm>


namespace ML {

template<typename Key, typename Data, class Extractor,
         size_t MaxLevel, size_t Level>
struct judy_multi_array_base;


/*****************************************************************************/
/* RECURSIVE MULTI_ARRAY_BASE                                                */
/*****************************************************************************/

/** This structure forms the recursive part of the judy_multi_array. */

template<typename Key, typename Data, class Extractor,
         size_t MaxLevel, size_t Level>
struct judy_multi_array_base {
    typedef judy_multi_array_base<Key, Data, Extractor, MaxLevel, Level + 1>
        next_type;

    typedef judyl_typed<next_type> array_type;
    array_type array;

    Data & get (const Key & key)
    {
        unsigned long key_section = Extractor().template get<Level-1>(key);
        return array[key_section].get(key);
    }

    const Data & get (const Key & key) const
    {
        unsigned long key_section = Extractor().template get<Level-1>(key);
        return array[key_section].get(key);
    }

    bool count(const Key & key) const
    {
        unsigned long key_section = Extractor().template get<Level-1>(key);
        if (array.count(key_section))
            return array[key_section].count(key);
        else return 0;
    }

    void clear()
    {
        array.clear();
    }

    size_t memusage_() const
    {
        return array.memusage_();
    }

    template<class Base, class Next, class QualData>
    struct iterator_base
        : public Base {
        iterator_base() {}

        template<class Base2>
        iterator_base(const Base2 & it)
            : Base(it)
        {
            if (!curr().end()) next() = curr()->begin();
        }

        QualData & operator * () const
        {
            return *next();
        }

        QualData * operator -> () const
        {
            return next().operator -> ();
        }

        iterator_base & operator ++ ()
        {
            if (curr().end()) throw Exception("increment past the end iterator");
            next().operator ++ ();
            while (!curr().end() && next().end()) {
                curr().operator ++ ();
                if (!curr().end())
                    next() = curr()->begin();
            }
            
            return *this;
        }

        iterator_base operator ++ (int)
        {
            iterator_base result = *this;
            operator ++ ();
            return result;
        }

        template <class Other>
        bool operator == (const Other & other) const
        {
            if (curr().end()) return other.curr().end();
            else if (other.curr().end()) return false;
            else return next() == other.next();
        }

        template<class Other>
        bool operator != (const Other & other) const
        {
            return ! operator == (other);
        }
        
        bool operator < (const iterator_base & other) const
        {
            if (end()) return !other.end();
            else if (other.end()) return true;
            else return next() < other.next();
        }

        Key key() const
        {
            if (curr().end())
                throw Exception("key(): past the end");
            Key result;
            assemble_key(result);
            return result;
        }

        void assemble_key(Key & key) const
        {
            //cerr << __PRETTY_FUNCTION__ << endl;
            Extractor().template put<Level-1>(key, curr().key());
            next().assemble_key(key);
        }

        Base & curr()
        {
            return *this;
        }
        
        const Base & curr() const
        {
            return *this;
        }
        
        Next & next()
        {
            return next_;
        }
        
        const Next & next() const
        {
            return next_;
        }

        Next next_;
    };

    typedef iterator_base<typename array_type::iterator,
                          typename next_type::iterator,
                          Data> iterator;
    typedef iterator_base<typename array_type::const_iterator,
                          typename next_type::const_iterator,
                          const Data> const_iterator;
    

    iterator begin() { return array.begin(); }
    iterator end() { return array.end(); }

    const_iterator begin() const { return array.begin(); }
    const_iterator end() const { return array.end(); }
};


/*****************************************************************************/
/* TERMINAL MULTI_ARRAY_BASE                                                 */
/*****************************************************************************/

/** This is the terminal multi array base.  It simply looks up the last element
    of the key in a judy array.
*/

template<typename Key, typename Data, class Extractor, size_t MaxLevel>
struct judy_multi_array_base<Key, Data, Extractor, MaxLevel, MaxLevel> {
    typedef judyl_typed<Data> array_type;
    array_type array;

    Data & get (const Key & key)
    {
        unsigned long key_section = Extractor().template get<MaxLevel-1>(key);
        return array[key_section];
    }

    const Data & get (const Key & key) const
    {
        unsigned long key_section = Extractor().template get<MaxLevel-1>(key);
        return array[key_section];
    }

    void clear()
    {
        array.clear();
    }

    bool count(const Key & key) const
    {
        unsigned long key_section = Extractor().template get<MaxLevel-1>(key);
        return array.count(key_section);
    }

    template<class Base>
    struct iterator_base
        : public Base {

        iterator_base()
        {
        }

        template<class Base2>
        iterator_base(const Base2 & it)
            : Base(it)
        {
        }

        Key key() const
        {
            if (end())
                throw Exception("key(): past the end");
            Key result;
            assemble_key(result);
            return result;
        }

        void assemble_key(Key & key) const
        {
            Extractor().template put<MaxLevel-1>(key, Base::key());
        }
    };
    
    typedef iterator_base<typename array_type::iterator> iterator;
    typedef iterator_base<typename array_type::const_iterator> const_iterator;

    iterator begin() { return array.begin(); }
    iterator end() { return array.end(); }

    const_iterator begin() const { return array.begin(); }
    const_iterator end() const { return array.end(); }

    size_t memusage_() const
    {
        return array.memusage_();
    }
};

template<typename Key, typename Data, class Extractor,
         size_t MaxLevel, size_t Level>
size_t memusage(const judy_multi_array_base<Key, Data, Extractor, MaxLevel, Level> & array)
{
    return array.memusage_();
}


/*****************************************************************************/
/* JUDY_MULTI_ARRAY                                                          */
/*****************************************************************************/

/** A multi-level nested array using the judy functions.  Allows for operator
    [] and an iterator interface.

    The Extractor object is needed to manipulate and interpret the key
    objects.
*/

template<typename Key, typename Data, class Extractor, size_t MaxLevel>
struct judy_multi_array {
    Data & operator [] (const Key & key)
    {
        return base.get(key);
    }
    
    const Data & operator [] (const Key & key) const
    {
        return base.get(key);
    }

    bool count(const Key & key) const
    {
        return base.count(key);
    }

    void clear()
    {
        base.clear();
    }

    size_t size() const { return std::distance(begin(), end()); }
    
    typedef judy_multi_array_base<Key, Data, Extractor, MaxLevel, 1> base_type;
    base_type base;

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;

    iterator begin() { return base.begin(); }
    iterator end() { return base.end(); }

    const_iterator begin() const { return base.begin(); }
    const_iterator end() const { return base.end(); }

    size_t memusage_() const
    {
        return base.memusage_();
    }
};

template<typename Key, typename Data, class Extractor, size_t MaxLevel>
size_t memusage(const judy_multi_array<Key, Data, Extractor, MaxLevel> & array)
{
    return array.memusage_();
}

} // namespace ML

#endif /* __boosting__judy_multi_array_h__ */
