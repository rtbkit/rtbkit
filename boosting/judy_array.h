/* judy_array.h                                                    -*- C++ -*-
   Jeremy Barnes, 18 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Judy array interface functions.  Makes it look more like a map.
*/

#ifndef __boosting__judy_array_h__
#define __boosting__judy_array_h__

#include "config.h"
#include "jml/judy/Judy.h"
#include <utility>
#include "jml/arch/exception.h"
#include <iostream>
#include <boost/type_traits.hpp>
#include "memusage.h"

#if 0
extern PPvoid_t j__udyLGet(      Pvoid_t   Pjpm,   Word_t    Index);
extern PPvoid_t JudyLGet(        Pcvoid_t  PArray, Word_t    Index,  P_JE);
extern PPvoid_t JudyLIns(        PPvoid_t PPArray, Word_t    Index,  P_JE);
extern int      JudyLInsArray(   PPvoid_t PPArray, Word_t    Count,
                                             const Word_t * const PIndex,
                                             const Word_t * const PValue,
                                                                     P_JE);
extern int      JudyLDel(        PPvoid_t PPArray, Word_t    Index,  P_JE);
extern Word_t   JudyLCount(      Pcvoid_t  PArray, Word_t    Index1,
                                                   Word_t    Index2, P_JE);
extern PPvoid_t JudyLByCount(    Pcvoid_t  PArray, Word_t    Count,
                                                   Word_t *  PIndex, P_JE);
extern Word_t   JudyLFreeArray(  PPvoid_t PPArray,                   P_JE);
extern Word_t   JudyLMemUsed(    Pcvoid_t  PArray);
extern Word_t   JudyLMemActive(  Pcvoid_t  PArray);
extern PPvoid_t JudyLFirst(      Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern PPvoid_t JudyLNext(       Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern PPvoid_t j__udyLNext(     Pvoid_t   Pjpm,   Word_t * PIndex);
extern PPvoid_t JudyLLast(       Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern PPvoid_t JudyLPrev(       Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern int      JudyLFirstEmpty( Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern int      JudyLNextEmpty(  Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern int      JudyLLastEmpty(  Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
extern int      JudyLPrevEmpty(  Pcvoid_t  PArray, Word_t * PIndex,  P_JE);
#endif

extern const uint8_t j__L_LeafWOffset[];

#undef JLG
#undef JLN


namespace ML {

using namespace std;

template<class Base, class Value>
struct judy_iterator : public Base {
    typedef std::bidirectional_iterator_tag iterator_category;
    typedef ptrdiff_t difference_type;
    typedef Value * pointer;
    typedef Value & reference;
    typedef Base base_type;
    typedef Value value_type;

    using Base::deref;

    judy_iterator()
    {
    }

    judy_iterator(const base_type & other)
        : base_type(other)
    {
    }
    
    judy_iterator & operator ++ ()
    {
        base_type::operator ++();
            return *this;
    }
    
    judy_iterator operator ++ (int)
    {
        judy_iterator result = *this;
        operator ++ ();
        return result;
    }

    judy_iterator & operator -- ()
    {
        base_type::operator --();
        return *this;
    }
    
    judy_iterator operator -- (int)
    {
        judy_iterator result = *this;
        operator -- ();
        return result;
    }
    
    value_type & operator * () const
    {
        return *deref();
    }
    
    value_type * operator -> () const
    {
        return deref();
    }
};

template<class Base, class Data>
struct judy_pair_iterator : public Base {
    typedef Base base_type;
    typedef typename base_type::key_type key_type;
    typedef std::pair<key_type, Data> value_type;

    using Base::deref;
    using Base::key_;
    
    judy_pair_iterator(const base_type & other)
        : base_type(other)
    {
    }
    
    judy_pair_iterator & operator ++ ()
    {
        base_type::operator ++();
        return *this;
    }
    
    judy_pair_iterator operator ++ (int)
    {
        judy_pair_iterator result = *this;
        operator ++ ();
        return result;
    }
    
    judy_pair_iterator & operator -- ()
    {
        base_type::operator --();
        return *this;
    }
    
    judy_pair_iterator operator -- (int)
    {
        judy_pair_iterator result = *this;
        operator -- ();
        return result;
    }
    
    value_type operator * () const
    {
        return std::make_pair(key_, *deref());
    }
    
private:
    value_type * operator -> () const;  // can't access
};


struct judyl_base {
    typedef unsigned long key_type;
    typedef unsigned long data_type;
    typedef std::pair<key_type, data_type> value_type;

    judyl_base()
        : array(0)
    {
    }

    judyl_base(const judyl_base & other)
        : array(0)
    {
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
    }

    judyl_base & operator = (const judyl_base & other)
    {
        if (&other == this) return *this;
        clear();
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
        return *this;
    }

    template<class Iterator>
    judyl_base(Iterator first, Iterator last)
    {
        insert(first, last);
    }

    ~judyl_base()
    { 
        clear();
    }

    struct iterator_base {
        typedef unsigned long key_type;

        /** Empty constructor; to be assigned to later. */
        iterator_base() {}
        
        /** Construct pointing to the given key */
        iterator_base(const judyl_base * base, key_type key)
            : base(base), key_(key), end_(false)
        {
            data_type * pvalue
                = (data_type *)JudyLFirst(base->array, &key_, PJE0);
            if (!pvalue) end_ = true;
        }

        /** Construct at the end() */
        iterator_base(const judyl_base * base)
            : base(base), key_(0), end_(true)
        {
        }

        typedef data_type value_type;

        const judyl_base * base;
        key_type key_;
        bool end_;
        
        iterator_base & operator ++ ()
        {
            if (end_)
                throw Exception("judyl_base::iterator_base::operator ++(): "
                                " incremented end() iterator");
            data_type * pvalue = base->jln(key_);
            //data_type * pvalue
            //    = (data_type *)JudyLNext(base->array, &key_, PJE0);
            if (!pvalue) end_ = true;
            return *this;
        }
        
        iterator_base operator ++ (int)
        {
            iterator_base result = *this;
            operator ++ ();
            return result;
        }

        iterator_base & operator -- ()
        {
            if (end_) {
                /* Handle special case of finding -1 iterator. */
                key_ = (key_type)-1;
                data_type * pvalue
                    = (data_type *)JudyLLast(base->array, &key_, PJE0);
                end_ = false;
                if (pvalue) return *this;
            }
            data_type * pvalue
                = (data_type *)JudyLPrev(base->array, &key_, PJE0);
            if (pvalue == 0) throw Exception("iterated off front");
            return *this;
        }
        
        iterator_base operator -- (int)
        {
            iterator_base result = *this;
            operator -- ();
            return result;
        }

        key_type key() const { return key_; }

        bool operator == (const iterator_base & other) const
        {
            if (end_) return other.end_;
            else if (other.end_) return false;
            else return key_ == other.key_;
        }

        bool operator != (const iterator_base & other) const
        {
            return ! operator == (other);
        }

        bool operator < (const iterator_base & other) const
        {
            if (end_) return !other.end_;
            else if (other.end_) return false;
            else return key_ < other.key_;
        }

        bool end() const { return end_; }

    protected:
        data_type * deref() const
        {
            if (end_)
                throw Exception("iterator_base: dereferencing off the end");

            //data_type * pvalue = (value_type *)JudyLGet(base->array, key_, PJE0);
            data_type * pvalue = base->jlg(key_);
            
            if (!pvalue) throw Exception("iterator_base: dereferencing deleted "
                                         "key");
            return pvalue;
        }
    };
    
    typedef judy_iterator<iterator_base, unsigned long> iterator;
    typedef judy_iterator<iterator_base, const unsigned long> const_iterator;
    typedef judy_pair_iterator<iterator_base, unsigned long> const_pair_iterator;

    iterator begin()
    {
        return iterator_base(this, 0);
    }

    iterator end()
    {
        return iterator_base(this);
    }

    const_iterator begin() const
    {
        return iterator_base(this, 0);
    }

    const_iterator end() const
    {
        return iterator_base(this);
    }

    size_t size() const
    {
        return JudyLCount(array, 0, (key_type)-1, PJE0);
    }

    size_t max_size() const
    {
        return (size_t)-1;
    }

    bool empty() const
    {
        key_type index = 0;
        data_type * pvalue = (data_type *)JudyLFirst(array, &index, PJE0);
        return pvalue;
    }

    void swap(judyl_base & other)
    {
        std::swap(array, other.array);
    }

    std::pair<iterator, bool> insert(const value_type & x)
    {
        data_type * pvalue = jlg(x.first);
        //data_type * pvalue = (data_type *)JudyLGet(array, x.first, PJE0);
        bool inserted = false;
        if (!pvalue) {
            pvalue = (data_type *)JudyLIns(&array, x.first, PJE0);
            *pvalue = x.second;
            inserted = true;
        }

        return std::make_pair(iterator_base(this, x.first), inserted);
    }

    template<class Iterator>
    void insert(Iterator first, Iterator last)
    {
        while (first != last) insert(*first++);
    }

    void erase(iterator_base pos)
    {
        erase(pos.key());
    }

    size_t erase(const key_type & key)
    {
        int res = JudyLDel(&array, key, PJE0);
        if ((void *)(size_t)res == PJERR) throw std::bad_alloc();
        return res;
    }

    void erase(iterator_base first, iterator_base last)
    {
        while (first != last) erase(first++.key());
    }

    size_t count(const key_type & key) const
    {
        data_type * pvalue = jlg(key);
        //value_type * pvalue
        //    = (value_type *)JudyLGet(array, key, PJE0);
        return (bool)pvalue;
    }

    iterator find(const key_type & key)
    {
        data_type * pvalue = jlg(key);
        //value_type * pvalue
        //    = (value_type *)JudyLGet(array, key, PJE0);
        if (pvalue) return iterator_base(this, key);
        else return end();
    }
    
    const_iterator find(const key_type & key) const
    {
        data_type * pvalue = jlg(key);
        //value_type * pvalue
        //    = (value_type *)JudyLGet(array, key, PJE0);
        if (pvalue) return iterator_base(this, key);
        else return end();
    }

    void clear()
    {
        JudyLFreeArray(&array, PJE0);
        array = 0;
    }

    data_type & operator [] (const key_type & key)
    {
        data_type * pvalue = jlg(key);
        //data_type * pvalue = (data_type *)JudyLGet(array, key, PJE0);
        if (!pvalue)
            pvalue = (data_type *)JudyLIns(&array, key, PJE0);
        return *pvalue;
    }

    data_type operator [] (const key_type & key) const
    {
        data_type * pvalue = jlg(key);
        //data_type * pvalue = (data_type *)JudyLGet(array, key, PJE0);
        if (pvalue) return *pvalue;
        else return data_type();
    }

    size_t memusage_() const
    {
        return JudyLMemUsed(array);
    }

    void * array;  ///< Array pointing to the judy value

    data_type * jlg(const key_type & key) const
    {
        PWord_t P_L  = reinterpret_cast<PWord_t>(array);
        data_type * result = 0;
        if (P_L) {
            if (P_L[0] < 31) {
                Word_t  _pop1 = P_L[0] + 1;                         
                Word_t  _EIndex = P_L[_pop1];                       
                Word_t  _off  = j__L_LeafWOffset[_pop1] - 1;        
                if (_pop1 >= 16)
                    if (key > P_L[_pop1/2]) P_L += _pop1/2;     
                if (key <= _EIndex) {
                    while(key > *(++P_L));                      
                    if (*P_L == key)
                        result = reinterpret_cast<data_type *>(P_L+_off);
                }  
            }                                               
            else {                                                   
                result = reinterpret_cast<data_type *>
                    (j__udyLGet(P_L, key));    
            }                                            
        }                                                           

        return result;
    }

    data_type * jln(key_type & key) const
    {
        PWord_t P_L  = reinterpret_cast<PWord_t>(array);
        data_type * result = 0;
                                                                
        if (P_L) {
            if (P_L[0] < 31) {
                Word_t _pop1 = P_L[0] + 1;                          
                Word_t _off  = j__L_LeafWOffset[_pop1] -1;          
                if (key < P_L[_pop1]) {
                    while(1) {
                        if (key < *(++P_L)) {
                            key = *P_L;                         
                            result = reinterpret_cast<data_type *>(P_L + _off);
                            break;                                
                        }
                    }
                }
            }
            else 
                result = reinterpret_cast<data_type *>
                    (JudyLNext(array, &key, PJE0)); 
        }

        return result;
    }                                                       
};

inline size_t memusage(const judyl_base & array)
{
    return array.memusage_();
}

template<typename Data,
         bool Contained = (sizeof(Data) <= sizeof(void *))>
class judyl_typed;

template<typename Data>
class judyl_typed<Data, true>
    : public judyl_base {
public:
    typedef unsigned long key_type;
    typedef Data data_type;
    typedef std::pair<key_type, Data> value_type;

    judyl_typed()
    {
    }

    judyl_typed(const judyl_typed & other)
    {
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
    }

    judyl_typed & operator = (const judyl_typed & other)
    {
        if (&other == this) return;
        clear();
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
        return *this;
    }
    
    template<class Iterator>
    judyl_typed(Iterator first, Iterator last)
    {
        insert(first, last);
    }

    ~judyl_typed()
    { 
        clear();
    }

    struct iterator_base : public judyl_base::iterator_base {
        typedef judyl_base::iterator_base base_type;

        iterator_base() {}

        iterator_base(const base_type & base)
            : base_type(base)
        {
        }

        iterator_base(const judyl_base * base, key_type key)
            : base_type(base, key)
        {
        }

        iterator_base(const judyl_base * base)
            : base_type(base)
        {
        }

        typedef data_type value_type;

        iterator_base & operator ++ ()
        {
            base_type::operator ++ ();
            return *this;
        }

        iterator_base operator ++ (int)
        {
            iterator_base result = *this;
            operator ++ ();
            return result;
        }

        iterator_base & operator -- ()
        {
            base_type::operator -- ();
            return *this;
        }

        iterator_base operator -- (int)
        {
            iterator_base result = *this;
            operator -- ();
            return result;
        }

    protected:
        data_type * deref() const
        {
            return (data_type *)base_type::deref();
        }
    };

    typedef judy_iterator<iterator_base, data_type> iterator;
    typedef judy_iterator<iterator_base, const data_type> const_iterator;
    typedef judy_pair_iterator<iterator_base, data_type> const_pair_iterator;

    iterator begin()
    {
        return iterator_base(this, 0);
    }

    iterator end()
    {
        return iterator_base(this);
    }

    const_iterator begin() const
    {
        return iterator_base(this, 0);
    }

    const_iterator end() const
    {
        return iterator_base(this);
    }

    std::pair<iterator, bool> insert(const value_type & x)
    {
        data_type * pvalue = reinterpret_cast<data_type *>(jlg(x.first));
        //data_type * pvalue = (data_type *)JudyLGet(array, x.first, PJE0);
        bool inserted = false;
        if (!pvalue) {
            pvalue = (data_type *)JudyLIns(&array, x.first, PJE0);
            ::new(static_cast<void *>(pvalue)) data_type(x.second);
            inserted = true;
        }

        return std::make_pair(iterator_base(this, x.first),
                              inserted);
    }

    template<class Iterator>
    void insert(Iterator first, Iterator last)
    {
        while (first != last) insert(*first++);
    }

    void erase(iterator_base pos)
    {
        erase(pos.key());
    }

    size_t erase(const key_type & key)
    {
        /* Need to find it first to call the destructor. */
        data_type * pvalue = reinterpret_cast<data_type *>(jlg(key));
        //data_type * pvalue = (data_type *)JudyLGet(array, key, PJE0);
        if (pvalue) {
            pvalue->~data_type();
            int res = JudyLDel(&array, key, PJE0);
            if ((void *)(size_t)res == PJERR) throw std::bad_alloc();
            return res;
        }
        else return 0;
    }

    void erase(iterator_base first, iterator_base last)
    {
        while (first != last) erase(first++.key());
    }

    iterator find(const key_type & key)
    {
        return judyl_base::find(key);
    }
    
    const_iterator find(const key_type & key) const
    {
        return judyl_base::find(key);
    }
    
    void clear()
    {
        /* Need to iterate over to run destructors. */
        for (iterator it = begin();  it != end();  ++it)
            it->~data_type();
        
        judyl_base::clear();
    }

    data_type & operator [] (const key_type & key)
    {
        data_type * pvalue = reinterpret_cast<data_type *>(jlg(key));
        //data_type * pvalue = (data_type *)JudyLGet(array, key, PJE0);
        if (!pvalue) {
            pvalue = (data_type *)JudyLIns(&array, key, PJE0);
            ::new(static_cast<void *>(pvalue)) data_type();
        }
        return *pvalue;
    }

    data_type operator [] (const key_type & key) const
    {
        data_type * pvalue = reinterpret_cast<data_type *>(jlg(key));
        //data_type * pvalue = (data_type *)JudyLGet(array, key, PJE0);
        if (pvalue) return *pvalue;
        else return data_type();
    }

    size_t memusage_() const
    {
        if (mem_traits<Data>::is_fixed)
            return size() * sizeof(Data) + judyl_base::memusage_();
        else {
            size_t result = judyl_base::memusage_();
            for (const_iterator it = begin();  it != end();  ++it)
                result += memusage(*it);
            return result;
        }
    }
};

template<typename Data>
class judyl_typed<Data, false>
    : public judyl_typed<Data *, true> {
public:
    typedef unsigned long key_type;
    typedef Data data_type;
    typedef std::pair<key_type, Data> value_type;
    typedef judyl_typed<Data *, true> base_type;

    using base_type::array;
    using base_type::size;

    judyl_typed()
    {
    }

    judyl_typed(const judyl_typed & other)
    {
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
    }

    judyl_typed & operator = (const judyl_typed & other)
    {
        if (&other == this) return;
        clear();
        insert(const_pair_iterator(other.begin()),
               const_pair_iterator(other.end()));
        return *this;
    }
    
    template<class Iterator>
    judyl_typed(Iterator first, Iterator last)
    {
        insert(first, last);
    }

    ~judyl_typed()
    { 
        clear();
    }

    struct iterator_base : public base_type::iterator_base {
        typedef typename base_type::iterator_base base_it_type;

        iterator_base() {}

        iterator_base(const base_it_type & base)
            : base_it_type(base)
        {
        }

        iterator_base(const judyl_base * base, key_type key)
            : base_it_type(base, key)
        {
        }

        iterator_base(const judyl_base * base)
            : base_it_type(base)
        {
        }

        typedef data_type value_type;

        iterator_base & operator ++ ()
        {
            base_it_type::operator ++ ();
            return *this;
        }

        iterator_base operator ++ (int)
        {
            iterator_base result = *this;
            operator ++ ();
            return result;
        }

        iterator_base & operator -- ()
        {
            base_it_type::operator -- ();
            return *this;
        }

        iterator_base operator -- (int)
        {
            iterator_base result = *this;
            operator -- ();
            return result;
        }

    protected:
        data_type * deref() const
        {
            return *base_it_type::deref();
        }
    };

    typedef judy_iterator<iterator_base, data_type> iterator;
    typedef judy_iterator<iterator_base, const data_type> const_iterator;
    typedef judy_pair_iterator<iterator_base, data_type> const_pair_iterator;

    iterator begin()
    {
        return iterator_base(this, 0);
    }

    iterator end()
    {
        return iterator_base(this);
    }

    const_iterator begin() const
    {
        return iterator_base(this, 0);
    }

    const_iterator end() const
    {
        return iterator_base(this);
    }

    std::pair<iterator, bool> insert(const value_type & x)
    {
        data_type ** ppvalue = reinterpret_cast<data_type **>(jlg(x.first));
        //data_type ** ppvalue = (data_type **)JudyLGet(array, x.first, PJE0);
        bool inserted = false;
        if (!ppvalue) {
            ppvalue = (data_type **)JudyLIns(&array, x.first, PJE0);
            (*ppvalue) = new data_type(x.second);
            inserted = true;
        }
        
        return std::make_pair(iterator_base(this, x.first),
                              inserted);
    }

    template<class Iterator>
    void insert(Iterator first, Iterator last)
    {
        while (first != last) insert(*first++);
    }

    void erase(iterator_base pos)
    {
        erase(pos.key());
    }

    size_t erase(const key_type & key)
    {
        /* Need to find it first to call the destructor. */
        data_type ** ppvalue = reinterpret_cast<data_type **>(this->jlg(key));
        //data_type ** pvalue = (data_type **)JudyLGet(array, key, PJE0);
        if (ppvalue) {
            delete *ppvalue;
            int res = JudyLDel(&array, key, PJE0);
            if ((void *)(size_t)res == PJERR) throw std::bad_alloc();
            return res;
        }
        else return 0;
    }

    void erase(iterator_base first, iterator_base last)
    {
        while (first != last) erase(first++.key());
    }

    iterator find(const key_type & key)
    {
        return judyl_base::find(key);
    }
    
    const_iterator find(const key_type & key) const
    {
        return judyl_base::find(key);
    }
    
    void clear()
    {
        /* Destroy all of the leaf objects. */
        for (iterator it = begin();  it != end();  ++it)
            delete &(*it);
        
        judyl_base::clear();
    }
    
    data_type & operator [] (const key_type & key)
    {
        data_type ** ppvalue = reinterpret_cast<data_type **>(this->jlg(key));
        //data_type ** ppvalue = (data_type **)JudyLGet(array, key, PJE0);
        if (!ppvalue) {
            ppvalue = (data_type **)JudyLIns(&array, key, PJE0);
            (*ppvalue) = new data_type();
        }
        return **ppvalue;
    }

    const data_type & operator [] (const key_type & key) const
    {
        static data_type NONE;
        data_type ** ppvalue = (data_type **)JudyLGet(array, key, PJE0);
        if (ppvalue) return **ppvalue;
        else return NONE;
    }

    size_t memusage_() const
    {
        if (mem_traits<Data>::is_fixed)
            return size() * (sizeof(Data) + sizeof(void *))
                + judyl_base::memusage_();
        else {
            size_t result = judyl_base::memusage_();
            for (const_iterator it = begin();  it != end();  ++it)
                result += memusage(*it) + sizeof(void *);
            return result;
        }
    }
};

template<typename Data, bool Contained>
size_t memusage(const judyl_typed<Data, Contained> & array)
{
    return array.memusage_();
}


} // namespace ML



#endif /* __boosting__judy_array_h__ */
