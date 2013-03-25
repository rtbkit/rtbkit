/* tags.h                                                          -*- C++ -*-
   Jeremy Barnes, 26 February 2013
   
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Definitions of the "tags" class used for creative and campaign filtering.
   
   This file is part of RTBkit.
*/

#pragma once


namespace RTBKIT {


/*****************************************************************************/
/* TAG                                                                       */
/*****************************************************************************/

/** This is how an individual tag is set up. */

struct Tag {
    std::string scope;
    std::string key;
    std::string value;
};


/*****************************************************************************/
/* TAGS                                                                      */
/*****************************************************************************/

struct Tags {

    

    /// List of active tags
    std::vector<uint64_t> active;
};

#if 0
struct TagsContext {
    BidRequest * breq;
};

struct Predicate {
    virtual ~Predicate()
    {
    }

    bool evaluate(const Tags & tags, TagsContext & context) const = 0;
};

struct OrPredicate: public Predicate {
    
    std::vector<std::unique_ptr<Predicate> > subexpr;
};

struct AndPredicate: public Predicate {


    std::vector<std::unique_ptr<Predicate> > subexpr;
};
#endif

/*****************************************************************************/
/* TAG FILTER                                                                */
/*****************************************************************************/

/** Tag filter.  Represents

    All in mustInclude are set and none in mustNotInclude are set.
*/

struct TagFilter {
    Tags mustIncludeOneOf;
    Tags mustIncludeAllOf;
    Tags mustNotIncludeAnyOf;

    /** Does this filter match the given set of tags? */
    bool matches(const Tags & tagsToMatch) const;
};


/*****************************************************************************/
/* TAG FILTER EXPRESSION                                                     */
/*****************************************************************************/

/** Class to deal with evaluating a filter expression. */

struct TagFilterExpression : public std::vector<TagFilter> {
    
};

} // namespace RTBKIT


