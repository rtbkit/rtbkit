/* data_aliases.h                                                  -*- C++ -*-
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Returns the aliases for the given function.
*/

#include "config.h"
#include <set>
#include "training_data.h"

namespace ML {

/** This structure holds the information about a group of aliased
    examples. */
struct Alias {
    std::set<int> examples;  ///< which examples are aliased
    bool homogenous;         ///< true if they have the same label
};

/** Returns a list of examples that are aliased. */
std::vector<Alias> aliases(const ML::Training_Data & dataset,
                           const Feature & predicted);

/** Remove the aliased examples.  This will go through and remove
    all of the non-homogenous examples from the list, and if
    \p homogeneous is true, then it will remove all but the first of the
    homogeneous examples.
    
    Returns the number of examples removed.

    The mapping argument, if non null, will initialize the vector with a
    vector that gives, for each example in the original dataset, the example
    number that it corresponds to in the new dataset.
    
    Note that this is an expensive function to call, as it requires that
    the data be completely reindexed.  It is better to call this before
    the finish() function is called, as otherwise that work will be
    wasted.
*/
int remove_aliases(ML::Training_Data & dataset,
                   const std::vector<Alias> & aliases,
                   bool homogenous = false,
                   std::vector<int> * mapping = 0);

/** Call this method to clean up the data.  It will detect examples that
    occur twice with different labels and remove them from the
    corpus.  Groups of aliased examples will be returned. */
std::vector<Alias>
remove_aliases(ML::Training_Data & dataset,
               const Feature & predicted,
               bool homogenous = false,
               std::vector<int> * mapping = 0);

} // namespace ML

