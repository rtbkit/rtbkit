/* training_index_iterators.cc
   Jeremy Barnes, 19 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Implementation of index iterators.
*/

#include "training_index_iterators.h"
#include "jml/db/persistent.h"
#include <iostream>
#include "jml/utils/string_functions.h"

using namespace std;

namespace ML {


/*****************************************************************************/
/* JOINT_INDEX                                                               */
/*****************************************************************************/

Joint_Index::
Joint_Index(const float * values, const uint16_t * buckets,
            const Label * labels, const unsigned * examples,
            const unsigned * counts, const float * divisors,
            unsigned size, const std::vector<float> * bucket_vals)
    : values_(values), buckets_(buckets), labels_(labels), examples_(examples),
      counts_(counts), divisors_(divisors),
      size_(size), bucket_vals_(bucket_vals)
{
    //cerr << "values: " << (bool)values << " buckets: " << (bool)buckets
    //     << " labels: " << (bool)labels << " examples: " << (bool)examples
    //     << " counts: " << (bool)counts << " size: " << size << endl;
}

Joint_Index::
Joint_Index()
    : values_(0), buckets_(0), labels_(0), examples_(0),
      counts_(0), divisors_(0),
      size_(0), bucket_vals_(0)
{
}

void Joint_Index::dump(std::ostream & stream) const
{
    stream << Index_Iterator::titles() << endl;
    for (unsigned i = 0;  i < size_;  ++i)
        stream << operator [] (i).print() << endl;
}


/*****************************************************************************/
/* INDEX_ITERATOR                                                            */
/*****************************************************************************/

std::string Index_Iterator::print() const
{
    return format("%10g %4d %4d %6d %3d %5.2f",
                  value(), bucket(), label().label(), example(),
                  example_counts(), divisor());
}

std::string Index_Iterator::titles()
{
    return "     value buck labl  examp cnt divis";
}

} // namespace ML
