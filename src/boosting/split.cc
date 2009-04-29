/* split.cc
   Jeremyk Barnes, 5 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "split.h"
#include "feature_space.h"
#include "feature_info.h"
#include "feature_set.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* SPLIT                                                                     */
/*****************************************************************************/

Split::Op
Split::
get_op_from_feature(const Feature_Info & info)
{
    if (info.type() == Feature_Info::CATEGORICAL
        || info.type() == Feature_Info::STRING)
        return EQUAL;
    return LESS;
}

Split::
Split(const Feature & feature, float split_val, const Feature_Space & fs)
    : feature_(feature), split_val_(split_val), opt_(false), idx_(0)
{
    Feature_Info info = fs.info(feature);
    op_ = get_op_from_feature(info);
}

void
Split::
apply(const Feature_Set & fset,
      Weights & weights,
      float weight) const
{
    Feature_Set::const_iterator first, last;
    boost::tie(first, last) = fset.find(feature_);
    return apply(first, last, weights, weight);
}

std::string
Split::
print(const Feature_Space & fs) const
{
    string feat_name = fs.print(feature_);
    string val = fs.print(feature_, split_val_);
    
    switch (op_) {
    case EQUAL:       return feat_name + " = " + val;
    case LESS:        return feat_name + " < " + val;
    case NOT_MISSING: return feat_name + " not missing";
    default:
        throw Exception("split::print(): invalid op");
    }
}

void
Split::
serialize(DB::Store_Writer & store,
          const Feature_Space & fs) const
{
    store << (unsigned char)0;  // version
    fs.serialize(store, feature_);
    store << (unsigned char)op_; // operation
    if (op_ == EQUAL || op_ == LESS)
        fs.serialize(store, feature_, split_val_);
}

void
Split::
reconstitute(DB::Store_Reader & store,
             const Feature_Space & fs)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw Exception("Split::reconstitute(): unknown version");
    fs.reconstitute(store, feature_);
    unsigned char op;
    store >> op;
    op_ = op;
    if (op_ == EQUAL || op_ == LESS)
        fs.reconstitute(store, feature_, split_val_);
    else split_val_ = numeric_limits<float>::quiet_NaN();
}

void
Split::
throw_invalid_op_exception(Op op)
{
    throw Exception("invalid split op: " + ML::print(op));
}

std::string print(Split::Op op)
{
    switch (op) {
    case Split::LESS: return "LESS";
    case Split::EQUAL: return "EQUAL";
    case Split::NOT_MISSING: return "NOT_MISSING";
    default: return format("Split::Op(%d)", op);
    }
}

std::ostream & operator << (std::ostream & stream, Split::Op op)
{
    return stream << print(op);
}


} // namespace ML
