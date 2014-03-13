/* split.cc
   Jeremyk Barnes, 5 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "split.h"
#include "feature_space.h"
#include "feature_info.h"
#include "feature_set.h"
#include "classifier.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* SPLIT                                                                     */
/*****************************************************************************/

Split::Op
Split::
get_op_from_feature(const Feature_Info & info)
{
    if (info.type() == CATEGORICAL
        || info.type() == STRING)
        return EQUAL;
    return LESS;
}

Split::
Split(const Feature & feature, float split_val, const Feature_Space & fs)
    : feature_(feature), split_val_(split_val), opt_(false), idx_(0)
{
    if (split_val == -INFINITY) {
        split_val_ = 0.0;
        op_ = NOT_MISSING;
    }
    else {
        Feature_Info info = fs.info(feature);
        op_ = get_op_from_feature(info);
    }
    validate(fs);
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

void
Split::
optimize(const Optimization_Info & info)
{
    map<Feature, int>::const_iterator it
        = info.feature_to_optimized_index.find(feature_);
    if (it == info.feature_to_optimized_index.end())
        throw Exception("Split::optimize(): feature not found");
    idx_ = it->second;
    opt_ = true;
}

std::string
Split::
print(const Feature_Space & fs, int branch) const
{
    string feat_name = fs.print(feature_);
    Feature_Type type = fs.info(feature_).type();

    //cerr << "feature " << feat_name << " type " << type << " branch "
    //     << branch << endl;

    if (type == BOOLEAN) {
        if (op_ != LESS || split_val_ != 0.5)
            throw Exception("unknown boolean branch: op %s, split_val %f, "
                            "branch %d",
                            ML::print((Op)op_).c_str(), split_val_, branch);
        switch (branch) {
        case true: return "!" + feat_name;
        case false: return " " + feat_name;
        case MISSING: return feat_name + " missing";
        default:
            throw Exception("bad branch");
        }
    }

    string val = fs.print(feature_, split_val_);

    switch (branch) {
    case true:
        switch (op_) {
        case EQUAL:       return feat_name + "  = " + val;
        case LESS:        return feat_name + "  < " + val;
        case NOT_MISSING: return feat_name + " not missing";
        default:
            throw Exception("split::print(): invalid op");
        };
    case false:
        switch (op_) {
        case EQUAL:       return feat_name + " != " + val;
        case LESS:        return feat_name + " >= " + val;
        case NOT_MISSING: return feat_name + " missing";
        default:
            throw Exception("split::print(): invalid op");
        };
    case MISSING:
        return feat_name + " missing";
    default:
        throw Exception("invalid branch name");
    };
}

void
Split::
serialize(DB::Store_Writer & store,
          const Feature_Space & fs) const
{
    store << (unsigned char)0;  // version
    fs.serialize(store, feature_);
    unsigned char ucop = (unsigned char)op_;
    store << ucop; // operation
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
validate() const
{
    if (isnanf(split_val_)) {
        using namespace std;
        cerr << "split_val_ = " << split_val_ << endl;
        cerr << "feature_ = " << feature_ << endl;
        cerr << "op_ = " << op_ << endl;
        throw Exception("bad split val");
    }
    
    if (!finite(split_val_)) {
        using namespace std;
        cerr << "split_val_ = " << split_val_ << endl;
        cerr << "feature_ = " << feature_ << endl;
        cerr << "op_ = " << op_ << endl;
        throw Exception("non-finite split val");
    }
}

void
Split::
validate(const Feature_Space & fs) const
{
    if (isnanf(split_val_)) {
        using namespace std;
        cerr << "split_val_ = " << split_val_ << endl;
        cerr << "feature_ = " << feature_ << endl;
        cerr << "feature name = " << fs.print(feature_) << endl;
        cerr << "op_ = " << op_ << endl;
        throw Exception("bad split val");
    }
    
    if (!finite(split_val_)) {
        using namespace std;
        cerr << "split_val_ = " << split_val_ << endl;
        cerr << "feature_ = " << feature_ << endl;
        cerr << "feature name = " << fs.print(feature_) << endl;
        cerr << "op_ = " << op_ << endl;
        throw Exception("non-finite split val");
    }
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
