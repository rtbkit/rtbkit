/* feature_space.cc
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of a feature space.
*/

#include "feature_space.h"
#include "jml/utils/parse_context.h"
#include "jml/db/persistent.h"
#include "registry.h"
#include "training_data.h"
#include <boost/lexical_cast.hpp>


using namespace std;

namespace ML {


/*****************************************************************************/
/* FEATURE_SPACE                                                             */
/*****************************************************************************/

Feature_Space::~Feature_Space()
{
}

std::string Feature_Space::print(const Feature & feature) const
{
    return feature.print();
}

std::string Feature_Space::print(const Feature & feature, float value) const
{
    std::shared_ptr<const Categorical_Info> cinfo
        = info(feature).categorical();

    if (round(value) == value && cinfo) return cinfo->print((int)value);
    else return format("%.8g", value);
}

void Feature_Space::parse(const std::string & name, Feature & feature) const
{
    Parse_Context context(name, name.c_str(), name.c_str() + name.length());
    if (parse(context, feature)) return;
    else throw Exception("Feature_Space::parse(): feature with name "
                         + name + " could not be parsed");
}

bool Feature_Space::parse(Parse_Context & context, Feature & feature) const
{
    Parse_Context::Revert_Token tok(context);
    if (!context.match_literal('(')) return false;

    int type;
    if (!context.match_int(type)) return false;
    feature.set_type(type);

    context.skip_whitespace();
    int arg1;
    if (!context.match_int(arg1)) return false;
    feature.set_arg1(arg1);

    context.skip_whitespace();
    int arg2;
    if (!context.match_int(arg2)) return false;
    feature.set_arg2(arg2);

    if (!context.match_literal(')')) return false;
    tok.ignore();
    return true;
}

void Feature_Space::
expect(Parse_Context & context, Feature & feature) const
{
    if (!parse(context, feature))
        context.exception("expected feature tuple '(type arg1 arg2)'");
}

static const DB::compact_size_t FS_SERIALIZE_VERSION(1);

void
Feature_Space::
serialize(DB::Store_Writer & store, const Feature & feature) const
{
    store << DB::compact_size_t(feature.type())
          << DB::compact_size_t(feature.arg1())
          << DB::compact_size_t(feature.arg2());
}

void
Feature_Space::
reconstitute(DB::Store_Reader & store, Feature & feature) const
{
    DB::compact_size_t type, arg1, arg2;
    store >> type >> arg1 >> arg2;
    feature = Feature(type, arg1, arg2);
}

void
Feature_Space::
serialize(DB::Store_Writer & store,
          const Feature & feature,
          float value) const
{
    Feature_Info finfo = info(feature);
    if (finfo.type() != STRING)
        store << value;
    else {
        //cerr << "serializing STRING for feature " << print(feature)
        //     << " with value " << value << endl;
        string s = (finite(value) ? finfo.categorical()->print((int)value)
                    : format("!!!%f", value));
        //cerr << " string was " << s << endl;
        store << s;
    }
}

void
Feature_Space::
reconstitute(DB::Store_Reader & store,
             const Feature & feature,
             float & value) const
{
    Feature_Info finfo = info(feature);
    if (finfo.type() != STRING)
        store >> value;
    else {
        string s;
        store >> s;

        if (s.size() > 3 && s[0] == '!' && s[1] == '!' && s[2] == '!') {
            /* Handle NaN, inf, -inf, etc. */
            s = string(s, 3);
            value = boost::lexical_cast<float>(s);
        }
        else value = finfo.categorical()->lookup(s);
    }
}

std::string
Feature_Space::
print(const Feature_Set & fs) const
{
    std::string result;
    for (unsigned i = 0;  i < fs.size();  ++i) {
        if (i > 0) result += ' ';
        result += escape_feature_name(print(fs[i].first))
            + ':' + escape_feature_name(print(fs[i].first, fs[i].second));
    }
    return result;
}

void
Feature_Space::
serialize(DB::Store_Writer & store, const Feature_Set & fs) const
{
    store << FS_SERIALIZE_VERSION;
    store << DB::compact_size_t(fs.size());
    for (unsigned i = 0;  i < fs.size();  ++i) {
        serialize(store, fs[i].first);
        store << fs[i].second;
    }
}

void Feature_Space::
reconstitute(DB::Store_Reader & store,
             std::shared_ptr<Feature_Set> & fs) const
{
    DB::compact_size_t version;
    store >> version;

    switch (version) {
    case 1: {
        DB::compact_size_t size(store);

        std::shared_ptr<Mutable_Feature_Set> fs2(new Mutable_Feature_Set());
        fs2->features.reserve(size);
        fs2->clear();

        for (unsigned i = 0;  i < size;  ++i) {
            Feature feat;
            float value;
            reconstitute(store, feat);
            store >> value;
            fs2->add(feat, value);
        }

        fs = fs2;

        break;
    }
        
    default:
        throw Exception(format("Feature_Space (concrete %s) reconstitute: "
                               "attempt to reconstitute feature set of "
                               "version %zd; only %zd supported",
                               demangle(typeid(*this).name()).c_str(),
                               version.size_,
                               FS_SERIALIZE_VERSION.size_));
    }
}

void Feature_Space::serialize(DB::Store_Writer & store) const
{
    store << class_id();
}

void Feature_Space::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & fs)
{
    // TODO: get to bottom of why we pass the feature space here.
    string id;
    store >> id;
    if (id != class_id())
        throw Exception("object of class_id '" + class_id() + "', type '"
                        + demangle(typeid(*this).name())
                        + "' attempted to reconstitute class_id of '"
                        + id + "'");
}

std::string Feature_Space::print() const
{
    return class_id();
}

std::shared_ptr<Training_Data>
Feature_Space::
training_data(const std::shared_ptr<const Feature_Space> & fs) const
{
    return std::shared_ptr<Training_Data>(new Training_Data(fs));
}

void
Feature_Space::
freeze()
{
}

namespace {

const vector<Feature> NO_FEATURES;

} // file scope

const std::vector<Feature> &
Feature_Space::
dense_features() const
{
    if (type() == DENSE)
        throw Exception("Feature space of type "
                        + demangle(typeid(*this).name())
                        + " should override dense_features()");
    return NO_FEATURES;
}

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const std::shared_ptr<Feature_Space> & fs)
{
    Registry<Feature_Space>::singleton().serialize(store, fs.get());
    return store;
}

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const std::shared_ptr<const Feature_Space> & fs)
{
    Registry<Feature_Space>::singleton().serialize(store, fs.get());
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             std::shared_ptr<Feature_Space> & fs)
{
    //cerr << "reconstituting feature space..." << endl;
    fs = Registry<Feature_Space>::singleton().reconstitute(store);
    //cerr << "done." << endl;
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             std::shared_ptr<const Feature_Space> & fs)
{
    //cerr << "reconstituting feature space..." << endl;
    fs = Registry<Feature_Space>::singleton().reconstitute(store);
    //cerr << "done." << endl;
    return store;
}


/*****************************************************************************/
/* MUTABLE_FEATURE_SPACE                                                     */
/*****************************************************************************/

Mutable_Feature_Space::~Mutable_Feature_Space()
{
}

void
Mutable_Feature_Space::
set_info(const Feature & feature, const Feature_Info & info)
{
    throw Exception("Feature_Space::set_info(): feature space of type "
                    + demangle(typeid(*this).name())
                    + " don't allow feature info to be set.");
}

void
Mutable_Feature_Space::
import(const Feature_Space & from)
{
    throw Exception("Feature_Set: import() needs to be overridden");
}



} // namespace ML

