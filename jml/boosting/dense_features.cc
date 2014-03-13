/* dense_features.cc
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.

   Dense feature set.  Implementation.
*/

#include "config_impl.h"
#include "dense_features.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/pair_utils.h"
#include "registry.h"
#include <boost/timer.hpp>
#include "jml/utils/fast_float_parsing.h"
#include <boost/timer.hpp>
#include "jml/utils/file_functions.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/smart_ptr_utils.h"
#include <boost/tuple/tuple.hpp>
#include "stdint.h"

using namespace std;
using namespace DB;



namespace ML {

/*****************************************************************************/
/* DENSE_FEATURE_MAPPING                                                     */
/*****************************************************************************/

bool Dense_Feature_Mapping::initialized() const
{
    return initialized_;
}

void Dense_Feature_Mapping::clear()
{
    vars.clear();
    categories.clear();
    initialized_ = false;
    num_vars_expected_ = -1;
}


/*****************************************************************************/
/* DENSE_FEATURE_SPACE                                                       */
/*****************************************************************************/

Dense_Feature_Space::Dense_Feature_Space()
{
}

Dense_Feature_Space::Dense_Feature_Space(DB::Store_Reader & store)
{
    reconstitute(store);
}

Dense_Feature_Space::Dense_Feature_Space(size_t feature_count)
{
    vector<string> feature_names(feature_count);
    for (unsigned i = 0;  i < feature_count;  ++i)
        feature_names[i] = format("FEATURE%d", i);
    init(feature_names);
}

Dense_Feature_Space::
Dense_Feature_Space(const std::vector<std::string> & feature_names)
{
    init(feature_names);
}

Dense_Feature_Space::~Dense_Feature_Space()
{
}

void Dense_Feature_Space::
init(const vector<string> & feature_names, Feature_Type type)
{
    Feature_Info def_info(type);
    vector<Mutable_Feature_Info> feature_info(feature_names.size(), def_info);
    init(feature_names, feature_info);
}

void
Dense_Feature_Space::
init(const vector<string> & feature_names,
     const vector<Mutable_Feature_Info> & feature_info)
{
    if (feature_names.size() != feature_info.size())
        throw Exception("Dense_Feature_Space: "
                        "names and info arrays don't match length");

    info_array = feature_info;
    set_feature_names(feature_names);

    features_.clear();
    for (unsigned i = 0;  i < feature_names.size();  ++i)
        features_.push_back(Feature(i));
}

std::shared_ptr<Mutable_Feature_Set>
Dense_Feature_Space::encode(const std::vector<float> & variables) const
{
    if (variables.size() != info_array.size())
        throw Exception
            (format("Dense_Feature_Space::encode(): expected %zd variables, "
                    "got %zd", info_array.size(), variables.size()));
    std::shared_ptr<Mutable_Feature_Set> result
        (new Mutable_Feature_Set(pair_merger(features_.begin(),
                                             variables.begin()),
                                 pair_merger(features_.end(),
                                             variables.end())));
    result->sort();
    return result;
}

std::shared_ptr<Mutable_Feature_Set>
Dense_Feature_Space::
encode(const std::vector<float> & variables,
       const Dense_Feature_Space & other) const
{
    Mapping mapping;  // temporary...
    return encode(variables, other, mapping);
}

void Dense_Feature_Space::
create_mapping(const Dense_Feature_Space & other,
               Mapping & mapping) const
{
    //cerr << "create_mapping" << endl;

    /* Set up the mapping.  We keep, for each of our variables, the index
       of the variable that it corresponds to in the other feature space.
       Note that we make sure that the feature info matches as well as
       the name.
    */

#if 0
    cerr << "other mapping:" << endl;
    for (map<string, int>::const_iterator it = other.names_bwd.begin();
         it != other.names_bwd.end();  ++it)
        cerr << it->first << " = " << it->second << endl;
    cerr << endl;

    cerr << "create_mapping: this = " << print() << endl;
    cerr << "create_mapping: other = " << other.print() << endl;
#endif

    mapping.clear();

    mapping.vars.resize(names_fwd.size());
    mapping.categories.resize(names_fwd.size());

    mapping.num_vars_expected_ = other.names_fwd.size();

    for (unsigned i = 0;  i < names_fwd.size();  ++i) {
        //cerr << "mapping feature " << names_fwd[i] << endl;
        map<string, int>::const_iterator it
            = other.names_bwd.find(names_fwd[i]);
        if (it == other.names_bwd.end()
            /*|| info_array[i] != other.info_array[it->second]*/) {
            //cerr << "us = " << info_array[i].print()
            //     << " them = " << other.info_array[it->second]
            //     << endl;
            mapping.vars[i] = -1;
            continue;
        }
        else {
            mapping.vars[i] = it->second;
        }
        //cerr << "mapping[" << i << "] = " << mapping.back() << endl;

        //cerr << "us:   " << info_array[i].print() << endl;
        //cerr << "them: " << other.info_array[it->second].print() << endl;

        /* Map the values of categories. */
        if (info_array[i].categorical()) {
            if (other.info_array[it->second].categorical()) {
                if (other.info_array[it->second].type() != info_array[i].type())
                    throw Exception("Dense_Feature_Space::map(): types not "
                                    "the same");

                //cerr << "mapping between "
                //     << demangle(typeid(ci).name())
                //     << " and "
                //    
                //     << demangle(typeid(*other.info_array[it->second]
                //                        .categorical()).name())
                //     << endl;


                if (info_array[i].type() == CATEGORICAL) {
                    mapping.categories[i]
                        = make_sp(new Fixed_Categorical_Mapping
                                  (info_array[i].categorical(),
                                   other.info_array[it->second].categorical()));
                }
                else if (info_array[i].type() == STRING) {
                    mapping.categories[i]
                        = make_sp(new Dynamic_Categorical_Mapping());
                }
                else throw Exception("Dense_Feature_Space::map(): unknown type");

            }
            else {
                mapping.clear();
                throw Exception("Mapping from non-categorical to categorical "
                                "for feature " + names_fwd[i]);
#if 0
                /* We are categories but the other is not.  We just encode
                   the variables as-is, and hope. */
                int nc = ci.count();
                mapping.categories[i].resize(nc);

                std::iota(mapping.categories[i].begin(),
                          mapping.categories[i].end(), 0);
#endif
            }
        }
        else if (other.info_array[it->second].categorical()) {
            mapping.clear();
            throw Exception("Mapping from categorical to non-categorical "
                            "for feature " + names_fwd[i]);
            /* We just hope that it works... */
            /* Maybe print a warning? */
        }
    }

    //cerr << "mapping.vars.size() = " << mapping.vars.size() << endl;
    //cerr << "mapping.categories.size() = " << mapping.categories.size() << endl;

    mapping.initialized_ = true;
}

std::shared_ptr<Mutable_Feature_Set>
Dense_Feature_Space::
encode(const std::vector<float> & variables,
       const Dense_Feature_Space & other,
       Mapping & mapping) const
{
    if (!mapping.initialized())
        create_mapping(other, mapping);

    return encode(variables, other, (const Mapping &)mapping);
}
    
std::shared_ptr<Mutable_Feature_Set>
Dense_Feature_Space::
encode(const std::vector<float> & variables,
       const Dense_Feature_Space & other,
       const Mapping & mapping) const
{
    if (!mapping.initialized()
        || mapping.vars.size() != names_fwd.size()
        || mapping.categories.size() != names_fwd.size())
        throw Exception("using wrong or stale mapping");

    if (variables.size() != mapping.num_vars_expected_) {
        cerr << "variables.size() = " << variables.size() << endl;
        cerr << "mapping.num_vars_expected = " << mapping.num_vars_expected_
             << endl;
        cerr << "mapping.vars.size() = " << mapping.vars.size() << endl;
        throw Exception("wrong number of variables expected; have you reversed "
                        "feature spaces?");
    }
    
    /* Map their variables into ours.  Those that have no corresponding
       variable get a NaN instead. */
    distribution<float> our_variables(names_fwd.size(), NAN);
    for (unsigned i = 0;  i < mapping.vars.size();  ++i) {
        if (mapping.vars[i] != -1) {
            if (mapping.categories[i]) {
                /* Map a categorical variable. */
                our_variables[i]
                    = mapping.categories[i]
                    ->map((int)variables[mapping.vars[i]],
                          *info_array[i].categorical(),
                          *other.info_array[mapping.vars[i]].categorical());
            }
            else our_variables[i] = variables[mapping.vars[i]];
        }
    }
    
    //cerr << "encode: with map done" << endl;
    /* Now we have variables in our usual space, we can call the usual
       encoding procedure. */
    return encode(our_variables);
}

void
Dense_Feature_Space::
encode(const float * features,
       float * output,
       const Dense_Feature_Space & other,
       const Mapping & mapping) const
{
    if (!mapping.initialized()
        || mapping.vars.size() != names_fwd.size()
        || mapping.categories.size() != names_fwd.size())
        throw Exception("using wrong or stale mapping");

    /* Map their variables into ours.  Those that have no corresponding
       variable get a NaN instead. */
    for (unsigned i = 0;  i < mapping.vars.size();  ++i) {
        if (mapping.vars[i] != -1) {
            if (mapping.categories[i]) {
                /* Map a categorical variable. */
                output[i]
                    = mapping.categories[i]
                    ->map((int)features[mapping.vars[i]],
                          *info_array[i].categorical(),
                          *other.info_array[mapping.vars[i]].categorical());
            }
            else output[i] = features[mapping.vars[i]];
        }
        else output[i] = NAN;
    }
}

distribution<float>
Dense_Feature_Space::
decode(const Feature_Set & feature_set) const
{
    /* TODO: make this be able to work on another feature space. */

    if (feature_set.size() != variable_count())
        throw Exception("Dense_Feature_Space::encode(): variable counts don't "
                        "match.");
    
    distribution<float> result(variable_count());

    for (unsigned i = 0;  i < feature_set.size();  ++i)
        result[i] = feature_set[i].second;

    return result;
}

Feature_Info
Dense_Feature_Space::info(const ML::Feature & feature) const
{
    if (feature == MISSING_FEATURE) return MISSING_FEATURE_INFO;
    
    if (feature.type() < 0 || feature.type() >= variable_count())
        throw Exception
            (format("unknown variable number %zd in Dense_Feature_Type::info "
                    "with variable_count() %zd", (size_t)feature.type(),
                    variable_count()));
    return info_array[feature.type()];
}

void Dense_Feature_Space::
set_info(const Feature & feature, const Feature_Info & info)
{
    //bool cat = info.categorical();

    //if (cat)
    //    cerr << "set_info: cat is "
    //         << demangle(typeid(*info.categorical()).name())
    //         << endl;

    if (feature.type() < 0 || feature.type() >= variable_count())
        throw Exception
            (format("unknown variable number %zd in Dense_Feature_Type::info "
                    "with variable_count() %zd", (size_t)feature.type(),
                    variable_count()));
    info_array[feature.type()] = info;

    //if (cat)
    //    cerr << "set_info: cat is after "
    //         << demangle(typeid(*info_array[feature.type()].categorical()).name())
    //         << endl;
}

std::string
Dense_Feature_Space::print(const ML::Feature & feature) const
{
    if (feature == MISSING_FEATURE) return "MISSING";
    if (feature.type() < 0 || feature.type() >= variable_count())
        return format("FEATURE%zd", (size_t)feature.type());
    //throw Exception("unknown variable number in Dense_Feature_Type");
    return names_fwd[feature.type()];
}

std::string
Dense_Feature_Space::
print(const Feature & feature, float value) const
{
    if (feature.type() < 0 || feature.type() >= variable_count())
        throw Exception("Dense_Feature_Space::print(): error: bad feature");
    
    if (info_array[feature.type()].categorical()) {
        if (!finite(value)) return ostream_format(value);

        if (round(value) != value)
            throw Exception("Dense_Feature_Space::print(): "
                            "error: categorical but non-integral value");
        //cerr << "printing value " << value << endl;
        return info_array[feature.type()].categorical()->print((int)value);
    }
    else return Feature_Space::print(feature, value);
}

void Dense_Feature_Space::
parse(const std::string & name, Feature & feature) const
{
    int index = feature_index(name);
    if (index == -1)
        throw Exception("Dense_Feature_Space::parse(): feature with name "
                        + name + " not found");
    feature = Feature(index);
}

bool
Dense_Feature_Space::
parse(Parse_Context & context, ML::Feature & feature) const
{
    /* Todo: quoting, etc. */
    vector<pair<string, size_t> > names_and_lengths;
    for (unsigned i = 0;  i < names_fwd.size();  ++i)
        if (names_fwd[i].size() > 0)
            names_and_lengths.push_back
                (make_pair(names_fwd[i], names_fwd[i].size()));
    sort_on_second_descending(names_and_lengths);

    for (unsigned i = 0;  i < names_and_lengths.size();  ++i) {
        //cerr << "trying " << names_and_lengths[i].first << endl;
        if (context.match_literal(names_and_lengths[i].first.c_str())) {
            parse(names_and_lengths[i].first, feature);
            //cerr << "got feature " << names_and_lengths[i].first << endl;
            return true;
        }
    }

    return false;
}

void
Dense_Feature_Space::
expect(Parse_Context & context, ML::Feature & feature) const
{
    if (!parse(context, feature))
        throw Exception("Expected name of feature");
}

void
Dense_Feature_Space::
serialize(DB::Store_Writer & store, const ML::Feature & feature) const
{
    store << compact_size_t(feature.type());
}

void
Dense_Feature_Space::
reconstitute(DB::Store_Reader & store, ML::Feature & feature) const
{
    compact_size_t new_feature(store);
    feature = Feature(new_feature);
}

std::string
Dense_Feature_Space::
print(const Feature_Set & fs) const
{
    if (fs.size() != variable_count())
        throw Exception("Attempt to print Feature_Set of wrong size for "
                        "Dense_Feature_Space");

    //cerr << "print: fs.size() = " << fs.size() << endl;

    /* Assume it's in the right order. */
    std::string result;
    for (unsigned i = 0;  i < fs.size();  ++i) {
        if (i != 0) result += ' ';
        if (i > 0 && fs[i].first.type() != fs[i-1].first.type() + 1) {
            throw Exception("Feature_Set out of order for "
                            "Dense_Feature_Space");
        }
        result += print(fs[i].first, fs[i].second);
    }

    return result;
}

namespace {

static const compact_size_t DFS_SERIALIZE_VERSION = 1;

} // file scope

void Dense_Feature_Space::
serialize(DB::Store_Writer & store, const Feature_Set & fs) const
{
    if (fs.size() != variable_count())
        throw Exception("Attempt to serialize Feature_Set of wrong size for "
                        "Dense_Feature_Space");

    store << DFS_SERIALIZE_VERSION;

    Feature_Space::serialize(store, fs);
}

void Dense_Feature_Space::
reconstitute(DB::Store_Reader & store, std::shared_ptr<Feature_Set> & fs) const
{
    compact_size_t version(store);

    switch (version) {
    case 1: {
        Feature_Space::reconstitute(store, fs);
        
        break;
    }
        
    default:
        throw Exception(format("Dense_Feature_Space::reconstitute(Feature_Set): "
                               "attempt to reconstitute feature set of "
                               "version %zd; only %zd supported",
                               version.size_,
                               DFS_SERIALIZE_VERSION.size_));
    }
}

/* Methods to deal with the feature space as a whole. */
Dense_Feature_Space * Dense_Feature_Space::make_copy() const
{
    return new Dense_Feature_Space(*this);
}

std::string Dense_Feature_Space::print() const
{
    string result;
    for (unsigned i = 0;  i < names_fwd.size();  ++i) {
        if (i != 0) result += " ";
        result += escape_feature_name(names_fwd[i]) + ":" + info_array[i].print();
    }

    //cerr << "Dense_Feature_Space::print(): result = '" << result << "'"
    //     << endl;
    
    return result;
}

const std::vector<std::string> & Dense_Feature_Space::feature_names() const
{
    return names_fwd;
}

void Dense_Feature_Space::
set_feature_names(const std::vector<std::string> & feature_names)
{
    if (feature_names.size() != variable_count())
        throw Exception(format("Dense_Feature_Space: %zd feature names given, "
                               "%zd required", feature_names.size(),
                               variable_count()));
    names_fwd = feature_names;
    names_bwd.clear();
    for (unsigned i = 0;  i < variable_count();  ++i)
        names_bwd[names_fwd[i]] = i;
}

Feature
Dense_Feature_Space::
make_feature(const std::string & name, const Feature_Info & info)
{
    return Feature(add_feature(name, info));
}

Feature
Dense_Feature_Space::
get_feature(const std::string & name) const
{
    int index = feature_index(name);
    if (index == -1)
        throw Exception("Dense_Feature_Space::get_feature(): "
                        "feature with name " + name + " not found");
    return Feature(index);
}

int Dense_Feature_Space::feature_index(const std::string & name) const
{
    map<string, int>::const_iterator it = names_bwd.find(name);
    if (it == names_bwd.end()) return -1;
    return it->second;
}

int Dense_Feature_Space::
add_feature(const string & name, const Feature_Info & info)
{
    int index = feature_index(name);
    
    if (index == -1) {
        /* Doesn't exist.  We add it. */
        index = variable_count();
        names_fwd.push_back(name);
        names_bwd[name] = index;
        info_array.push_back(info);
        features_.push_back(Feature(index));
    }
    else {
        Feature_Info & current = info_array[index];
        current = promote(current, info);

        cerr << "warning: adding feature " << name
             << " to dense feature face that already contains it; "
             << "they are being mapped onto the same feature" << endl;
            
#if 0
        /* Already exists.  Check that the info is compatible. */
        if (info_array[index] != info)
            throw Exception("Dense_Feature_Space::add_feature(): feature '"
                            + name + "' already exists with info '"
                            + info_array[index].print()
                            + "'; tried to add with incompatible info '"
                            + info.print() + "'.");
#endif
    }
    return index;
}

void
Dense_Feature_Space::
add(const Dense_Feature_Space & other_fs,
    const std::string & name_prefix)
{
    for (unsigned i = 0;  i < other_fs.names_fwd.size();  ++i)
        add_feature(name_prefix + other_fs.names_fwd[i],
                    other_fs.info_array[i]);
}

std::shared_ptr<Mutable_Categorical_Info>
Dense_Feature_Space::
make_categorical(Feature feature)
{
    unsigned var = feature.type();
    if (var >= info_array.size())
        throw Exception("make_categorical: bad feature");

    info_array[var].make_categorical();
    return info_array[var].mutable_categorical();
}

std::string Dense_Feature_Space::class_id() const
{
    return "DENSE_FEATURE_SPACE";
}

void Dense_Feature_Space::serialize(DB::Store_Writer & store) const
{
    store << string("DENSE_FEATURE_SPACE") << compact_size_t(0)
          << names_fwd << info_array;
}

void Dense_Feature_Space::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & feature_space)
{
    //cerr << "dense_feature_space reconstitution" << endl;
    string type;
    compact_size_t version;
    store >> type >> version;
    if (type != "DENSE_FEATURE_SPACE")
        throw Exception("Attempt to reconstitute " + type + "; expected "
                        "DENSE_FEATURE_SPACE");
    if (version != 0)
        throw Exception(format("Attemp to reconstitute dense fs version %zd, "
                               "only <= %d supported", version.size_,
                               0));

    store >> names_fwd >> info_array;

    //cerr << "done reconstitution" << endl;

    set_feature_names(names_fwd);

    //cerr << "done set_feature_names()" << endl;

    features_.clear();
    for (unsigned i = 0;  i < names_fwd.size();  ++i)
        features_.push_back(Feature(i));
}

void Dense_Feature_Space::
reconstitute(DB::Store_Reader & store)
{
    //cerr << "dense_feature_space reconstitution" << endl;
    string type;
    compact_size_t version;
    store >> type >> version;
    if (type != "DENSE_FEATURE_SPACE")
        throw Exception("Attempt to reconstitute " + type + "; expected "
                        "DENSE_FEATURE_SPACE");
    if (version != 0)
        throw Exception(format("Attemp to reconstitute dense fs version %zd, "
                               "only <= %d supported", version.size_,
                               0));

    store >> names_fwd >> info_array;

    set_feature_names(names_fwd);

    features_.clear();
    for (unsigned i = 0;  i < names_fwd.size();  ++i)
        features_.push_back(Feature(i));
}

void
Dense_Feature_Space::
freeze()
{
    //cerr << "freezing dense_feature_space" << endl;
    for (unsigned i = 0;  i < info_array.size();  ++i)
        info_array[i].freeze();
}


/*****************************************************************************/
/* DENSE_TRAINING_DATA                                                       */
/*****************************************************************************/

Dense_Training_Data::Dense_Training_Data()
{
}

Dense_Training_Data::
Dense_Training_Data(const std::string & filename)
{
    init(filename);
}

Dense_Training_Data::
Dense_Training_Data(const std::string & filename,
                    std::shared_ptr<Dense_Feature_Space> feature_space)
{
    init(filename, feature_space);
}

Dense_Training_Data::~Dense_Training_Data()
{
}

namespace {

template<class Contained>
struct Buffer {
    Contained * begin_;
    Contained * end_;
};

boost::tuple<size_t, size_t, string>
get_sizes(Parse_Context & context,
          vector<unsigned> & row_start_ofs)
{
    row_start_ofs.clear();


    /* Parse the header. */
    string header = context.expect_line("expected dataset header");

    unsigned var_count = 0;

    if (context.eof()) return boost::make_tuple(0, 0, header);  // empty file

    //cerr << "header = " << header << endl;

    /* Parse the first data line.  This tells us how many variables we
       have. */
    while (context && (!context.match_eol() || var_count == 0)) {
        row_start_ofs.push_back(context.get_offset());

        if (context.match_literal('#')) {
            //cerr << "matched comment" << endl;
            context.skip_line();
            if (var_count == 0) continue;  // skip lines with just comments
            break;
        }
        else if (var_count == 0 && context.match_eol()) {
            //cerr << "matched eol" << endl;
            continue;
        }

        string got = context.expect_text(" \t\n");
        //cerr << "got val \"" << got << "\"" << endl;
        context.skip_whitespace();
        ++var_count;
    }

    if (var_count == 0) return boost::make_tuple(0, 0, header);  // empty file

    /* Count the number of rows.  This is simply just skipping lines until
       we are finished. */
    unsigned row_count = 1;  // already have the first one...
    while (context) {
        context.skip_whitespace();
        if (context.match_literal('#')) {
            context.skip_line();
            continue;
        }
        if (context.match_eol()) continue;
        context.skip_line();
        ++row_count;
    }
    
    return boost::make_tuple(row_count, var_count, header);
}

boost::tuple<size_t, size_t, string>
get_sizes(const std::string & filename,
          vector<unsigned> & row_start_ofs)
{
    Parse_Context context(filename);

    return get_sizes(context, row_start_ofs);
}

} // file scope

void Dense_Training_Data::
add_data()
{
    /* Go through the dataset and turn each row into an example. */
    Training_Data::clear();
    Training_Data::init(feature_space());
    
    int nv = dataset.shape()[1];
    int nx = dataset.shape()[0];

    /* Feature vector */
    std::shared_ptr<vector<Feature> >
        feature_vec(new vector<Feature>(nv));
    for (unsigned j = 0;  j < nv;  ++j)
        (*feature_vec)[j] = Feature(j);
    
    /* Add data */
    for (unsigned x = 0;  x < nx;  ++x) {
        std::shared_ptr<Dense_Feature_Set> features
            (new Dense_Feature_Set(feature_vec, &dataset[x][0]));
        //cerr << "got row " << feature_space()->print(*features) << endl;
        add_example(features);
    }
}

void Dense_Training_Data::
init(const std::string & filename)
{
    std::shared_ptr<Dense_Feature_Space> fs(new Dense_Feature_Space());
    init(filename, fs);
}

void
Dense_Training_Data::
init(const char * data, const char * data_end)
{
    std::shared_ptr<Dense_Feature_Space> fs(new Dense_Feature_Space());
    init(data, data_end, fs);
}

void
Dense_Training_Data::
init(const std::string & filename,
     std::shared_ptr<Dense_Feature_Space> feature_space)
{
    vector<string> filenames(1, filename);
    init(filenames, feature_space);
}

struct Dense_Training_Data::Data_Source {
    Data_Source(const std::string & filename,
                const char * data = 0,
                const char * data_end = 0)
        : filename(filename), data(data), data_end(data_end)
    {
    }

    std::string filename;
    const char * data;
    const char * data_end;
    mutable std::shared_ptr<filter_istream> stream;

    Parse_Context get_context() const
    {
        if (data)
            return Parse_Context(filename, data, data_end);
        else {
            stream.reset(new filter_istream(filename));
            return Parse_Context(filename, *stream);
        }
    }
};

void
Dense_Training_Data::
init(const std::vector<std::string> & filenames,
     std::shared_ptr<Dense_Feature_Space> feature_space)
{
    vector<Data_Source> data_sources;
    for (unsigned i = 0;  i < filenames.size();  ++i)
        data_sources.push_back(Data_Source(filenames[i]));
    init(data_sources, feature_space);
}

void
Dense_Training_Data::
init(const char * data, const char * data_end,
     std::shared_ptr<Dense_Feature_Space> feature_space)
{
    vector<Data_Source> data_sources;
    data_sources.push_back(Data_Source("inline data", data, data_end));
    init(data_sources, feature_space);
}

void
Dense_Training_Data::
init(const std::vector<Data_Source> & data_sources,
     std::shared_ptr<Dense_Feature_Space> feature_space)
{
    Training_Data::init(feature_space);

    /* Get all of the counts from all of the files. */
    size_t row_count = 0, var_count = 0;
    string header;

    vector<unsigned> row_counts(data_sources.size());
    vector<vector<unsigned> > row_start_ofs(data_sources.size());

    if (feature_space->variable_count() != 0)
        var_count = feature_space->variable_count();

    int first_nonempty = -1;

    for (unsigned i = 0;  i < data_sources.size();  ++i) {

        size_t my_row_count, my_var_count;
        string my_header;

        Parse_Context context = data_sources[i].get_context();

        boost::tie(my_row_count, my_var_count, my_header)
            = get_sizes(context, row_start_ofs[i]);

        cerr << "file: " << data_sources[i].filename
             << " rows: " << my_row_count
             << " vars: " << my_var_count << endl;

        //cerr << "count vars: " << timer.elapsed() << "s" << endl;

        if (first_nonempty == -1 && my_row_count != 0)
            first_nonempty = i;

        if (var_count == 0)
            var_count = my_var_count;
        else {
            if (my_var_count != 0 && my_var_count != var_count)
                throw Exception
                    (format("Dense_Training_Data::init(): wrong number of "
                            "variables: %zd expected, %zd found in dataset '%s'",
                            var_count, my_var_count,
                            data_sources[i].filename.c_str()));
        }

        if (header == "")
            header = my_header;
        else if (my_header != header)
            throw Exception("headers don't match");
        
        row_counts[i] = my_row_count;
        row_count += my_row_count;
    }

    //cerr << "loading " << data_sources << endl;

    /* Whether or not each feature_info is immutable. */
    vector<bool> immutable_features;
    
    //cerr << "feature space is " << feature_space->print() << endl;
    //cerr << "header is " << header << endl;

    /* Construct the feature space, if it's not already done. */
    if (feature_space->variable_count() == 0) {
        //cerr << "constructing feature space" << endl;

        vector<string> feature_names;
        vector<Mutable_Feature_Info> feature_info;

        //cerr << "parsing header " << header << endl;

        Parse_Context context(data_sources[0].filename, header.c_str(),
                              header.c_str() + header.size());

        while (context) {
            context.skip_whitespace();
            if (!context) break;
            string name = expect_feature_name(context);

            //cerr << "name = " << name << endl;

            feature_names.push_back(name);
            if (context.match_literal(':')) {
                /* Get the feature info from the feature. */
                Mutable_Feature_Info info;
                info.parse(context);
                feature_info.push_back(info);
                immutable_features.push_back(true);
            }
            else {
                feature_info.push_back(Feature_Info(UNKNOWN));
                immutable_features.push_back(false);
            }
        }

        if (feature_names.size() != var_count && row_count > 0) {
            throw Exception
                (format("variable counts (%zd) and header counts (%zd) "
                        "don't match", feature_names.size(), var_count));
        }

        feature_space->init(feature_names, feature_info);
    }
    else {
        // TODO: check that headers are OK
        immutable_features.resize(var_count, true);
    }

    //cerr << "done feature space init" << endl;
    //cerr << "feature space is " << feature_space->print() << endl;
    
    /* Allocate our array. */
    dataset.resize(boost::extents[row_count][var_count]);
    row_comments.resize(row_count);
    row_offsets.resize(row_count);

    //cerr << "dataset.shape()[0] = " << dataset.shape()[0] << endl;
    //cerr << "dataset.shape()[1] = " << dataset.shape()[1] << endl;

    //cerr << "row_count = " << row_count << " var_count = "
    //     << var_count << endl;

    if (first_nonempty == -1)
        return;  // nothing to load

    //cerr << "first_nonempty = " << first_nonempty << endl;

    /* Find if any are already known to be categorical. */
    vector<std::shared_ptr<Mutable_Categorical_Info> >
        categorical(var_count);
    
    for (unsigned v = 0;  v < var_count;  ++v)
        categorical[v] = feature_space->info_array[v].mutable_categorical();

    /* Did we guess wrongly about which are categorical?  If so, we need to
       reparse the array.  We keep on going until we are right about which
       are categorical. */
    bool guessed_wrong = false;

    boost::timer timer;
    
    /* Keep on going until we get all the categorical values correct. */
    do {
        //cerr << "trying to read..." << endl;
        for (unsigned i = first_nonempty;  i < data_sources.size();  ++i) {

            Parse_Context context = data_sources[i].get_context();
            
            /* Skip over the header */
            context.skip_line();
            
            /* Now fill it in value by value. */
            int row = 0;
            while (row < row_counts[i]) {
                //cerr << "row " << row << " of " << row_counts[i] << endl;

                //__builtin_prefetch(context.get_pos() + 256, 0, 0);
                
                context.skip_whitespace();
                if (context.match_literal('#')) {
                    context.skip_line();
                    continue;
                }
                
                if (context.match_eol()) continue;

                row_offsets[row] = context.get_offset();
                
                //cerr << "row " << row << endl;

                for (unsigned v = 0;  v < var_count;  ++v) {
                    
                    //cerr << "v = " << v << endl;
                    //cerr << "info = " << feature_space->info_array[v] << endl;
                    //cerr << "categorical = " << categorical[v] << endl;

                    __builtin_prefetch(&dataset[row][v] + 64, 1, 0);
                    
                    if (!categorical[v] && !immutable_features[v]) {
                        if (match_float(dataset[row][v], context)) ;
                        else {
                            if (row != 0) guessed_wrong = true;
                            categorical[v]
                                = feature_space
                                ->make_categorical(Feature(v));
                        }
                    }
                    else if (!categorical[v]) {
                        /* immutable; float must parse */
                        dataset[row][v] = expect_float<float>(context);
                    }
                    
                    if (categorical[v]) {
                        string category = context.expect_text(" \n");

                        //cerr << "v = " << v << " type = "
                        //     << feature_space->info_array[v].type()
                        //     << " category = " << category << " info = "
                        //     << feature_space->info_array[v] << endl;

                        float value;
                        if (immutable_features[v]
                            && feature_space->info_array[v].type()
                               != STRING) 
                            value = categorical[v]->parse(category);
                        else value = categorical[v]->parse_or_add(category);
                        
                        dataset[row][v] = value;
                    }
                    
                    context.skip_whitespace();
                }
                if (context.match_literal('#')) {
                    /* end of line comment; keep it for if we want to rewrite */
                    row_comments[row] = context.expect_line();
                }
                else
                    context.expect_eol("too many values in line (expected EOL)");
                ++row;
            }
            
            while (context) {
                context.skip_whitespace();
                if (context.match_literal('#')) {
                    //cerr << "skipping comment..." << endl;
                    context.skip_line();
                continue;
                }
                
                if (context.match_eol()) {
                    continue;
                }

                //cerr << "row = " << row << " row_count = " << row_count
                //     << endl;
                
                context.exception("extra junk after end of file");
            }
        
            context.expect_eof();
        }

        if (guessed_wrong)
            throw Exception("guessed wrong");

    } while (guessed_wrong);

    //cerr << "read files: " << timer.elapsed() << "s" << endl;

    //timer.restart();
    add_data();
    //cerr << "add_data: " << timer.elapsed() << "s" << endl;

    //if (example_count()) {
    //    cerr << "got dataset" << endl;
    //    dump(cerr);
    //}
}

Dense_Training_Data * Dense_Training_Data::make_copy() const
{
    return new Dense_Training_Data(*this);
}

Dense_Training_Data * Dense_Training_Data::make_type() const
{
    return new Dense_Training_Data();
}

size_t
Dense_Training_Data::
row_offset(size_t row) const
{
    if (row > row_offsets.size())
        throw Exception("Dense_Training_Data::row_offset(): "
                        "row out of range");
    return row_offsets[row];
}

std::string
Dense_Training_Data::
row_comment(size_t row) const
{
    if (row >= row_comments.size())
        throw Exception("Dense_Training_Data::row_comment(): "
                        "row out of range");
    return row_comments[row];
}

float
Dense_Training_Data::
modify_feature(int example_number,
               const Feature & feature,
               float new_value)
{
    //return Training_Data::modify_feature(example_number, feature, new_value);

    if (feature.type() < 0 || feature.type() >= example_count())
        throw Exception("can't add feature to dense dataset");

    float & val = dataset[example_number][feature.type()];
    float result = val;
    val = new_value;
    return result;
}


/*****************************************************************************/
/* DENSE_FEATURE_SET                                                         */
/*****************************************************************************/

Dense_Feature_Set * Dense_Feature_Set::make_copy() const
{
    return new Dense_Feature_Set(*this);
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

Register_Factory<Feature_Space, Dense_Feature_Space>
DFS_REG("DENSE_FEATURE_SPACE");


} // namespace ML

