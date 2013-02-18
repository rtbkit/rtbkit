/* sparse_features.cc
   Jeremy Barnes, 25 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of a sparse feature space.
*/

#include "sparse_features.h"
#include "registry.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/file_functions.h"
#include <boost/utility.hpp>
#include "config_impl.h"


using namespace std;
using namespace DB;



namespace ML {


/*****************************************************************************/
/* SPARSE_FEATURE_SPACE                                                      */
/*****************************************************************************/

Sparse_Feature_Space::Sparse_Feature_Space()
{
}

Sparse_Feature_Space::Sparse_Feature_Space(DB::Store_Reader & store)
{
    reconstitute(store);
}

Sparse_Feature_Space::~Sparse_Feature_Space()
{
}

void Sparse_Feature_Space::init()
{
}

void Sparse_Feature_Space::
set_info(const Feature & feature, const Feature_Info & info)
{
    if (feature == MISSING_FEATURE)
        throw Exception("attempt to set feature info for missing feature");
    //cerr << this << endl;
    //cerr << "setting info for feature " << feature << endl;
    //cerr << "string_lookup.size() = " << string_lookup.size() << endl;
    Mutable_Feature_Info & infoa = info_array.at(feature.type());
    infoa = info;
}
    
Feature_Info
Sparse_Feature_Space::
info(const ML::Feature & feature) const
{
    if (feature == MISSING_FEATURE)
        return MISSING_FEATURE_INFO;

    int id = feature.type();
    if (id < 0 || id >= info_array.size())
        throw Exception("Feature with unknown ID " + ostream_format(id)
                        + " requested from Sparse_Feature_Space: "
                        + feature.print());

    return info_array[id];
}

std::string Sparse_Feature_Space::
print(const ML::Feature & feature) const
{
    if (feature == MISSING_FEATURE)
        return "MISSING";
    return get_name(feature);
}

bool Sparse_Feature_Space::
parse(Parse_Context & context, ML::Feature & feature) const
{
    throw Exception("unimplemented");
}

void Sparse_Feature_Space::
expect(Parse_Context & context, ML::Feature & feature) const
{
    throw Exception("unimplemented");
}

void Sparse_Feature_Space::
serialize(DB::Store_Writer & store, const ML::Feature & feature) const
{
    store << get_name(feature);
}

void Sparse_Feature_Space::
reconstitute(DB::Store_Reader & store, ML::Feature & feature) const
{
    std::string name;
    store >> name;
    feature = get_feature(name);
}

Sparse_Feature_Space * Sparse_Feature_Space::make_copy() const
{
    return new Sparse_Feature_Space(*this);
}

namespace {
static const std::string ID = "SPARSE_FEATURE_SPACE";
static const compact_size_t VERSION(0);
} // file scope

std::string Sparse_Feature_Space::class_id() const
{
    return ID;
}

void Sparse_Feature_Space::serialize(DB::Store_Writer & store) const
{
    store << ID << VERSION;
}

void Sparse_Feature_Space::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & feature_space)
{
    string type;
    compact_size_t version;
    store >> type >> version;
    if (type != ID)
        throw Exception("Attempt to reconstitute " + type + "; expected "
                        + ID);
    if (version > VERSION)
        throw Exception(format("Attemp to reconstitute sparse fs version %zd, "
                               "only <= %d supported", version.size_,
                               0));
}

void Sparse_Feature_Space::
reconstitute(DB::Store_Reader & store)
{
    //cerr << "sparse_feature_space reconstitution" << endl;
    string type;
    compact_size_t version;
    store >> type >> version;
    if (type != ID)
        throw Exception("Attempt to reconstitute " + type + "; expected "
                        + ID);
    if (version > VERSION)
        throw Exception(format("Attemp to reconstitute sparse fs version %zd, "
                               "only <= %d supported", version.size_,
                               0));
}

Feature
Sparse_Feature_Space::
make_feature(const std::string & name, const Feature_Info & info)
{
    //cerr << "make_feature(\"" << name << "\")" << endl;

    string_lookup_type::const_iterator it = string_lookup.find(name);
    if (it == string_lookup.end()) {
        string_lookup[name] = info_array.size();
        info_array.push_back(info);
        name_array.push_back(name);
        return Feature(info_array.size() - 1);
    }

    return Feature(it->second);
}

std::string Sparse_Feature_Space::get_name(const Feature & feature) const
{
    int id = feature.type();
    if (id < 0 || id >= name_array.size())
        throw Exception("Feature with unknown ID " + ostream_format(id)
                        + " requested from Sparse_Feature_Space: "
                        + feature.print());

    return name_array[id];
}

Feature Sparse_Feature_Space::get_feature(const std::string & name) const
{
    string_lookup_type::const_iterator it = string_lookup.find(name);
    if (it == string_lookup.end()) {
        return const_cast<Sparse_Feature_Space *>(this)->make_feature(name);
        //throw Exception("Feature with name " + name + " not found in "
        //                "Sparse_Feature_Space");
    }
    return Feature(it->second);
}


/*****************************************************************************/
/* SPARSE_TRAINING_DATA                                                      */
/*****************************************************************************/

Sparse_Training_Data::Sparse_Training_Data()
{
}

Sparse_Training_Data::
Sparse_Training_Data(const std::string & filename)
{
    init(filename);
}

Sparse_Training_Data::
Sparse_Training_Data(const std::string & filename,
                     const std::shared_ptr<Sparse_Feature_Space> & fs)
{
    init(filename, fs);
}

Sparse_Training_Data::~Sparse_Training_Data()
{
}

void
Sparse_Training_Data::
expect_feature(Parse_Context & c, Mutable_Feature_Set & features,
               Sparse_Feature_Space & feature_space,
               bool & guessed_wrong)
{
    //__builtin_prefetch(c.get_pos() + 128, 0, 0);

    string name = expect_feature_name(c);
    //cerr << "name = " << name << endl;
    Feature feature = feature_space.make_feature(name);
    //cerr << "feature = " << feature << endl;
    int num = feature.type();

    //cerr << "num = " << num << endl;

    Mutable_Feature_Info & info
        = feature_space.info_array[num];

    bool categorical = !!info.categorical();

    if (categorical && (!c || *c != ':')) {
        /* A categorical feature can't have a missing label. */
        c.exception("categorical feature " + name + " missing value");
    }
                                   
    if (!categorical) {
        float value = 1.0;
        
        if (c && *c == ':') {
            /* It has a value attached. */
            ++c;
            if (c.match_float(value))
                features.add(feature, value);
            else {
                /* We guessed wrong... this is categorical */
                cerr << "guessed_wrong" << endl;
                cerr << "name = " << name << endl;
                cerr << "line = " << c.get_line() << endl;
                guessed_wrong = true;
                info.make_categorical();
                categorical = true;
            }

            /* Allow further values to be specified. */
            while (!categorical && c && *c == ',') {
                ++c;  // skip the ','
                if (c.match_float(value))
                    features.add(feature, value);
                else {
                    guessed_wrong = true;
                    info.make_categorical();
                    categorical = true;
                }
            }
        }
        else if (!c || *c == '\n' || isspace(*c)) {
            /* Feature with defaulted value */
            features.add(feature, 1.0);
        }
    }
    
    if (categorical) {
        if (c && c.match_literal(':')) {
            /* It has a value attached. */
            string category = expect_feature_name(c);
            float value = info.mutable_categorical()->parse_or_add(category);
            features.add(feature, value);
        }
        
        /* Allow further values to be specified. */
        while (c && c.match_literal(',')) {
            string category = expect_feature_name(c);
            float value = info.mutable_categorical()->parse_or_add(category);
            features.add(feature, value);
        }
    }
}

namespace {

bool match_label(Parse_Context & c, Mutable_Feature_Set & features,
                 Sparse_Feature_Space & feature_space)
{
    /* If the first part is numeric and doesn't include a ":", we assume that
       it's a label.  This is for compatibility with old-format files.

       The new files will have LABEL:xxx, so this will not be necessary.
    */
    float val;
    if (c.match_float(val) && isspace(*c)) {
        features.add(feature_space.make_feature("LABEL"), val);
        return true;
    }
    return false;
}

} // file scope

void Sparse_Training_Data::
init(const std::string & filename,
     std::shared_ptr<Sparse_Feature_Space> feature_space)
{
    vector<string> filenames(1, filename);
    init(filenames, feature_space);
}

void Sparse_Training_Data::
init(const std::vector<std::string> & filenames,
     std::shared_ptr<Sparse_Feature_Space> feature_space)
{
    //cerr << "sparse data init from file" << endl;

    Training_Data::init(feature_space);
    sparse_fs = feature_space;

    /* Did we guess wrongly about which are categorical?  If so, we need to
       reparse the array.  We keep on going until we are right about which
       are categorical. */
    bool guessed_wrong = false;
    
    /* Keep on going until we get all the categorical values correct. */
    do {
        guessed_wrong = false;

        clear();

        for (unsigned i = 0;  i < filenames.size();  ++i) {
            string filename = filenames[i];
            //cerr << "loading data from " << filename << endl;
            /* We load the training data from the file. */
            File_Read_Buffer file(filename);
            
            Parse_Context c(filename, file.start(), file.end());
            
            /* Skip over the header. */
            c.skip_line();
            
            while (c) {
                std::shared_ptr<Mutable_Feature_Set>
                    features(new Mutable_Feature_Set());
                
                try {
                    c.skip_whitespace();
                    if (c.match_eol()) continue;   // skip blank lines
                    if (*c == '#') {
                        c.skip_line();  // skip comments
                        continue;
                    }
                    
                    /* Allow a label to be specified, with the implicit feature
                       name "LABEL", in the first position. */
                    if (match_label(c, *features, *feature_space))
                        c.skip_whitespace();
                    
                    while (!c.match_eol()) {
                        if (*c == '#') {
                            // skip comments after the data
                            c.skip_line();
                            break;
                        }
                        else {
                            expect_feature(c, *features, *feature_space,
                                           guessed_wrong);
                            c.skip_whitespace();
                        }
                    }
                }
                catch (const Exception & exc) {
                    cerr << "warning: " << exc.what() << ": skipping line"
                         << endl;
                    c.skip_line();
                }

                //cerr << "features are " << feature_space->print(*features) << endl;
                
                add_example(features);
            }
        }

    } while (guessed_wrong);
}

void Sparse_Training_Data::
init(const std::string & filename)
{
    init(filename,
         std::shared_ptr<Sparse_Feature_Space>(new Sparse_Feature_Space()));
}

Sparse_Training_Data * Sparse_Training_Data::make_copy() const
{
    return new Sparse_Training_Data(*this);
}

Sparse_Training_Data * Sparse_Training_Data::make_type() const
{
    Sparse_Training_Data * result = new Sparse_Training_Data();
    result->sparse_fs = sparse_fs;
    return result;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {
Register_Factory<Feature_Space, Sparse_Feature_Space> SFS_REG(ID);
} // file scope



} // namespace ML

