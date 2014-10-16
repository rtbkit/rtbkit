/* sparse_feature_space.h                                           -*- C++ -*-
   Jeremy Barnes, 12 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   A feature space based upon sparse features.
*/

#ifndef __boosting__sparse_features_h__
#define __boosting__sparse_features_h__


#include "feature_set.h"
#include "training_data.h"
#include <map>
#include "jml/utils/hash_map.h"
#include <string>



namespace ML {

        
/*****************************************************************************/
/* SPARSE_FEATURE_SPACE                                                      */
/*****************************************************************************/

/** This is a feature space that has a fixed number of sparse variables, and
    always returns a vector of this number of real features for each feature
    set.  It is essentially an adaptor around the "file-o-data" feature
    representation, used by the more classical algorithms.
*/

class Sparse_Feature_Space : public ML::Mutable_Feature_Space {
public:
    Sparse_Feature_Space();
    Sparse_Feature_Space(DB::Store_Reader & store);
    virtual ~Sparse_Feature_Space();

    /** Initialise */
    void init();

    /** Change the info for a given feature. */
    virtual void set_info(const Feature & feature, const Feature_Info & info);
    
    /** Methods to do with the features. */
    virtual Feature_Info info(const ML::Feature & feature) const;
    virtual std::string print(const ML::Feature & feature) const;
    virtual bool parse(Parse_Context & context, ML::Feature & feature) const;
    virtual void expect(Parse_Context & context, ML::Feature & feature) const;
    virtual void serialize(DB::Store_Writer & store,
                           const ML::Feature & feature) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              ML::Feature & feature) const;

    /* Methods to deal with the feature space as a whole. */
    virtual Sparse_Feature_Space * make_copy() const;

    /** Serialization and reconstitution. */
    virtual std::string class_id() const;
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store,
                              const std::shared_ptr<const Feature_Space>
                                  & feature_space);
    virtual void reconstitute(DB::Store_Reader & store);

    virtual Feature
    make_feature(const std::string & name,
                 const Feature_Info & info = UNKNOWN);

    /** Return the feature with the given name.  Throws if the name is
        unknown. */
    virtual Feature get_feature(const std::string & name) const;

    virtual Type type() const { return SPARSE; }

    using Feature_Space::serialize;
    using Feature_Space::reconstitute;
    using Feature_Space::print;
    using Feature_Space::parse;

protected:    
    typedef std::hash_map<std::string, int> string_lookup_type;
    mutable string_lookup_type string_lookup;

    mutable std::vector<Mutable_Feature_Info> info_array;
    mutable std::vector<std::string> name_array;

    /** Return the name of the given feature.  Throws if the feature is
        unknown. */
    std::string get_name(const Feature & feature) const;

    friend class Sparse_Training_Data;
};


/*****************************************************************************/
/* SPARSE_TRAINING_DATA                                                      */
/*****************************************************************************/

/** The training data component.  Its main utility is that it can read a
    file of a very large number of very sparse features, and happily work
    with it.

    The format of the data file is:

    \verbatim
    file         ::= line*
    line         ::= ['#' comment] '\n' | feature* ['#' comment] '\n'
    feature      ::= tag | tag ':' value_list | tag ':' value_list
    value_list   ::= value [',' value_list]
    value        ::= <real number> | category
    category     ::= tag
    tag          ::= quote_string | escaped_string
    escaped_str  ::= <string with '\', ' ', ':', '"' and '|' escaped with '\'>
    quoted_str   ::= '"' <string with '"' escaped with '\'> '"'
    \endverbatim
    
    so:

    "word-1=hello", "word-1:hello,hell", "0:12.7" and
    "0:12.7,0.93" are all valid.
*/

class Sparse_Training_Data : public ML::Training_Data {
public:
    /** Default do-nothing constructor.  Requires that init() be called
        afterwards. */
    Sparse_Training_Data();

    /** Initialise from a filename.  Loads it into a dataset. */
    Sparse_Training_Data(const std::string & filename);

    /** Initialise from a filename, using the given feature space. */
    Sparse_Training_Data(const std::string & filename,
                         const std::shared_ptr<Sparse_Feature_Space> & fs);

    virtual ~Sparse_Training_Data();


    /** Initialise from a data file.  A new feature space will be
        constructed. */
    void init(const std::string & filename);

    /** Initialise from a data file.  The given feature space will be used,
        and modified as the dataset is read. */
    void init(const std::string & filename,
              std::shared_ptr<Sparse_Feature_Space> fs);

    /** Initialise from a set of data files.  The given feature space will be
        used, and modified as the dataset is read. */
    void init(const std::vector<std::string> & filename,
              std::shared_ptr<Sparse_Feature_Space> fs);

    /** Polymorphic copy. */
    virtual Sparse_Training_Data * make_copy() const;

    /** Polymorphic construct.  Makes another object of the same type, but
        doesn't populate it. */
    virtual Sparse_Training_Data * make_type() const;

private:
    void expect_feature(Parse_Context & c, Mutable_Feature_Set & features,
                        Sparse_Feature_Space & feature_space,
                        bool & guessed_wrong);

    std::shared_ptr<Sparse_Feature_Space> sparse_fs;
};


} // namespace ML



#endif /* __boosting__sparse_features_h__ */
