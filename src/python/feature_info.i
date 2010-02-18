/* feature_set.i                                                   -*- C++ -*-
   Jeremy Barnes, 24 September 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   SWIG wrapper for the Feature_Set class.
*/

%module jml 
%{
#include "jml/boosting/feature_info.h"
%}

%include "std_string.i"
%include "boost_shared_ptr.i"

%template(svector) std::vector<std::string>;

namespace ML {


class Feature_Space;
class Mutable_Feature_Space;
class Training_Data;
class Categorical_Info;
class Mutable_Categorical_Info;


/*****************************************************************************/
/* CATEGORICAL_INFO                                                          */
/*****************************************************************************/

/** This structure tells us how we encode and decode categorical features.
    It is split off from the main Feature_Info structure.
*/
struct Categorical_Info {

    virtual ~Categorical_Info() {}

    /** Print the entire set of categories. */
    virtual std::string print() const = 0;
    
    /** Print the given value. */
    virtual std::string print(int value) const = 0;

    /** Parse the given value, returning -1 if not found. */
    virtual int lookup(const std::string & value) const = 0;

    /** Parse the given value, throwing an exception if not found.  Default
        is implemented in terms of lookup. */
    virtual unsigned parse(const std::string & value) const;

    /** Return the number of possible categories. */
    virtual unsigned count() const = 0;

    /** Serialize to a store. */
    virtual void serialize(DB::Store_Writer & store) const = 0;

    /** Reconstitute from a store. */
    virtual void reconstitute(DB::Store_Reader & store) = 0;

    /** Return the name of this class (for serialization). */
    virtual std::string class_id() const = 0;

    /** Freeze it so that it can no longer grow. */
    virtual void freeze() = 0;

    /** Serialize in such a manner to allow polymorphic reconstitution. */
    static void poly_serialize(DB::Store_Writer & store,
                               const Categorical_Info & info);

    /** Reconstitute polymorphically a Categorical_Info from a store. */
    static boost::shared_ptr<Categorical_Info>
    poly_reconstitute(DB::Store_Reader & store);

    %extend {
        std::string __str__() const
        {
            return $self->print();
        }

        std::string __repr__() const
        {
            return $self->print();
        }
    }
};

} // namespace ML

SWIG_SHARED_PTR(ML::Categorical_Info_Ptr, ML::Categorical_Info);

namespace ML {

/*****************************************************************************/
/* FEATURE_TYPE                                                              */
/*****************************************************************************/

/** Encodes the type of the feature, which in turn encodes how the
    learning algorithms attempt to learn rules for the algorithm.
*/
enum Feature_Type {
    UNKNOWN,      ///< we have not yet determined the feature type
    PRESENCE,     ///< feature is present or not present; value unimportant
    BOOLEAN,      ///< feature is true (1.0) or false (0.0)
    CATEGORICAL,  ///< feature is categorical; ordering makes no sense
    REAL,         ///< feature is real valued
    UNUSED1,      ///< Was PROB
    INUTILE,      ///< feature is inutile and should be ignored
    STRING        ///< feature is an open categorical feature
};


/*****************************************************************************/
/* FEATURE_INFO                                                              */
/*****************************************************************************/

/** This class provides information on a single feature.  This is the minimum
    amount that the algorithms need to do their job.
*/

struct Feature_Info {
public:
    /** Initialise for one of the non-categorical types. */
    Feature_Info(Feature_Type type = REAL, bool optional = false, bool biased = false,
                 bool grouping = false);

    /** Initialise for a categorical feature info. */
    Feature_Info(boost::shared_ptr<const Categorical_Info> categorical,
                 bool optional = false, bool biased = false,
                 Feature_Type type = CATEGORICAL, bool grouping = false);
    
    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);

    /** Allow testing for equality. @{ */
    bool operator == (const Feature_Info & other) const;
    bool operator != (const Feature_Info & other) const;
    //@}

    /** Print in in ASCII format.  This can be parsed later. */
    std::string print() const;

    /** Return the number of distinct values for this feature.  Returns
        0 for real features (which take an infinite number of values). */
    size_t value_count() const;

    Feature_Type type() const { return (Feature_Type)type_; }

    boost::shared_ptr<const Categorical_Info> categorical() const
    {
        return categorical_;
    }

    /** If true, then nothing should be inferred from the absence of this
        feature from a dataset. */
    bool optional() const { return optional_; }

    /** If true, then this feature is biased (contains some outside information
        about the thing being measured, for example the label variable) and
        should not be learned from. */
    bool biased() const { return biased_; }

    /** If true, this feature is used to group parts of datasets together, and
        it can be adjusted so that it will be strictly increasing over the
        dataset. */
    bool grouping() const { return grouping_; }

    %extend {
        std::string __str__() const
        {
            return $self->print();
        }

        std::string __repr__() const
        {
            return $self->print();
        }
    }
};


std::string print(Feature_Type type);

extern const Feature_Info MISSING_FEATURE_INFO;


/** Guess the feature type, based upon its training data. */
Feature_Info
guess_info(const Training_Data & data,
           const Feature & feat,
           const Feature_Info & before = UNKNOWN);

/** Return the most inclusive of the two feature info values.  Used when two
    have been automatically detected over different datasets, to get the
    real (combined) feature info.
*/
Feature_Info promote(const Feature_Info & i1, const Feature_Info & i2);


/*****************************************************************************/
/* FIXED_CATEGORICAL_INFO                                                    */
/*****************************************************************************/

struct Fixed_Categorical_Info : public Categorical_Info {
public:
    /** Default construct.  For when we will reconstitute after. */
    Fixed_Categorical_Info();

    /** Construct a bogus list of the given length. */
    Fixed_Categorical_Info(unsigned num);
    
    /** Construct from a list of names. */
    Fixed_Categorical_Info(const std::vector<std::string> & names);

    /** Reconstitute from a store. */
    Fixed_Categorical_Info(DB::Store_Reader & store);

    virtual ~Fixed_Categorical_Info();

    virtual std::string print() const;
    virtual std::string print(int value) const;
    virtual int lookup(const std::string & value) const;
    virtual unsigned count() const;
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);
    virtual std::string class_id() const;
    virtual void freeze();

    %extend {
        std::string __str__() const
        {
            return $self->print();
        }

        std::string __repr__() const
        {
            return $self->print();
        }
    }
};


/*****************************************************************************/
/* MUTABLE_CATEGORICAL_INFO                                                  */
/*****************************************************************************/

struct Mutable_Categorical_Info : public Fixed_Categorical_Info {
public:
    /** Construct an empty list */
    Mutable_Categorical_Info();

    /** Construct from a list of names. */
    Mutable_Categorical_Info(const std::vector<std::string> & names);

    /** Construct a bogus list of names. */
    Mutable_Categorical_Info(unsigned num);

    /** Reconstitute from a store. */
    Mutable_Categorical_Info(DB::Store_Reader & store);

    /** Copy another Categorical_Info object */
    Mutable_Categorical_Info(const Categorical_Info & other);

    /** Either parse (if it is already there) or add (if not) the given name
        to the internal structures. */
    int parse_or_add(const std::string & name) const;

    virtual int lookup(const std::string & value) const;

    virtual void freeze();

    bool frozen;

    %extend {
        std::string __str__() const
        {
            return $self->print();
        }

        std::string __repr__() const
        {
            return $self->print();
        }
    }
};

} // namespace ML


SWIG_SHARED_PTR_DERIVED(Fixed_Categorical_Info_Ptr, ML::Categorical_Info, ML::Fixed_Categorical_Info)
SWIG_SHARED_PTR_DERIVED(Mutable_Categorical_Info_Ptr, ML::Categorical_Info, ML::Mutable_Categorical_Info)

namespace ML {

/*****************************************************************************/
/* MUTABLE_FEATURE_INFO                                                      */
/*****************************************************************************/

/** Same as Feature_Info, but mutable. */

struct Mutable_Feature_Info : public Feature_Info {

    /** Initalize from a Feature_Info object. */
    Mutable_Feature_Info(const Feature_Info & info);

    /** Initialise for one of the non-categorical types. */
    Mutable_Feature_Info(Feature_Type type = REAL, bool optional = false);

    /** Initialise for a categorical feature info. */
    Mutable_Feature_Info(boost::shared_ptr<Mutable_Categorical_Info> categorical,
                         bool optional = false,
                         Feature_Type type = CATEGORICAL /* or STRING */);

    void reconstitute(DB::Store_Reader & store);

    /** Turn a non-categorical feature info into a categorical one. */
    void make_categorical(Feature_Type type = CATEGORICAL);

    /** Set the categorical info. */
    void set_categorical(boost::shared_ptr<Mutable_Categorical_Info> info,
                         Feature_Type type = CATEGORICAL);

    /** Set the categorical info. */
    void set_categorical(Mutable_Categorical_Info * info,
                         Feature_Type type = CATEGORICAL);
    
    boost::shared_ptr<Mutable_Categorical_Info> mutable_categorical() const
    {
        if (categorical_ != mutable_categorical_)
            throw Exception("Mutable_Feature_Info::categorical(): out of sync");
        return mutable_categorical_;
    }

    /** Set the feature type. */
    void set_type(Feature_Type type);

    /** Set the optional flag. */
    void set_optional(bool optional);

    /** Set the biased flag. */
    void set_biased(bool biased);

    /** Set the grouping flag. */
    void set_grouping(bool grouping);

    /** Parse from a text file. */
    void parse(Parse_Context & context);

    /** Stop it from growing. */
    void freeze();

    %extend {
        std::string __str__() const
        {
            return $self->print();
        }

        std::string __repr__() const
        {
            return $self->print();
        }
    }
};

/** Guess the feature info for each of the features, and modify the
    given feature space to reflect this.  Requires that finish() has
    already been called.
*/
void guess_all_info(const Training_Data & data,
                    Mutable_Feature_Space & fs, bool use_existing);


} // namespace ML
