/* feature_info.cc
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of the feature info class.
*/

#include "feature_info.h"
#include "training_data.h"
#include "training_index.h"
#include "registry.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;


namespace ML {

using namespace DB;


const Feature_Info MISSING_FEATURE_INFO(INUTILE);
const Feature_Info MISSING_INFO(INUTILE);



/*****************************************************************************/
/* FEATURE_INFO::TYPE                                                        */
/*****************************************************************************/

std::string print(Feature_Type type)
{
    switch (type) {
    case UNKNOWN:       return "UNKNOWN";
    case PRESENCE:      return "PRESENCE";
    case BOOLEAN:       return "BOOLEAN";
    case CATEGORICAL:   return "CATEGORICAL";
    case REAL:          return "REAL";
    case INUTILE:       return "INUTILE";
    case STRING:        return "STRING";
    default:
        throw Exception("Unknown Feature_Info");
    }
}

std::ostream &
operator << (std::ostream & stream, Feature_Type type)
{
    return stream << ML::print(type);
}


/*****************************************************************************/
/* CATEGORICAL_INFO                                                          */
/*****************************************************************************/

unsigned Categorical_Info::parse(const std::string & name) const
{
    int code = lookup(name);
    if (code == -1) {
        char * name_end = 0;
        code = strtol(name.c_str(), &name_end, 10);

        if (*name_end != 0 || code < 0 || code >= count())
            throw Exception("couldn't parse name " + name);
    }
    return code;
}

void Categorical_Info::
poly_serialize(DB::Store_Writer & store, const Categorical_Info & info)
{
    Registry<Categorical_Info>::singleton().serialize(store, &info);
}

std::shared_ptr<Categorical_Info>
Categorical_Info::poly_reconstitute(DB::Store_Reader & store)
{
    return Registry<Categorical_Info>::singleton().reconstitute(store);
}


/*****************************************************************************/
/* FEATURE_INFO                                                              */
/*****************************************************************************/

Feature_Info::
Feature_Info(Type type, bool optional, bool biased, bool grouping)
    : type_(type), optional_(optional), biased_(biased), grouping_(grouping)
{
}

Feature_Info::
Feature_Info(std::shared_ptr<const Categorical_Info> categorical,
             bool optional, bool biased, Type type, bool grouping)
    : type_(type), optional_(optional), biased_(biased),
      grouping_(grouping), categorical_(categorical)
{
}

namespace {

static const unsigned char FI_VERSION = 4;

struct FI_Flags {
    union {
        // Endian safe since only one byte
        struct {
            uint8_t type:4;
            uint8_t optional:1;
            uint8_t biased:1;
            uint8_t grouping:1;
            uint8_t has_cat:1;
        };
        uint8_t bits;
    };
};

} // file scope


COMPACT_PERSISTENT_ENUM_IMPL(Feature_Type);

void Feature_Info::serialize(DB::Store_Writer & store) const
{
    FI_Flags flags;
    flags.type = type_;
    flags.optional = optional_;
    flags.biased = biased_;
    flags.grouping = grouping_;
    flags.has_cat = (categorical_ && type_ != STRING);
    
    store << FI_VERSION << flags.bits;
    if (flags.has_cat)
        Categorical_Info::poly_serialize(store, *categorical_);
}

void Feature_Info::reconstitute(DB::Store_Reader & store)
{
    compact_size_t version(store);

    if (version == 0) {
        compact_size_t type(store);
        type_ = type;
        unsigned cat;
        store >>  cat;
        optional_ = false;
        if (cat && type_ == CATEGORICAL)
            categorical_.reset(new Fixed_Categorical_Info(cat));
    }
    else if (version == 1) {
        compact_size_t type(store);
        type_ = type;
        compact_size_t opt(store), cat(store);
        optional_ = opt;
        if (cat && type_ == CATEGORICAL)
            categorical_.reset(new Fixed_Categorical_Info(cat));
    }
    else if (version == 2 || version == 3) {
        compact_size_t type(store);
        type_ = type;

        compact_size_t opt(store);
        optional_ = opt;

        if (version == 3) {
            unsigned char b;
            store >> b;
            biased_ = b;
        }
        else biased_ = false;

        compact_size_t cat_there(store);
        if (cat_there) {
            std::shared_ptr<Categorical_Info> ci
                = Categorical_Info::poly_reconstitute(store);
            categorical_ = ci;
            mutable_categorical_
                = std::dynamic_pointer_cast<Mutable_Categorical_Info>(ci);
        }
        else if (type_ == STRING) {
            mutable_categorical_.reset(new Mutable_Categorical_Info());
            categorical_ = mutable_categorical_;
        }
        
        grouping_ = false;
    }
    else if (version == 4) {

        FI_Flags flags;

        store >> flags.bits;
        type_ = flags.type;
        optional_ = flags.optional;
        biased_ = flags.biased;
        grouping_ = flags.grouping;

        if (flags.has_cat) {
            std::shared_ptr<Categorical_Info> ci
                = Categorical_Info::poly_reconstitute(store);
            categorical_ = ci;
            mutable_categorical_
                = std::dynamic_pointer_cast<Mutable_Categorical_Info>(ci);
        }
        else if (type_ == STRING) {
            mutable_categorical_.reset(new Mutable_Categorical_Info());
            categorical_ = mutable_categorical_;
        }
    }
    else 
        throw Exception
            (format("Attempt to reconstitute Feature_Info with too high "
                    "a version number (%zd vs %d)", version.size_,
                    FI_VERSION));
}

bool Feature_Info::operator == (const Feature_Info & other) const
{
    return type_ == other.type_ && categorical_ == other.categorical_
        && optional_ == other.optional_ && biased_ == other.biased_
        && grouping_ == other.grouping_;
}

bool Feature_Info::operator != (const Feature_Info & other) const
{
    return ! operator == (other);
}

std::string Feature_Info::print() const
{
    string result = "k=" + ML::print(type());
    if (type() == CATEGORICAL)
        result += "/c=" + categorical()->print();
    if (optional()) result += "/o=OPTIONAL";
    if (biased()) result += "/o=BIASED";
    if (grouping()) result += "/o=GROUPING";
    return result;
}

namespace {

std::string expect_category_name(Parse_Context & context)
{
    std::string result;
    
    bool after_slash = false;
    while (after_slash || (context && !isspace(*context)
                           && *context != ','
                           && *context != '/')) {
        if (after_slash) {
            result += *context++;
            after_slash = false;
        }
        else if (context.match_literal('\\')) after_slash = true;
        else if (!(*context) || isspace(*context) || *context == ',') break;
        else result += *context++;
    }
    
    return result;
}

} // file scope

void Mutable_Feature_Info::parse(Parse_Context & context)
{
    type_ = UNKNOWN;
    biased_ = false;
    optional_ = false;
    grouping_ = false;

    while (context && !(isspace(*context))) {
        switch (*context) {
        case 'k': {
            ++context;
            context.expect_literal('=');
            /* Feature kind */
            if (context.match_literal("UNKNOWN"))
                type_ = UNKNOWN;
            else if (context.match_literal("PRESENCE"))
                type_ = PRESENCE;
            else if (context.match_literal("BOOLEAN"))
                type_ = BOOLEAN;
            else if (context.match_literal("CATEGORICAL"))
                type_ = CATEGORICAL;
            else if (context.match_literal("STRING"))
                set_categorical(make_sp(new Mutable_Categorical_Info()), STRING);
            else if (context.match_literal("REAL"))
                type_ = REAL;
            else if (context.match_literal("UNUTILE"))
                type_ = INUTILE;
            else context.exception("Feature_Info::parse(): unknown type");
            break;
        }
        case 'c': {
            if (type_ != CATEGORICAL)
                throw Exception("Feature_Info::parse(): "
                                "categories with non-categorical feature");
            
            /* Categories */
            ++context;
            context.expect_literal('=');
            
            /* First we have a number of categories, then a list of values. */
            unsigned num_cat = context.expect_unsigned();
            
            if (num_cat > 1000)
                context.exception("num_cat: too many categories (>1000)");
            
            //cerr << "num_cat = " << num_cat << endl;

            vector<string> cat_names(num_cat);
            
            for (unsigned i = 0;  i < num_cat;  ++i) {
                //cerr << "i = " << i << endl;
                context.expect_literal(',');
                cat_names[i] = expect_category_name(context);
                //cerr << "  name = " << cat_names[i] << endl;
            }
            
            set_categorical(make_sp(new Mutable_Categorical_Info(cat_names)));
            break;
        }
        case 'o': {
            /* Option */
            ++context;
            context.expect_literal('=');
            if (context.match_literal("BIASED")) biased_ = true;
            else if (context.match_literal("OPTIONAL")) optional_ = true;
            else if (context.match_literal("GROUPING")) grouping_ = true;
           else context.exception("Feature_Info::parse(): unknown option");
            break;
        }
        default:
            context.exception("Feature_Info::parse(): unknown option char "
                              + string(1, *context));
        }

        if (!context.match_literal('/')) break;
    }
}

size_t Feature_Info::value_count() const
{
    switch (type_) {
    case UNKNOWN:       return 0;
    case PRESENCE:      return 1;
    case BOOLEAN:       return 2;
    case CATEGORICAL:   // fall through
    case STRING:        return categorical_ ? categorical_->count() : 1;
    case REAL:          return 1;
    case INUTILE:       return 0;
    default:
        throw Exception("Feature_Info::value_count(): invalid type passed");
    }
}

#if 0
bool
Feature_Info::
enumeratable() const
{
    switch (type_) {
    case UNKNOWN:       return false;
    case PRESENCE:      return true;
    case BOOLEAN:       return true;
    case CATEGORICAL:   return true;
    case STRING:        return true;
    case REAL:          return false;
    case INUTILE:       return false;
    default:
        throw Exception("Feature_Info::enumeratable(): invalid type passed");
    }
}
#endif

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Feature_Info & info)
{
    info.reconstitute(store);
    return store;
}

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Feature_Info & info)
{
    info.serialize(store);
    return store;
}

std::ostream &
operator << (std::ostream & stream, const Feature_Info & info)
{
    return stream << info.print();
}

Feature_Info promote(const Feature_Info & i1, const Feature_Info & i2)
{
    static const Feature_Type UNKN = UNKNOWN;
    static const Feature_Type PRES = PRESENCE;
    static const Feature_Type BOOL = BOOLEAN;
    static const Feature_Type CAT  = CATEGORICAL;
    static const Feature_Type INUT = INUTILE;
    static const Feature_Type STR  = STRING;
    
    static const Feature_Type lookup[8][8] = {
        { UNKN, PRES, BOOL, CAT,  REAL, UNKN, INUT, STR }, // UNKNOWN
        { PRES, PRES, BOOL, CAT,  REAL, UNKN, PRES, STR }, // PRESENCE
        { BOOL, BOOL, BOOL, CAT,  REAL, UNKN, BOOL, STR }, // BOOLEAN
        { CAT,  CAT,  CAT,  CAT,  REAL, UNKN, CAT,  STR }, // CATEGORICAL
        { REAL, REAL, REAL, REAL, REAL, UNKN, REAL, STR }, // REAL
        { UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, UNKN, STR }, // UNUSED1
        { INUT, PRES, BOOL, CAT,  REAL, UNKN, INUT, STR }, // INUTILE
        { STR,  STR,  STR,  STR,  STR,  UNKN, STR,  STR }  // STRING
    };
    
    /* TODO: map categorical/string features. */

    if (i1.type() == STRING)
        return i1;
    else if (i2.type() == STRING)
        return i2;

    if (i1.type() == CATEGORICAL)
        return i1;
    else if (i2.type() == CATEGORICAL)
        return i2;

    if (i1.type() < 0 || i1.type() >= 8 || i2.type() < 0 || i2.type() >= 8)
        throw Exception("promote(): bad feature info passed");
    
    Feature_Info result(lookup[i1.type()][i2.type()],
                        i1.optional() || i2.optional(),
                        i1.biased() || i2.biased(),
                        i1.grouping() || i2.grouping());

    return result;
}

/** Guess the feature type, based upon its training data. */
Feature_Info
guess_info(const Training_Data & data, const Feature & feature,
           const Feature_Info & before)
{
    /* Use the training data to do the job. */
    Feature_Info guessed = data.index().guess_info(feature);
    Feature_Info result = promote(before, guessed);
    return result;
}

void guess_all_info(const Training_Data & data,
                    Mutable_Feature_Space & fs, bool use_existing)
{
    /* Try to extract info for these features. */
    vector<Feature> all_features = data.all_features();
    for (unsigned i = 0;  i < all_features.size();  ++i) {
        const Feature & feat = all_features[i];
        Feature_Info before
            = use_existing ? fs.info(feat) : UNKNOWN;
        Feature_Info info = guess_info(data, feat, before);
        fs.set_info(feat, info);
    }
}


/*****************************************************************************/
/* MUTABLE_FEATURE_INFO                                                      */
/*****************************************************************************/

Mutable_Feature_Info::
Mutable_Feature_Info(const Feature_Info & info)
    : Feature_Info(info)
{
    //if (categorical_) {
    //    mutable_categorical_.reset(new Mutable_Categorical_Info(*categorical_));
    //    categorical_ = mutable_categorical_;
    //}
}

Mutable_Feature_Info::
Mutable_Feature_Info(Type type, bool optional)
    : Feature_Info(type, optional)
{
}

Mutable_Feature_Info::
Mutable_Feature_Info(std::shared_ptr<Mutable_Categorical_Info> categorical,
                     bool optional, Type type)
    : Feature_Info(categorical, optional, false, type)
{
    mutable_categorical_ = categorical;
}

void
Mutable_Feature_Info::
reconstitute(DB::Store_Reader & store)
{
    Feature_Info info;
    info.reconstitute(store);
    *this = Mutable_Feature_Info(info);
}

void
Mutable_Feature_Info::
make_categorical(Type type)
{
    type_ = type;
    mutable_categorical_.reset(new Mutable_Categorical_Info());
    categorical_ = mutable_categorical_;
}

void
Mutable_Feature_Info::
set_categorical(std::shared_ptr<Mutable_Categorical_Info> info,
                Type type)
{
    mutable_categorical_ = info;
    categorical_ = info;

    if (info) type_ = type;
}

void
Mutable_Feature_Info::
set_categorical(Mutable_Categorical_Info * info,
                Type type)
{
    set_categorical(make_sp(info), type);
}

void
Mutable_Feature_Info::
set_type(Type type)
{
    if (type == CATEGORICAL || type == STRING) {
        set_categorical(new Mutable_Categorical_Info(), type);
    }
    else {
        type_ = type;
        categorical_.reset();
        mutable_categorical_.reset();
    }
    
}

void
Mutable_Feature_Info::
set_optional(bool optional)
{
    optional_ = optional;
}

void
Mutable_Feature_Info::
set_biased(bool biased)
{
    biased_ = biased;
}

void
Mutable_Feature_Info::
set_grouping(bool grouping)
{
    grouping_ = grouping;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Mutable_Feature_Info & info)
{
    info.reconstitute(store);
    return store;
}

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Mutable_Feature_Info & info)
{
    info.serialize(store);
    return store;
}

void
Mutable_Feature_Info::
freeze()
{
#if 0
    if (type_ == STRING) {
        cerr << "freezing mutable feature info " << print() << endl;
        cerr << "mutable_categorical_ = " << mutable_categorical_ << endl;
        cerr << "categorical_ = " << categorical_ << endl;
        cerr << "type " << demangle(typeid(*categorical_).name()) << endl;
    }
#endif
    if (mutable_categorical_)
        mutable_categorical_->freeze();
}


/*****************************************************************************/
/* FIXED_CATEGORICAL_INFO                                                    */
/*****************************************************************************/

Fixed_Categorical_Info::
Fixed_Categorical_Info()
    : is_mutable(false)
{
}

Fixed_Categorical_Info::
Fixed_Categorical_Info(unsigned num)
    : is_mutable(false)
{
    for (unsigned i = 0;  i < num;  ++i)
        print_.push_back(format("VAL%d", i));
    make_parse_from_print();
}

Fixed_Categorical_Info::
Fixed_Categorical_Info(const std::vector<std::string> & names)
    : is_mutable(false), print_(names)
{
    make_parse_from_print();
}

Fixed_Categorical_Info::
Fixed_Categorical_Info(DB::Store_Reader & store)
    : is_mutable(false)
{
    reconstitute(store);
}

Fixed_Categorical_Info::~Fixed_Categorical_Info()
{
}

namespace {

/* Escape categorical info.  We have to escape any commas or spaces with
   backslashes.
*/
string escape_categorical_info(const std::string & value)
{
    string result;
    for (unsigned i = 0;  i < value.size();  ++i) {
        if (isspace(value[i])
            || value[i] == ',' || value[i] == '/' || value[i] == '\\')
            result += '\\';
        result += value[i];
    }
    return result;
}

} // file scope

string
Fixed_Categorical_Info::
print() const
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    string result = format("%zd", print_.size());
    for (unsigned i = 0;  i < print_.size();  ++i)
        result += "," + escape_categorical_info(print_[i]);
    return result;
}

std::string Fixed_Categorical_Info::print(int value) const
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    if (value < 0 || value >= print_.size()) {
        cerr << "value = " << value << endl;
        cerr << "size  = " << print_.size() << endl;
        //throw Exception("Fixed_Categorical_Info::print(): out of range");
        cout << "Fixed_Categorical_Info::print(): out of range" << endl;
        return "";
        //return format("invalid(%d, size %zd)", value, print_.size());
    }

    return print_.at(value);
}

int Fixed_Categorical_Info::lookup(const std::string & name) const
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    /* Get the value of the category */
    std::hash_map<std::string, int>::const_iterator it = parse_.find(name);

    if (it == parse_.end()) return -1;
    
    return it->second;
}


compact_size_t FIXED_CI_VERSION(0);


void Fixed_Categorical_Info::serialize(DB::Store_Writer & store) const
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    store << FIXED_CI_VERSION << string("FIXED_CI");
    store << print_;
}

void Fixed_Categorical_Info::reconstitute(DB::Store_Reader & store)
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    compact_size_t version(store);

    if (version == 0) {
        string name;
        store >> name;
        if (name != "FIXED_CI")
            throw Exception("Fixed_Categorical_Info: reconstituting unknown "
                            "name");
        store >> print_;
        make_parse_from_print();
    }
    else throw Exception("Fixed_Categorical_Info: reconstituting unknown "
                         "version");
}

std::string Fixed_Categorical_Info::class_id() const
{
    return "FIXED_CI";
}

unsigned Fixed_Categorical_Info::count() const
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    return print_.size();
}

void Fixed_Categorical_Info::make_parse_from_print()
{
    Guard guard(lock, std::defer_lock);  // guard that doesn't acquire the lock
    if (is_mutable) guard.lock();

    parse_.clear();
    for (unsigned i = 0;  i < print_.size();  ++i)
        parse_[print_[i]] = i;
}

void
Fixed_Categorical_Info::
freeze()
{
}


/*****************************************************************************/
/* MUTABLE_CATEGORICAL_INFO                                                  */
/*****************************************************************************/

Mutable_Categorical_Info::
Mutable_Categorical_Info()
    : Fixed_Categorical_Info(0), frozen(false)
{
    is_mutable = true;
}

Mutable_Categorical_Info::
Mutable_Categorical_Info(const std::vector<std::string> & names)
    : Fixed_Categorical_Info(names), frozen(false)
{
    is_mutable = true;
}

Mutable_Categorical_Info::
Mutable_Categorical_Info(unsigned num)
    : Fixed_Categorical_Info(num), frozen(false)
{
    is_mutable = true;
}

Mutable_Categorical_Info::
Mutable_Categorical_Info(DB::Store_Reader & store)
    : Fixed_Categorical_Info(0), frozen(false)
{
    reconstitute(store);
    is_mutable = true;
}

Mutable_Categorical_Info::
Mutable_Categorical_Info(const Categorical_Info & other)
    : Fixed_Categorical_Info(0), frozen(false)
{
    int cnt = other.count();

    for (unsigned i = 0;  i < cnt;  ++i)
        parse_or_add(other.print(i));

    is_mutable = true;
}

int
Mutable_Categorical_Info::
parse_or_add(const std::string & name) const
{
    if (frozen)
        throw Exception("Mutable_Categorical_Info::parse_or_add(): "
                        "frozen");

    Guard guard(lock);

    /* Get the value of the category */
    std::hash_map<std::string, int>::const_iterator it = parse_.find(name);
    
    /* Get or insert the name */
    if (it == parse_.end()) {
        it = parse_.insert(std::make_pair(name, parse_.size())).first;
        print_.push_back(name);
    }
    
    return it->second;
}

int
Mutable_Categorical_Info::
lookup(const std::string & name) const
{
    Guard guard(lock);
    if (!frozen) return parse_or_add(name);
    else return Fixed_Categorical_Info::lookup(name);
}

void
Mutable_Categorical_Info::
freeze()
{
    //cerr << "freezing Mutable_Categorical_Info with values "
    //     << print_ << endl;

    if (frozen) return;
    Guard guard(lock);
    frozen = true;
    is_mutable = false;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/


namespace {

Register_Factory<Categorical_Info, Fixed_Categorical_Info>
    FIXED_CI_REGISTER("FIXED_CI");


} // file scope


/*****************************************************************************/
/* FIXED_CATEGORICAL_MAPPING                                                 */
/*****************************************************************************/

Fixed_Categorical_Mapping::
Fixed_Categorical_Mapping(std::shared_ptr<const Categorical_Info> info1,
                          std::shared_ptr<const Categorical_Info> info2)
{
    const Categorical_Info & ci = *info1;
    const Categorical_Info & oci = *info2;

    /* How many categories are there? */
    int nc = ci.count();
    mapping.resize(nc);

    /* Fill in with the mapping, or with -1 if it is not found
       (-1 is returned by lookup if not found). */
    for (unsigned c = 0;  c < nc;  ++c) {
        string name = ci.print(c);
        int index = oci.lookup(name);
        mapping[c] = index;
    }
}

Fixed_Categorical_Mapping::
~Fixed_Categorical_Mapping()
{
}

int
Fixed_Categorical_Mapping::
map(int value,
    const Categorical_Info & info1,
    const Categorical_Info & info2) const
{
    if (value > mapping.size() || value < 0) return -1;
    //throw Exception("Fixed_Categorical_Mapping::map(): unknown value");
    return mapping[value];
}


/*****************************************************************************/
/* DYNAMIC_CATEGORICAL_MAPPING                                               */
/*****************************************************************************/

/** A dynamic mapping that simply keeps two Categorical_Info objects and
    maps the objects as it goes.  Used when they might change.
*/

Dynamic_Categorical_Mapping::
Dynamic_Categorical_Mapping()
{
}

Dynamic_Categorical_Mapping::
~Dynamic_Categorical_Mapping()
{
}

int
Dynamic_Categorical_Mapping::
map(int value,
    const Categorical_Info & info1,
    const Categorical_Info & info2) const
{
#if 0
    cerr << "info1: " << &info1 << " " << demangle(typeid(info1).name()) << endl;
    cerr << "info2: " << &info2 << " " << demangle(typeid(info2).name()) << endl;

    for (unsigned i = 0;  i < info1.count();  ++i) {
        cerr << "info1[" << i << "] = " << info1.print(i) << endl;
    }

    for (unsigned i = 0;  i < info2.count();  ++i) {
        cerr << "info2[" << i << "] = " << info2.print(i) << endl;
    }
#endif

    //cerr << "value = " << value;
    string s = info2.print(value);
    //cerr << " string = " << s;
    int result = info1.lookup(s);
    //cerr << " result = " << result << endl;
    return result;
}

} // namespace ML


