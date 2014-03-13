/* segments.h                                                      -*- C++ -*-
   Structure that holds a list of segments.
*/

#pragma once

#include "jml/utils/compact_vector.h"
#include "jml/db/persistent_fwd.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/value_description.h"
#include "soa/types/value_description_fwd.h"
#include <boost/shared_ptr.hpp>
#include <map>


namespace RTBKIT {


/// Result of querying for segment presence
enum SegmentResult {
    SEG_NOT_PRESENT,   ///< Segment is not present
    SEG_PRESENT,       ///< Segment is present
    SEG_MISSING        ///< Segment is missing
};

/*****************************************************************************/
/* SEGMENTS                                                                  */
/*****************************************************************************/

/** A set of integral "segments".
    Immutable once created.
*/

struct SegmentList {
    SegmentList();
    SegmentList(const std::vector<std::string> & segs);
    SegmentList(const std::vector<int> & segs);
    SegmentList(const std::vector<std::pair<int, float> > & segs);

    bool contains(int i) const;
    bool contains(const std::string & str) const;

    //float weight(int i) const;
    //float weight(const std::string & str) const;

    bool match(const SegmentList & other) const;
    bool match(const std::vector<int> & other) const;
    bool match(const std::vector<std::string> & other) const;

    size_t size() const;
    bool empty() const;

    /* JSON */
    
    static SegmentList createFromJson(const Json::Value & json);
    Json::Value toJson() const;
    std::string toJsonStr() const;
    std::string toString() const;

    /* Mutation */

    // add(...) does not sort the segments. Should be done manually.
    void add(int i, float weight = 1.0);
    void add(const std::string & str, float weight = 1.0);

    void sort();

    static int parseSegmentNum(const std::string & str);

    /** Return true if there are only integers in the list. */
    bool intsOnly() const { return strings.empty(); }

    /** Iterate over all segments and call the given callback. */
    void forEach(const std::function<void (int, std::string, float)> & onSegment)
        const;

    //private:    
    ML::compact_vector<int, 7> ints;          ///< Categories
    std::vector<std::string> strings;         ///< Those that aren't an integer
    ML::compact_vector<float, 5> weights;     ///< Weights over ints and strings
    
    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
    std::string serializeToString() const;
    static SegmentList reconstituteFromString(const std::string & str);
};

IMPL_SERIALIZE_RECONSTITUTE(SegmentList);

inline std::ostream &
operator << (std::ostream & stream, const SegmentList & segs)
{
    return stream << segs.toString();
}


/*****************************************************************************/
/* SEGMENTS BY SOURCE                                                        */
/*****************************************************************************/

typedef std::map<std::string, std::shared_ptr<SegmentList> >
SegmentsBySourceBase;

/** A set of segments per segment provider. */

struct SegmentsBySource
    : public SegmentsBySourceBase {

    SegmentsBySource();

    SegmentsBySource(SegmentsBySourceBase && other);
    SegmentsBySource(const SegmentsBySourceBase & other);

    const SegmentList & get(const std::string & str) const;

    void sortAll();
    
    void add(const std::string & source,
             const std::shared_ptr<SegmentList> & segs)
    {
        addSegment(source, segs);
    }

    void addSegment(const std::string & source,
                    const std::shared_ptr<SegmentList> & segs);
    void addInts(const std::string & source,
                 const std::vector<int> & segs);
    void addWeightedInts(const std::string & source,
                         const std::vector<std::pair<int, float> > & segs);
    void addStrings(const std::string & source,
                    const std::vector<std::string> & segs);

    /** Add the given segment to the given source, creating if it didn't
        exist already.
    */
    void add(const std::string & source, const std::string & segment,
                float weight = 1.0);

    /** Add the given segment to the given source, creating if it didn't
        exist already.
    */
    void add(const std::string & source, int segment,
             float weight = 1.0);

    Json::Value toJson() const;

    static SegmentsBySource createFromJson(const Json::Value & json);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string serializeToString() const;
    static SegmentsBySource reconstituteFromString(const std::string & str);
};

IMPL_SERIALIZE_RECONSTITUTE(SegmentsBySource);

} // namespace RTBKIT


namespace Datacratic {

using namespace RTBKIT;

template<>
struct DefaultDescription<SegmentList>
    : public ValueDescriptionI<SegmentList, ValueKind::ARRAY> {

    virtual void parseJsonTyped(SegmentList * val,
                                JsonParsingContext & context) const;
    virtual void printJsonTyped(const SegmentList * val,
                                JsonPrintingContext & context) const;
    virtual bool isDefaultTyped(const SegmentList * val) const;
};

template<>
struct DefaultDescription<SegmentsBySource>
    : public ValueDescriptionI<SegmentsBySource, ValueKind::MAP> {
    DefaultDescription(ValueDescriptionT<SegmentList> * newInner
                       = getDefaultDescription((SegmentList *)0));

    ValueDescriptionT<SegmentList> * inner;

    virtual void parseJsonTyped(SegmentsBySource * val,
                                JsonParsingContext & context) const;
    virtual void printJsonTyped(const SegmentsBySource * val,
                                JsonPrintingContext & context) const;
    virtual bool isDefaultTyped(const SegmentsBySource * val) const;
};

}

