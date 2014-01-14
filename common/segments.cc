/* segments.cc
   Jeremy Barnes, 12 March 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Implementation of segments.
*/

#include "rtbkit/common/segments.h"
#include <boost/function_output_iterator.hpp>
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/exc_assert.h"
#include "soa/types/value_description.h"
#include "jml/db/persistent.h"
#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace ML;
using namespace ML::DB;
using namespace Datacratic;

namespace Datacratic {

void
DefaultDescription<SegmentList>::
parseJsonTyped(SegmentList * val, JsonParsingContext & context)
    const
{
    Json::Value v = context.expectJson();
    //cerr << "got segments " << v << endl;
    *val = std::move(SegmentList::createFromJson(v));
}

void
DefaultDescription<SegmentList>::
printJsonTyped(const SegmentList * val, JsonPrintingContext & context)
    const
{
    context.startArray(val->ints.size() + val->strings.size());
    if (val->weights.empty()) {
        for (unsigned i = 0;  i < val->ints.size();  ++i) {
            context.newArrayElement();
            context.writeInt(val->ints[i]);
        }
        for (unsigned i = 0;  i < val->strings.size();  ++i) {
            context.newArrayElement();
            context.writeString(val->strings[i]);
        }
    }
    else if (val->weights.size() == val->ints.size())
    {
        for (unsigned i = 0;  i < val->ints.size();  ++i) {
            context.newArrayElement();
            context.writeString(to_string(val->ints[i])+":"+to_string(val->weights[i]));
        }
    }
    else if (val->weights.size() == val->strings.size())
    {
        for (unsigned i = 0;  i < val->strings.size();  ++i) {
            context.newArrayElement();
            context.writeString(val->strings[i]+":"+to_string(val->weights[i]));
        }
    }
    else {
        throw ML::Exception("weights unsupported");
    }
    context.endArray();
}

bool
DefaultDescription<SegmentList>::
isDefaultTyped(const SegmentList * val)
    const
{
    return val->empty();
}

DefaultDescription<SegmentsBySource>::
DefaultDescription(ValueDescriptionT<SegmentList> * newInner)
    : inner(newInner)
{
    // inner = reinterpret_cast<DefaultDescription<SegmentList> *>(newInner);
}

void
DefaultDescription<SegmentsBySource>::
parseJsonTyped(SegmentsBySource * val, JsonParsingContext & context)
    const
{
    Json::Value v = context.expectJson();
    //cerr << "got segments " << v << endl;
    *val = std::move(RTBKIT::SegmentsBySource::createFromJson(v));
}

void
DefaultDescription<SegmentsBySource>::
printJsonTyped(const SegmentsBySource * val,
               JsonPrintingContext & context) const
{
    context.startObject();
    for (const auto & v: *val) {
        context.startMember(v.first);
        inner->printJsonTyped(v.second.get(), context);
    }
    context.endObject();
}

bool
DefaultDescription<SegmentsBySource>::
isDefaultTyped(const SegmentsBySource * val)
    const
{
    return val->empty();
}

}

namespace RTBKIT {


/*****************************************************************************/
/* SEGMENTS                                                                  */
/*****************************************************************************/

SegmentList::
SegmentList()
{
}

SegmentList::
SegmentList(const std::vector<string> & segs)
{
    for (unsigned i = 0;  i < segs.size();  ++i)
        add(segs[i]);
    sort();
}

SegmentList::
SegmentList(const std::vector<int> & segs)
    : ints(segs.begin(), segs.end())
{
    sort();
}

SegmentList::
SegmentList(const std::vector<std::pair<int, float> > & segs)
{
    for (unsigned i = 0;  i < segs.size();  ++i)
        add(segs[i].first, segs[i].second);
    sort();
}

bool
SegmentList::
contains(int i) const
{
    return std::binary_search(ints.begin(), ints.end(), i);
}

bool
SegmentList::
contains(const std::string & str) const
{
    int i = parseSegmentNum(str);
    if (i == -1)
        return std::binary_search(strings.begin(), strings.end(), str);
    else return contains(i);
}

#if 0
float
SegmentList::
weight(int i) const
{
}

float
SegmentList::
weight(const std::string & str) const
{
}
#endif

template<typename Seq1, typename Seq2>
bool anyMatchesLookup(const Seq1 & seq1, const Seq2 & seq2)
{
    auto it2 = seq2.begin(), end2 = seq2.end();
    for (auto it1 = seq1.begin(), end1 = seq1.end();
         it1 != end1;  ++it1)
        if (std::binary_search(it2, end2, *it1)) return true;
    return false;
}

template<typename Seq1, typename Seq2>
bool anyMatches(const Seq1 & seq1, const Seq2 & seq2)
{
    if (seq1.empty() || seq2.empty())
        return false;
    else if (seq1.size() * 5 < seq2.size()) {
        // seq2 is much bigger... look up individually each element
        return anyMatchesLookup(seq1, seq2);
    }
    else if (seq2.size() * 5 < seq1.size()) {
        // seq1 is much bigger... look up individually each element
        return anyMatchesLookup(seq2, seq1);
    }
    else {
        // roughly equal sizes; jointly iterate
        auto it1 = seq1.begin(), end1 = seq1.end();
        auto it2 = seq2.begin(), end2 = seq2.end();
    
        while (it1 != end1 && it2 != end2) {
            if (*it1 == *it2) return true;
            else if (*it1 < *it2) ++it1;
            else ++it2;
        }

        return false;
    }
}

bool
SegmentList::
match(const SegmentList & other) const
{
    return anyMatches(ints, other.ints)
        || anyMatches(strings, other.strings);
}

bool
SegmentList::
match(const std::vector<int> & other) const
{
    return anyMatches(ints, other);
}

bool
SegmentList::
match(const std::vector<std::string> & other) const
{
    return anyMatches(strings, other);
}

size_t
SegmentList::
size() const
{
    return ints.size() + strings.size();
}

bool
SegmentList::
empty() const
{
    return ints.empty() && strings.empty();
}

SegmentList
SegmentList::
createFromJson(const Json::Value & json)
{
    SegmentList result;

    if (!json.isArray())
        throw Exception("augment must be an array of augmentations");

    for (unsigned i = 0;  i < json.size();  ++i) {
        const Json::Value & val = json[i];
        if (val.isArray()) {
            if (val.size() != 2)
                throw ML::Exception("can't create weighted segment from "
                                    + json.toString());
            float weight = val[1].asDouble();
            
            if (val[0].isInt())
                result.add(val[0].asInt(), weight);
            else if (val[0].isNumeric())
                result.add(val[0].asDouble(), weight);
            else result.add(val[0].asString(), weight);
        }
        else if (val.isInt())
            result.add(val.asInt());
        else if (val.isNumeric())
            result.add(val.asDouble());
        else result.add(val.asString());
    }

    result.sort();
    return result;
}

Json::Value
SegmentList::
toJson() const
{
    Json::Value result(Json::arrayValue);

    if (weights.empty()) {
        if (strings.empty()) {
            for (unsigned i = 0;  i < ints.size();  ++i)
                result[i] = ints[i];
        }
        else {
            for (unsigned i = 0;  i < ints.size();  ++i)
                result[i] = ML::format("%d", ints[i]);
            for (unsigned i = 0;  i < strings.size();  ++i)
                result[i + ints.size()] = strings[i];
        }
    }
    else {
        if (strings.empty()) {
            for (unsigned i = 0;  i < ints.size();  ++i) {
                result[i][0] = ints[i];
                result[i][1] = weights[i];
            }
        }
        else {
            for (unsigned i = 0;  i < ints.size();  ++i) {
                result[i][0] = ML::format("%d", ints[i]);
                result[i][1] = weights[i];
            }
            for (unsigned i = 0;  i < strings.size();  ++i) {
                result[i + ints.size()][0] = strings[i];
                result[i + ints.size()][1] = weights[i + ints.size()];
            }
        }
    }
    return result;
}

std::string
SegmentList::
toJsonStr() const
{
    return boost::trim_copy(toJson().toString());
}

std::string
SegmentList::
toString() const
{
    return toJsonStr();
}

void
SegmentList::
add(int i, float weight)
{
    ints.push_back(i);
    if (weight != 1.0 || !weights.empty()) {
        if (weights.empty())
            weights.resize(size() - 1, 1.0);
        weights.insert(weights.begin() + ints.size() - 1, weight);
        ExcAssertEqual(weights.size(), size());
    }
}

void
SegmentList::
add(const std::string & str, float weight)
{
    int i = parseSegmentNum(str);
    if (i == -1) {
        strings.push_back(str);
        if (weight != 1.0 || !weights.empty()) {
            if (weights.empty())
                weights.resize(size() - 1, 1.0);
            weights.push_back(weight);
            ExcAssertEqual(weights.size(), size());
        }
    }
    else add(i, weight);
}

int
SegmentList::
parseSegmentNum(const std::string & str)
{
    if (str.empty()) return -1;

    else if (str.length() == 1 && str[0] == '0') {
        return 0;
    }
    else if (str[0] != '0' && isdigit(str[0])) {
        char * endptr = const_cast<char *>(str.c_str() + str.length());
        long i = strtol(str.c_str(), &endptr, 10);
        if (endptr == str.c_str() + str.length()) {
            if (i < 0) return -1;
            return i;
        }
    }

    return -1;
}

void
SegmentList::
sort()
{
    if (weights.empty()) {
        std::sort(ints.begin(), ints.end());
        std::sort(strings.begin(), strings.end());
    }
    else {
        ExcAssertEqual(weights.size(), size());
        vector<pair<int, float> > isorted(ints.size());
        for (unsigned i = 0;  i < ints.size();  ++i)
            isorted[i] = make_pair(ints[i], weights[i]);
        std::sort(isorted.begin(), isorted.end());

        vector<pair<string, float> > ssorted(strings.size());
        for (unsigned i = 0;  i < strings.size();  ++i)
            ssorted[i] = make_pair(strings[i], weights[i + ints.size()]);
        std::sort(ssorted.begin(), ssorted.end());

        for (unsigned i = 0;  i < ints.size(); ++i) {
            ints[i] = isorted[i].first;
            weights[i] = isorted[i].second;
        }
        for (unsigned i = 0;  i < strings.size(); ++i) {
            strings[i] = ssorted[i].first;
            weights[i + ints.size()] = ssorted[i].second;
        }
    }
}

void
SegmentList::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 0;
    store << version << ints << strings << weights;
}

void
SegmentList::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version > 0)
        throw ML::Exception("unknown SegmentList version");
    store >> ints >> strings >> weights;
}

std::string
SegmentList::
serializeToString() const
{
    return ML::DB::serializeToString(*this);
}

SegmentList
SegmentList::
reconstituteFromString(const std::string & str)
{
    return ML::DB::reconstituteFromString<SegmentList>(str);
}

void
SegmentList::
forEach(const std::function<void (int, string, float)> & onSegment) const
{
    for (unsigned i = 0;  i < ints.size();  ++i)
        onSegment(ints[i], ML::format("%d", ints[i]),
                  weights.empty() ? 1.0 : weights[i]);
    for (unsigned i = 0;  i < strings.size();  ++i)
        onSegment(-1, strings[i],
                  weights.empty() ? 1.0 : weights[i + ints.size()]);
}


/*****************************************************************************/
/* SEGMENTS BY SOURCE                                                        */
/*****************************************************************************/

SegmentsBySource::
SegmentsBySource()
{
}

SegmentsBySource::
SegmentsBySource(SegmentsBySourceBase && other)
    : SegmentsBySourceBase(other)
{
}

SegmentsBySource::
SegmentsBySource(const SegmentsBySourceBase & other)
    : SegmentsBySourceBase(other)
{
}

void
SegmentsBySource::
sortAll()
{
    for (auto it = begin(), end = this->end();
         it != end;  ++it)
        it->second->sort();
}

const SegmentList &
SegmentsBySource::
get(const std::string & str) const
{
    static const SegmentList NONE;
    
    auto it = find(str);
    if (it == end()) return NONE;
    if (!it->second)
        throw ML::Exception("invalid segment list in segments");
    return *it->second;
}

void
SegmentsBySource::
addSegment(const std::string & source,
           const std::shared_ptr<SegmentList> & segs)
{
    if (!insert(make_pair(source, segs)).second)
        throw ML::Exception("attempt to add same segments twice");
}

void
SegmentsBySource::
addInts(const std::string & source,
        const std::vector<int> & segs)
{
    if (!insert(make_pair(source, std::make_shared<SegmentList>(segs))).second)
        throw ML::Exception("attempt to add same segments twice");
}

void
SegmentsBySource::
addStrings(const std::string & source,
           const std::vector<string> & segs)
{
    if (!insert(make_pair(source, std::make_shared<SegmentList>(segs))).second)
        throw ML::Exception("attempt to add same segments twice");
}

void
SegmentsBySource::
addWeightedInts(const std::string & source,
                const std::vector<pair<int, float> > & segs)
{
    if (!insert(make_pair(source, std::make_shared<SegmentList>(segs))).second)
        throw ML::Exception("attempt to add same segments twice");
}

void
SegmentsBySource::
add(const std::string & source, const std::string & segment, float weight)
{
    auto & entry = (*this)[source];
    if (!entry) entry.reset(new SegmentList());
    entry->add(segment, weight);
}

void
SegmentsBySource::
add(const std::string & source, int segment, float weight)
{
    auto & entry = (*this)[source];
    if (!entry) entry.reset(new SegmentList());
    entry->add(segment, weight);
}

Json::Value
SegmentsBySource::
toJson() const
{
    Json::Value result;
    for (auto it = begin(), end = this->end();  it != end;  ++it)
        result[it->first] = it->second->toJson();
    return result;
}

SegmentsBySource
SegmentsBySource::
createFromJson(const Json::Value & json)
{
    SegmentsBySource result;

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it->isNull()) continue;
        auto segs = std::make_shared<SegmentList>();
        *segs = SegmentList::createFromJson(*it);
        result.addSegment(it.memberName(), segs);
    }
    
    return result;
}

void
SegmentsBySource::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 0;
    store << version;
    store << compact_size_t(size());
    for (auto it = begin(), end = this->end();  it != end;  ++it) {
        store << it->first;
        it->second->serialize(store);
    }
}

void
SegmentsBySource::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("invalid version");
    compact_size_t sz(store);
    
    SegmentsBySourceBase newMe;
    
    for (unsigned i = 0;  i < sz;  ++i) {
        string k;
        store >> k;
        auto l = std::make_shared<SegmentList>();
        store >> *l;
        newMe[k] = l;
    }
    
    swap(newMe);
}

} // namespace RTBKIT

