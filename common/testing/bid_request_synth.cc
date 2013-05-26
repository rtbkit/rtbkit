/** bid_request_synth.cc                                 -*- C++ -*-
    Rémi Attab, 25 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Bid request synthetizer.

    \todo The only no pointer thing in NodeObject and NodeArray didn't quite
    work out (a class can't contain an in instance of itself) so we should
    probably go back to the original vtable setup. Will make load() considerably
    more complicated to write though.

*/

#include "bid_request_synth.h"
#include "soa/jsoncpp/value.h"
#include "soa/jsoncpp/reader.h"
#include "jml/utils/rng.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/json_parsing.h"

#include <unordered_map>

using namespace std;
using namespace ML;

namespace RTBKIT {
namespace Synth {

struct NodeLeaf;
struct nodeArray;
struct NodeObject;

enum { debug = false };

/******************************************************************************/
/* UTILITIES                                                                  */
/******************************************************************************/

template<typename T>
T pickRandom(const unordered_map<T, size_t>& map, size_t max, RNG& rng)
{
    T value;
    size_t target = rng.random(max + 1);
    size_t sum = 0;

    for (const auto& it : map)
        if ((sum += it.second) >= target)
            return it.first;

    ExcAssert(false);
    return value;
}

template<typename T>
T& get(T* & p)
{
    if (!p) p = new T;
    return *p;
}

template<typename T>
T& get(unique_ptr<T>& p)
{
    if (!p) p.reset(new T);
    return *p;
}

template<typename T>
const T& get(T* const & p)
{
    ExcAssert(p);
    return *p;
}

template<typename T>
const T& get(const unique_ptr<T>& p)
{
    ExcAssert(p);
    return *p;
}

template<typename K, typename V>
V& get(unordered_map<K,V*>& map, const K& key)
{
    auto it = map.find(key);
    if (it == map.end())
        it = map.insert(make_pair(key, new V())).first;
    return *it->second;
}


template<typename K, typename V>
const V& get(const unordered_map<K,V*>& map, const K& key)
{
    auto it = map.find(key);
    ExcAssert(it != map.end());
    return *it->second;
}



/******************************************************************************/
/* GEN CONTEXT                                                                */
/******************************************************************************/

struct GenContext
{
    RNG rng;
};


/******************************************************************************/
/* NODE TYPE                                                                  */
/******************************************************************************/

enum NodeType
{
    Null,

    Object,
    Array,

    Bool,
    Integer,
    Float,
    String,
    Json,
};

void print(NodeType type)
{
    if (type == Null)         cerr << "null";

    else if (type == Object)  cerr << "object";
    else if (type == Array)   cerr << "array";

    else if (type == Bool)    cerr << "bool";
    else if (type == Integer) cerr << "integer";
    else if (type == Float)   cerr << "float";
    else if (type == String)  cerr << "string";
    else if (type == Json)    cerr << "json";

    else ExcAssert(false);
}


/******************************************************************************/
/* NODE                                                                       */
/******************************************************************************/

struct Node
{
    NodeType type;
    size_t count;

    Node() : type(Null), count(0) {}

    void record(NodeType newType)
    {
        // We assume that once set, the type of a node can't change.
        ExcAssert(type == Null || type == newType);
        type = newType;
        count++;
    }

    void dump(ostream& stream) const
    {
        stream << "{"
            << "\"type\":" << int(type)
            << ",\"count\":" << count
            << "}";
    }

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
                {
                    if (debug) cerr << "    nod.load: " << field << endl;

                    if      (field == "type")  type = NodeType(ctx.expect_int());
                    else if (field == "count") count = ctx.expect_unsigned_long();

                    else ExcCheck(false, "Unknown Node field: " + field);
                });
    }
};


/******************************************************************************/
/* NODE LEAF                                                                  */
/******************************************************************************/

struct NodeLeaf : public Node
{
    unordered_map<string, size_t> values;

    void record(const Json::Value& json)
    {
        switch (json.type()) {
        case Json::booleanValue: Node::record(Bool); break;
        case Json::intValue:
        case Json::uintValue:    Node::record(Integer); break;
        case Json::realValue:    Node::record(Float); break;
        case Json::stringValue:  Node::record(String); break;
        case Json::nullValue:    // We treat nulls as json blobs.
        default:                 Node::record(Json); break;
        }

        string value = json.toString();

        if (type == String) value = value.substr(1, value.size() - 3);
        else value.erase(value.end() - 1);

        values[value]++;
        if (debug) cerr << "    lef.rec: " << value << endl;
    }

    Json::Value generate(GenContext& ctx) const
    {
        string value = pickRandom(values, count, ctx.rng);

        if (debug) cerr << "    lef.gen: " << value << endl;

        switch (type) {
        case Null:    return Json::Value();
        case Bool:    return Json::Value(value == "true");
        case Integer: return Json::Value(stoll(value));
        case Float:   return Json::Value(stod(value));
        case String:  return Json::Value(value);
        case Json:    return Json::parse(value);
        default: break;
        }

        ExcAssert(false);
        return Json::Value();
    }

    void dump(ostream& stream) const
    {
        stream << '{';

        stream << "\"node\":";
        Node::dump(stream);

        stream << ",\"values\":";
        dumpValues(stream);

        stream << '}';
    }

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
                {
                    if (debug) cerr << "    lef.load: " << field << endl;

                    if      (field == "node")   Node::load(ctx);
                    else if (field == "values") loadValues(ctx);

                    else ExcCheck(false, "Unknown NodeLeaf field: " + field);
                });
    }

private:

    void dumpValues(ostream& stream) const
    {
        stream << '{';

        bool addSep = false;
        for (const auto& val : values) {
            if (addSep) stream << ',';
            addSep = true;

            // Hopefully the parser can handle complex keys
            stream << '"' << jsonEscape(val.first) << "\":"
                << to_string(val.second);
        }

        stream << '}';
    }

    void loadValues(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                    if (debug) cerr << "        values: " << field << endl;
                    values[field] = ctx.expect_unsigned_long();
                });
    }

};


/******************************************************************************/
/* NODE ARRAY                                                                 */
/******************************************************************************/

struct NodeArray : public Node
{
    unordered_map<size_t, size_t> sizes;

    unique_ptr<NodeObject> objects;
    unique_ptr<NodeArray> arrays;
    unique_ptr<NodeLeaf> leafs;

    void record(const Json::Value& json);
    Json::Value generate(GenContext& ctx) const;

    void dump(ostream& stream) const;
    void load(Parse_Context& ctx);

private:

    void check();

    template<typename T>
    Json::Value generate(size_t size, const T& values, GenContext& ctx) const;

    void dumpSizes(ostream& stream) const;
    void loadSizes(Parse_Context& ctx);

    template<typename T>
    void dump(ostream& stream, const string& name, const T& values) const;
};


/******************************************************************************/
/* NODE OBJECT                                                                */
/******************************************************************************/

struct NodeObject : public Node
{
    unordered_map<string, NodeObject*> objects;
    unordered_map<string, NodeArray*> arrays;
    unordered_map<string, NodeLeaf*> leafs;

    ~NodeObject()
    {
        destroy(objects);
        destroy(arrays);
        destroy(leafs);
    }

    void record(const Json::Value& json)
    {
        Node::record(Object);

        const auto& fields = json.getMemberNames();
        for (const string& field : fields) {
            const Json::Value& value = json[field];

            if (value.type() == Json::objectValue)
                get(objects, field).record(value);

            else if (value.type() == Json::arrayValue)
                get(arrays, field).record(value);

            else get(leafs, field).record(value);
        }
    }

    Json::Value generate(GenContext& ctx) const
    {
        Json::Value value;

        generate(value, objects, ctx);
        generate(value, arrays, ctx);
        generate(value, leafs, ctx);

        return value;
    }

    void dump(ostream& stream) const
    {

        stream << '{';

        stream << "\"node\":";
        Node::dump(stream);

        if (objects.size()) dump(stream, "objects", objects);
        if (arrays.size())  dump(stream, "arrays", arrays);
        if (leafs.size())   dump(stream, "leafs", leafs);

        stream << '}';
    }

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
                {
                    if (debug) cerr << "    obj.load: " << field << endl;

                    if      (field == "node")    Node::load(ctx);
                    else if (field == "objects") load(ctx, objects);
                    else if (field == "arrays")  load(ctx, arrays);
                    else if (field == "leafs")   load(ctx, leafs);

                    else ExcCheck(false, "Unknown NodeObject field: " + field);
                });

        if (debug) {
            cerr << "    => " << this << " { ";
            for (const auto& f : leafs) cerr << f.first << " ";
            cerr << "}" << endl;
        }

    }

private:

    template<typename T>
    void destroy(unordered_map<string, T>& values)
    {
        for (auto& val : values) {
            delete val.second;
            val.second = nullptr;
        }
        values.clear();
    }

    template<typename T>
    void generate(
            Json::Value& value,
            const unordered_map<string, T> fields,
            GenContext& ctx) const
    {
        for (const auto& field : fields) {
            double p = double(get(field.second).count) / double(count);
            if (p < ctx.rng.random01()) continue;

            Json::Value val = get(field.second).generate(ctx);
            if (debug) cerr << "    obj.gen: " << field.first << " -> " << val.toString();
            if (!val.isNull()) value[field.first] = val;
        }
    }

    template<typename T>
    void dump(
            ostream& stream,
            const string& name,
            const unordered_map<string, T> fields) const
    {
        stream << ",\"" << name << "\":{";

        bool addSep = false;
        for (const auto& field : fields) {
            if (addSep) stream << ',';
            addSep = true;

            stream << "\"" << field.first << "\":";
            get(field.second).dump(stream);
        }

        stream << '}';
    }

    template<typename T>
    void load(Parse_Context& ctx, unordered_map<string, T>& fields) const
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                    if (debug) cerr << "        load: " << field << endl;
                    get(fields, field).load(ctx);
                });

        if (debug) {
            cerr << "        => " << this << " { ";
            for (const auto& f : fields) cerr << f.first << " ";
            cerr << "}" << endl;
        }
    }

};


/******************************************************************************/
/* NODE ARRAY IMPL                                                            */
/******************************************************************************/

void
NodeArray::
record(const Json::Value& json)
{
    Node::record(Array);
    sizes[json.size()]++;

    for (size_t i = 0; i < json.size(); ++i) {
        const Json::Value& value = json[i];

        if (value.type() == Json::objectValue)
            get(objects).record(value);

        else if (value.type() == Json::arrayValue)
            get(arrays).record(value);

        else get(leafs).record(value);
    }

    check();
}

Json::Value
NodeArray::
generate(GenContext& ctx) const
{
    size_t size = pickRandom(sizes, count, ctx.rng);

    if (objects) return generate(size, get(objects), ctx);
    if (arrays)  return generate(size, get(arrays), ctx);
    if (leafs)   return generate(size, get(leafs), ctx);

    ExcAssert(false);
    return Json::Value();
}


void
NodeArray::
dump(ostream& stream) const
{
    stream << '{';

    stream << "\"node\":";
    Node::dump(stream);

    stream << ",\"sizes\":";
    dumpSizes(stream);

    if      (objects) dump(stream, "objects", get(objects));
    else if (arrays)  dump(stream, "arrays", get(arrays));
    else if (leafs)   dump(stream, "leafs", get(leafs));

    stream << '}';
}

void
NodeArray::
load(Parse_Context& ctx)
{
    expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
            {
                if (debug) cerr << "    arr.load: " << field << endl;

                if      (field == "node")    Node::load(ctx);
                else if (field == "sizes")   loadSizes(ctx);
                else if (field == "objects") get(objects).load(ctx);
                else if (field == "arrays")  get(arrays).load(ctx);
                else if (field == "leafs")   get(leafs).load(ctx);

                else ExcCheck(false, "Unknown NodeArray field: " + field);
            });
    check();
}

void
NodeArray::
check()
{
    // We're assuming that there's only one type of element in the array.
    ExcAssert(
            ( objects && !arrays && !leafs) ||
            (!objects &&  arrays && !leafs) ||
            (!objects && !arrays &&  leafs));
}

template<typename T>
Json::Value
NodeArray::
generate(size_t size, const T& values, GenContext& ctx) const
{
    Json::Value array;

    for (size_t i = 0; i < size; ++i) {
        Json::Value val = values.generate(ctx);
        if (!val.isNull()) array.append(val);
    }

    return array;
}

void
NodeArray::
dumpSizes(ostream& stream) const
{
    stream << '{';

    bool addSep = false;
    for (const auto& it : sizes) {
        if (addSep) stream << ',';
        else addSep = true;

        stream << "\"" << it.first << "\":" << to_string(it.second);
    }
    stream << '}';
}

void
NodeArray::
loadSizes(Parse_Context& ctx)
{
    expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                sizes[stoull(field)] = ctx.expect_unsigned_long();
            });
}

template<typename T>
void
NodeArray::
dump(ostream& stream, const string& name, const T& values) const
{
    stream << ",\"" << name << "\":";
    values.dump(stream);
}


} // namepsace Synth


/******************************************************************************/
/* BID REQUEST SYNTH                                                          */
/******************************************************************************/

BidRequestSynth::
BidRequestSynth() :
    values(make_shared<Synth::NodeObject>())
{}

void
BidRequestSynth::
record(const Json::Value& json)
{
    if (Synth::debug) cerr << "RECORD:" << endl;
    values->record(json);
}

Json::Value
BidRequestSynth::
generate() const
{
    if (Synth::debug) cerr << "GENERATE:" << endl;

    Synth::GenContext ctx;
    return values->generate(ctx);
}

void
BidRequestSynth::
dump(std::ostream& stream)
{
    values->dump(stream);
}

void
BidRequestSynth::
load(std::istream& stream)
{
    if (Synth::debug) cerr << "LOAD:" << endl;

    auto newValues = make_shared<Synth::NodeObject>();

    Parse_Context ctx("", stream);
    newValues->load(ctx);

    values.swap(newValues);
}


} // namespace RTBKIT
