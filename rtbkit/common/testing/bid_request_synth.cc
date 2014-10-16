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
    size_t target = rng.random(max + 1);
    size_t sum = 0;

    for (const auto& it : map)
        if ((sum += it.second) >= target)
            return it.first;

    ExcAssert(false);
    return T();
}


/******************************************************************************/
/* CONTEXT                                                                    */
/******************************************************************************/

struct Context
{
    void enter(const std::string& key) { path_.push_back(key); }
    void exit() { path_.pop_back(); }
    const NodePath& path() { return path_; }

private:
    NodePath path_;
};

struct GenerateCtx : public Context
{
    GenerateCtx(RNG& rng) : rng(rng) {}

    RNG& rng;
    GeneratorFn generator;
};

struct RecordCtx : public Context
{
    TestPathFn isGenerated;
    TestPathFn isCutoff;
};


template<typename Ctx>
struct CtxGuard
{
    CtxGuard(Ctx& ctx, const std::string& key) : ctx(ctx)
    {
        ctx.enter(key);
    }

    ~CtxGuard() { ctx.exit(); }

private:
    Ctx& ctx;
};


/******************************************************************************/
/* NODE TYPE                                                                  */
/******************************************************************************/

enum NodeType
{
    Null,

    Generated,
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

NodeType getType(const Json::Value& json)
{
    switch (json.type()) {
    case Json::objectValue:  return Object;
    case Json::arrayValue:   return Array;

    case Json::booleanValue: return Bool;
    case Json::intValue:
    case Json::uintValue:    return Integer;
    case Json::realValue:    return Float;
    case Json::stringValue:  return String;
    case Json::nullValue:    // We treat nulls as json blobs.
    default:                 return Json;
    }
}

struct NodeTypeHash
{
    size_t operator() (NodeType type) const
    {
        return std::hash<int>()(type);
    }
};

/******************************************************************************/
/* PROTOTYPES                                                                 */
/******************************************************************************/

struct Node;
void recordNode(RecordCtx& ctx, Node* & node, const Json::Value& value);
Node* recordNode(RecordCtx& ctx, const Json::Value& value);
Node* loadNode(Parse_Context& ctx);
void dumpNode(Node* node, ostream& stream);


/******************************************************************************/
/* NODE                                                                       */
/******************************************************************************/

struct Node
{
    NodeType type;
    size_t count;

    explicit Node(size_t count = 0) : type(Null), count(count) {}
    Node(NodeType type, size_t count) : type(type), count(count) {}

    virtual void record(RecordCtx& ctx, const Json::Value& json) = 0;
    virtual Json::Value generate(GenerateCtx& ctx) const = 0;
    virtual void dump(ostream& stream) const = 0;
    virtual void load(Parse_Context& ctx) = 0;

    virtual ~Node() {}
};



/******************************************************************************/
/* NODE LEAF                                                                  */
/******************************************************************************/

struct NodeLeaf : public Node
{
    unordered_map<string, size_t> values;

    explicit NodeLeaf(NodeType type, size_t count = 0) : Node(type, count) {}

    virtual void record(RecordCtx& ctx, const Json::Value& json)
    {
        if (type != Json)
            ExcCheckEqual(type, getType(json), "Mixing json types");

        count++;
        string value = json.toString();

        if (type == String) value = value.substr(1, value.size() - 3);
        else value.erase(value.end() - 1);

        values[value]++;
        if (debug) cerr << "    lef.rec: " << value << endl;
    }

    Json::Value generate(GenerateCtx& ctx) const
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

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                    if (debug) cerr << "        values: " << field << endl;
                    values[field] = ctx.expect_unsigned_long();
                });
    }

};


/******************************************************************************/
/* NODE GENERATED                                                             */
/******************************************************************************/

struct NodeGenerated : public Node
{
    NodeGenerated(size_t count = 0) : Node(Generated, count) {}

    void record(RecordCtx&, const Json::Value&)
    {
        if (debug) cerr << "    gen.rec: " << endl;

        count++;
    }

    Json::Value generate(GenerateCtx& ctx) const
    {
        if (debug) cerr << "    gen.gen: " << endl;

        ExcCheck(ctx.generator, "can't generate a value without a generator");
        return ctx.generator(ctx.path());
    }

    void dump(ostream& stream) const { stream << "null"; }
    void load(Parse_Context& ctx) { ctx.expect_literal("null"); }
};


/******************************************************************************/
/* NODE OBJECT                                                                */
/******************************************************************************/

struct NodeObject : public Node
{
    unordered_map<string, Node*> fields;

    explicit NodeObject(size_t count = 0) : Node(Object, count) {}

    ~NodeObject()
    {
        for (auto& field: fields)
            delete field.second;
    }

    void record(RecordCtx& ctx, const Json::Value& json)
    {
        count++;

        const auto& members = json.getMemberNames();

        for (const string& member : members) {
            CtxGuard<RecordCtx> guard(ctx, member);

            const Json::Value& value = json[member];
            if (debug) cerr << "    obj.rec: " << member << endl;

            recordNode(ctx, fields[member], value);
        }
    }

    Json::Value generate(GenerateCtx& ctx) const
    {
        Json::Value json;

        for (const auto& field : fields) {
            double p = double(field.second->count) / double(count);
            if (p < ctx.rng.random01()) continue;

            CtxGuard<GenerateCtx> guard(ctx, field.first);

            Json::Value value = field.second->generate(ctx);
            if (debug) cerr << "    obj.gen: " << field.first << " -> " << value.toString();
            if (!value.isNull()) json[field.first] = value;
        }

        return json;
    }

    void dump(ostream& stream) const
    {
        stream << "{";

        bool addSep = false;
        for (const auto& field : fields) {
            if (addSep) stream << ',';
            addSep = true;

            stream << "\"" << field.first << "\":";
            dumpNode(field.second, stream);
        }

        stream << '}';
    }

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                    if (debug) cerr << "        load: " << field << endl;
                    fields[field] = loadNode(ctx);
                });

        if (debug) {
            cerr << "        => " << this << " { ";
            for (const auto& f : fields) cerr << f.first << " ";
            cerr << "}" << endl;
        }
    }

};


/******************************************************************************/
/* NODE ARRAY                                                                 */
/******************************************************************************/

struct NodeArray : public Node
{
    NodeArray(size_t count) : Node(Array, count) {}

    void record(RecordCtx& ctx, const Json::Value& json)
    {
        count++;
        sizes[json.size()]++;

        CtxGuard<RecordCtx> guard(ctx, ArrayIndex);

        for (size_t i = 0; i < json.size(); ++i) {
            const Json::Value& value = json[i];
            NodeType type = getType(value);
            recordNode(ctx, values[type], value);
        }
    }

    Json::Value generate(GenerateCtx& ctx) const
    {
        Json::Value value;
        size_t size = pickRandom(sizes, count, ctx.rng);

        CtxGuard<GenerateCtx> guard(ctx, ArrayIndex);

        for (size_t i = 0; i < size; ++i) {
            Json::Value json = generateValue(ctx);
            if (!json.isNull()) value.append(json);
        }

        return value;
    }

    void dump(ostream& stream) const
    {
        stream << '{';

        stream << "\"sizes\":";
        dumpSizes(stream);

        stream << ",\"values\":";
        dumpValues(stream);

        stream << '}';
    }

    void load(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
                {
                    if (debug) cerr << "    arr.load: " << field << endl;

                    if      (field == "sizes")  loadSizes(ctx);
                    else if (field == "values") loadValues(ctx);

                    else ExcCheck(false, "Unknown NodeArray field: " + field);
                });
    }

private:

    Json::Value generateValue(GenerateCtx& ctx) const
    {
        size_t target = ctx.rng.random(count + 1);
        size_t sum = 0;

        for (const auto& it: values) {
            if ((sum += it.second->count) >= target)
                return it.second->generate(ctx);
        }

        ExcAssert(false);
    }

    void dumpSizes(ostream& stream) const
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

    void loadSizes(Parse_Context& ctx)
    {
        expectJsonObject(ctx, [&] (string field, Parse_Context& ctx) {
                    sizes[stoull(field)] = ctx.expect_unsigned_long();
                });
    }

    void dumpValues(ostream& stream) const
    {
        stream << '[';

        bool addSep = false;
        for (const auto& value: values) {
            if (addSep) stream << ',';
            else addSep = true;

            dumpNode(value.second, stream);
        }
        stream << ']';
    }

    void loadValues(Parse_Context& ctx)
    {
        expectJsonArray(ctx, [&] (int, Parse_Context& ctx) {
                    std::unique_ptr<Node> node(loadNode(ctx));
                    ExcAssert(!values.count(node->type));
                    values[node->type] = node.release();
                });
    }

    unordered_map<size_t, size_t> sizes;
    unordered_map<NodeType, Node*, NodeTypeHash> values;
};


/******************************************************************************/
/* UTILS IMPL                                                                 */
/******************************************************************************/

void recordNode(RecordCtx& ctx, Node* & node, const Json::Value& value)
{
    if (node) node->record(ctx, value);
    else node = recordNode(ctx, value);
}

Node* recordNode(RecordCtx& ctx, const Json::Value& value)
{
    std::unique_ptr<Node> node;

    if (ctx.isCutoff && ctx.isCutoff(ctx.path()))
        node.reset(new NodeLeaf(Json, 0));

    else if (ctx.isGenerated && ctx.isGenerated(ctx.path()))
        node.reset(new NodeGenerated());

    else {
        switch (getType(value)) {
        case Object: node.reset(new NodeObject(0)); break;
        case Array:  node.reset(new NodeArray(0)); break;
        default:     node.reset(new NodeLeaf(getType(value), 0)); break;
        }
    }

    node->record(ctx, value);
    return node.release();
}


Node* loadNode(Parse_Context& ctx)
{
    NodeType type = Null;
    size_t count = 0;
    unique_ptr<Node> node;

    expectJsonObject(ctx, [&] (string field, Parse_Context& ctx)
            {
                if (debug) cerr << "    nod.load: " << field << endl;

                if      (field == "type") type = NodeType(ctx.expect_int());
                else if (field == "count") count = ctx.expect_unsigned_long();
                else if (field == "node") {
                    switch (type) {
                    case Generated: node.reset(new NodeGenerated(count)); break;
                    case Object:    node.reset(new NodeObject(count)); break;
                    case Array:     node.reset(new NodeArray(count)); break;
                    default:        node.reset(new NodeLeaf(type, count)); break;
                    }
                    node->load(ctx);
                }

                else ExcCheck(false, "Unknown Node field: " + field);
            });

    return node.release();
}

void dumpNode(Node* node, ostream& stream)
{
    stream << "{"
        << "\"type\":" << int(node->type)
        << ",\"count\":" << node->count
        << ",\"node\":";
    node->dump(stream);
    stream << "}";
}

} // namepsace Synth


/******************************************************************************/
/* BID REQUEST SYNTH                                                          */
/******************************************************************************/

BidRequestSynth::
BidRequestSynth() :
    values(new Synth::NodeObject(0))
{}

void
BidRequestSynth::
record(const Json::Value& json)
{
    if (Synth::debug) cerr << "RECORD:" << endl;

    Synth::RecordCtx ctx;
    ctx.isGenerated = isGeneratedFn;
    ctx.isCutoff = isCutoffFn;

    values->record(ctx, json);
}

Json::Value
BidRequestSynth::
generate(uint32_t seed) const
{
    if (Synth::debug) cerr << "GENERATE:" << endl;

    ML::RNG rng(seed);
    Synth::GenerateCtx ctx(rng);
    ctx.generator = generatorFn;

    return values->generate(ctx);
}

Json::Value
BidRequestSynth::
generate(RNG& rng) const
{
    if (Synth::debug) cerr << "GENERATE:" << endl;

    Synth::GenerateCtx ctx(rng);
    ctx.generator = generatorFn;

    return values->generate(ctx);
}

void
BidRequestSynth::
dump(std::ostream& stream)
{
    if (Synth::debug) cerr << "DUMP:" << endl;
    Synth::dumpNode(values.get(), stream);
}

void
BidRequestSynth::
load(std::istream& stream)
{
    if (Synth::debug) cerr << "LOAD:" << endl;

    Parse_Context ctx("", stream);
    values.reset(Synth::loadNode(ctx));
}


} // namespace RTBKIT
