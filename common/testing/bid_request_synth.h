/** bid_request_synth.h                                 -*- C++ -*-
    Rémi Attab, 25 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Bid request synthetizer.

*/

#pragma once

#include <memory>
#include <istream>
#include <ostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <string>

namespace Json { struct Value; }
namespace ML { struct RNG; }

namespace RTBKIT {

namespace Synth {

struct Node;

struct NodePath : public std::vector<std::string>
{
    // Make all the other vector constructors available.
    template<typename... Args>
    NodePath(Args&&... args) :
        std::vector<std::string>(std::forward<Args>(args)...)
    {}

    NodePath(const std::initializer_list<std::string>& list) :
        std::vector<std::string>(list)
    {}

    bool operator== (const NodePath& other) const
    {
        return
            size() == other.size() &&
            std::equal(begin(), end(), other.begin());
    }

    bool operator!= (const NodePath& other) const
    {
        return ! operator==(other);
    }
};

typedef std::function<Json::Value(const NodePath&)> GeneratorFn;
typedef std::function<bool(const NodePath&)> TestPathFn;

static constexpr const char* ArrayIndex = "_i_";
}

/******************************************************************************/
/* BID REQUEST SYNTH                                                          */
/******************************************************************************/

struct BidRequestSynth
{
    BidRequestSynth();

    Synth::TestPathFn isGeneratedFn;
    Synth::TestPathFn isCutoffFn;
    Synth::GeneratorFn generatorFn;

    void record(const Json::Value& json);
    Json::Value generate(uint32_t seed = 0) const;
    Json::Value generate(ML::RNG& rng) const;

    void dump(std::ostream& stream);
    void load(std::istream& stream);

private:
    std::shared_ptr<Synth::Node> values;
};


} // namespace RTBKIT
