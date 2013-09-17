/** filter_test.cc                                 -*- C++ -*-
    Rémi Attab, 16 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests the for common filter utilities.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/filter.h"
#include "rtbkit/common/exchange_connector.h"
#include "rtbkit/common/bid_request.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace RTBKIT;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE(configSetTest)
{
    enum { n = 200 };

    {
        ConfigSet set;

        BOOST_CHECK_EQUAL(set.size(), 0);
        BOOST_CHECK(set.empty());
        BOOST_CHECK_EQUAL(set.count(), 0);

        for (size_t i = 0; i < n; ++i) {
            set.set(i);
            BOOST_CHECK(set.test(i));
            BOOST_CHECK_LE(i, set.size());
            BOOST_CHECK_EQUAL(set.count(), 1);
            set.reset(i);
        }

        BOOST_CHECK(set.empty());
        BOOST_CHECK_EQUAL(set.count(), 0);
    }

    {
        ConfigSet set;
        for (size_t i = 0; i < n; ++i) {
            ConfigSet mask;
            mask.set(i);
            set |= mask;

            for (size_t j = 0; j <= i; ++j)
                BOOST_CHECK(set.test(j));

            for (size_t j = i + 1; j < n; ++j)
                BOOST_CHECK(!set.test(j));
        }
    }

    {
        ConfigSet set(true);

        for (size_t i = 0; i < n; ++i) {
            ConfigSet mask;
            mask.set(i);
            set &= mask.negate();

            for (size_t j = 0; j <= i; ++j)
                BOOST_CHECK(!set.test(j));

            for (size_t j = i + 1; j < n; ++j)
                BOOST_CHECK(set.test(j));

        }
    }

    {
        ConfigSet setA, setB;

        for (size_t i = 0; i < n; ++i) {
            setA.set(i);
            setB.set(i+n);
        }

        ConfigSet resultA = setA;
        resultA |= setB;

        ConfigSet resultB = setB;
        resultB |= setA;

        BOOST_CHECK_EQUAL(resultA.size(), resultB.size());
        BOOST_CHECK_EQUAL(resultA.count(), resultB.count());

        for (size_t i = 0; i < n*2; ++i) {
            BOOST_CHECK(resultA.test(i));
            BOOST_CHECK(resultB.test(i));
        }
    }

    {
        for (size_t i = 1; i < 64; ++i) {
            ConfigSet set;

            for (size_t j = 0; j < n; j += i)
                set.set(j);

            for (size_t j = set.next(), k = 0;
                 j < set.size();
                 j = set.next(j+1), k++)
            {
                BOOST_CHECK_EQUAL(j, k*i);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(creativeMatrixTest)
{
    enum { n = 10, m = 100 };


    {
        CreativeMatrix matrix;

        BOOST_CHECK(matrix.empty());
        BOOST_CHECK_EQUAL(matrix.size(), 0);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                BOOST_CHECK(!matrix.test(i,j));
                matrix.set(i,j);
                BOOST_CHECK(matrix.test(i,j));
                matrix.reset(i,j);
                BOOST_CHECK(!matrix.test(i,j));
                BOOST_CHECK(matrix.empty());
            }
        }

        BOOST_CHECK(matrix.empty());
    }

    {
        CreativeMatrix matrix;

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                BOOST_CHECK(!matrix.test(i,j));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                CreativeMatrix mask;
                mask.set(i,j);
                matrix |= mask;
                BOOST_CHECK(matrix.test(i,j));
            }
        }

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                BOOST_CHECK(matrix.test(i,j));
    }

    {
        CreativeMatrix matrix(true);

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                BOOST_CHECK(matrix.test(i,j));

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                CreativeMatrix mask;
                mask.set(i,j);
                matrix &= mask.negate();
                BOOST_CHECK(!matrix.test(i,j));
            }
        }

        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < m; ++j)
                BOOST_CHECK(!matrix.test(i,j));
    }
}

vector<CreativeMatrix>
toMatrix(const unordered_map<unsigned, BiddableSpots>& spots)
{
    vector<CreativeMatrix> creatives;

    for (const auto& configs : spots) {
        for (const auto& spot : configs.second) {
            if (spot.first >= creatives.size())
                creatives.resize(spot.first + 1);

            for (const auto& creative : spot.second)
                creatives[spot.first].set(creative, configs.first);
        }
    }

    return creatives;
}

vector<CreativeMatrix>
getMatrix(const FilterState& state, unsigned maxImp)
{
    vector<CreativeMatrix> creatives(maxImp);

    for (size_t imp = 0; imp < maxImp; ++imp)
        creatives[imp] = state.creatives(imp);

    return creatives;
}

void checkBiddableSpots(
        const vector<CreativeMatrix>& value, const vector<CreativeMatrix>& exp)
{
    BOOST_CHECK_EQUAL(value.size(), exp.size());
    size_t n = std::min(value.size(), exp.size());

    for (size_t i = 0; i < n; ++i) {
        CreativeMatrix diff = value[i];
        diff ^= exp[i];
        BOOST_CHECK(diff.empty());

        if (!diff.empty()) {
            cerr << "=== imp=" << i << endl;
            cerr << "value=" << value[i].print() << endl;
            cerr << "  exp=" << exp[i].print() << endl;
            cerr << " diff=" << diff.print() << endl;
        }
    }
}

void checkBiddableSpots(FilterState& state)
{
    const auto& value = toMatrix(state.biddableSpots());
    const auto& exp = getMatrix(state, state.request.imp.size());
    checkBiddableSpots(value, exp);
}

BOOST_AUTO_TEST_CASE(filterStateTest)
{
    enum { spots = 10, creatives = 10, configs = 10 };

    ExchangeConnector* ex = nullptr;
    BidRequest br;
    br.imp.resize(spots);

    {
        cerr << "[basic]_______________________________________________________"
            << endl;

        CreativeMatrix activeConfigs;
        for (size_t i = 0; i < configs; ++i)
            activeConfigs.setConfig(i, creatives);

        FilterState state(br, ex, activeConfigs);
        checkBiddableSpots(state);
    }

    {
        cerr << "[diag]________________________________________________________"
            << endl;

        CreativeMatrix activeConfigs;
        for (size_t i = 0; i < configs; ++i)
            activeConfigs.setConfig(i, i);

        FilterState state(br, ex, activeConfigs);
        checkBiddableSpots(state);

        CreativeMatrix mask;
        for (size_t cfg = 0; cfg < configs; ++cfg)
            for (size_t cr = 0; cr < creatives / 2; ++cr)
                mask.set(cr, cfg);

        state.narrowAllCreatives(mask);
        checkBiddableSpots(state);
    }

    {
        cerr << "[per-imp]___________________________________________________"
            << endl;

        CreativeMatrix activeConfigs;
        for (size_t i = 0; i < configs; ++i)
            activeConfigs.setConfig(i, configs - i);

        FilterState state(br, ex, activeConfigs);

        for (size_t imp = 0; imp < spots; ++imp) {
            CreativeMatrix mask;

            for (size_t cr = 0; cr < creatives; ++cr)
                for (size_t cfg = 0; cfg < imp; ++cfg)
                    mask.set(cr, cfg);

            state.narrowCreativesForImp(imp, mask);
        }

        checkBiddableSpots(state);
    }
}
